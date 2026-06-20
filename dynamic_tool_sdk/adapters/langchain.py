"""
Dynamic Tool SDK — LangChain adapter.

Wraps a ``BaseChatModel`` (e.g. ``ChatOpenAI``) so tool lists are
auto-filtered on every call to ``bind_tools`` or ``invoke``.  The
``use_toolset`` bootstrap tool is injected as a ``StructuredTool``.

LangChain tools are ``BaseTool`` instances (not raw dicts), so this
adapter uses the tool's ``.name`` attribute rather than schema keys
when matching against the active toolsets.

Because ``BaseChatModel`` is a Pydantic model, its instances cannot
be monkey-patched directly.  ``wrap_chat_model()`` returns a thin
proxy (``_LangChainModelWrapper``) that intercepts ``bind_tools`` and
``invoke`` while delegating every other attribute to the real model.

Usage::
    from dynamic_tool_sdk.adapters.langchain import LangChainAdapter
    from langchain_openai import ChatOpenAI

    adapter = LangChainAdapter()
    model   = adapter.wrap_chat_model(ChatOpenAI(api_key="sk-..."))

    # bind_tools now filters to the active toolset + use_toolset bootstrap
    chain = model.bind_tools([tool_alpha, tool_beta]) | parser
    result = chain.invoke(messages)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..core import AdapterNotInstalled
from .base import BaseAdapter


class LangChainAdapter(BaseAdapter):
    """Adapter that wraps a LangChain ``BaseChatModel`` to filter tool lists."""

    # OpenAI-compatible dict schema — returned by get_use_toolset_schema()
    # and used as a fallback when injecting into dict-based tool lists.
    _USE_TOOLSET_SCHEMA: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "use_toolset",
            "description": (
                "Dynamically load or unload tool groups for the next turn. "
                "Actions: set (replace all), add (add to current), "
                "remove (remove from current), reset (back to core), "
                "list (show available). "
                "Examples: 'add browser', 'set terminal,web', 'reset'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "set", "load", "replace", "add", "enable",
                            "include", "remove", "disable", "drop",
                            "reset", "default", "list",
                        ],
                    },
                    "toolsets": {
                        "type": "string",
                        "description": (
                            "Comma-separated toolset names. "
                            "Omit for 'reset' and 'list'."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    }

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    @classmethod
    def _check_deps(cls) -> None:
        """Verify langchain and langchain_core are installed."""
        missing = []
        for pkg in ("langchain_core", "langchain"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg.replace("_", "-"))
        if missing:
            raise AdapterNotInstalled(
                f"{', '.join(missing)} package(s) are required for "
                "LangChainAdapter. "
                "Install with: pip install langchain langchain-core"
            )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, core_toolset: str = "core", auto_inject: bool = True):
        self._check_deps()
        super().__init__(core_toolset=core_toolset)
        self._auto_inject = auto_inject
        self._bootstrap_structured_tool: Any = None  # lazy StructuredTool

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def get_use_toolset_schema(self) -> Dict[str, Any]:
        """Return the OpenAI-compatible dict schema for use_toolset."""
        return dict(self._USE_TOOLSET_SCHEMA)

    def _get_bootstrap_tool(self) -> Any:
        """Return (lazily created) use_toolset as a LangChain StructuredTool."""
        if self._bootstrap_structured_tool is None:
            from pydantic import BaseModel, Field
            from langchain_core.tools import StructuredTool

            class _UseToolsetInput(BaseModel):
                action: str = Field(
                    description=(
                        "Toolset action: set, load, replace, add, enable, "
                        "include, remove, disable, drop, reset, default, list"
                    )
                )
                toolsets: str = Field(
                    default="",
                    description="Comma-separated toolset names (omit for reset/list).",
                )

            self._bootstrap_structured_tool = StructuredTool.from_function(
                func=lambda action, toolsets="": {
                    "action": action,
                    "toolsets": toolsets,
                },
                name="use_toolset",
                description=(
                    "Dynamically load or unload tool groups for the next turn. "
                    "Actions: set (replace all), add (add to current), "
                    "remove (remove from current), reset (back to core), "
                    "list (show available)."
                ),
                args_schema=_UseToolsetInput,
                return_direct=False,
            )
        return self._bootstrap_structured_tool

    # ------------------------------------------------------------------
    # Public API: _filter_tools  (mirrors OpenAI / Anthropic adapters)
    # ------------------------------------------------------------------

    def _filter_tools(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Public alias for _filter_and_process. Used by tests directly."""
        return self._filter_and_process(kwargs)

    # ------------------------------------------------------------------
    # Client / model wrapping
    # ------------------------------------------------------------------

    def wrap_client(self, model: Any) -> "_LangChainModelWrapper":
        """Alias for wrap_chat_model — satisfies BaseAdapter contract."""
        return self.wrap_chat_model(model)

    def wrap_chat_model(self, model: Any) -> "_LangChainModelWrapper":
        """Return a proxy that filters tools on bind_tools() and invoke().

        Because ``BaseChatModel`` is a Pydantic model, methods cannot be
        monkey-patched on the instance.  A thin proxy is returned instead;
        it intercepts ``bind_tools`` and ``invoke`` while delegating
        every other attribute to the original model via ``__getattr__``.
        """
        return _LangChainModelWrapper(model, self)

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    def _filter_and_process(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter the 'tools' key in kwargs and apply any pending changes."""
        tools = kwargs.get("tools")

        # Apply any deferred toolset change at the turn boundary
        self._filter.apply()

        if not tools:  # None or empty list
            if self._auto_inject:
                kwargs["tools"] = [self._get_bootstrap_tool()]
            return kwargs

        filtered = self._filter_tools_list(tools)

        if self._auto_inject:
            has_bootstrap = any(
                self._get_tool_name(t) == "use_toolset" for t in filtered
            )
            if not has_bootstrap:
                filtered.append(self._get_bootstrap_tool())

        kwargs["tools"] = filtered
        return kwargs

    def _filter_tools_list(self, tools: List[Any]) -> List[Any]:
        """Filter a list of BaseTool instances (or dicts) by active toolsets.

        Extracts names from registered tool schemas and keeps only those
        tools whose ``.name`` (or ``["name"]``) is in the allowed set.
        Falls through (returns all) if no schemas are registered for the
        active toolsets.
        """
        from ..core.registry import get_all_toolsets

        active: Set[str] = self._filter.state.active
        all_ts = get_all_toolsets()
        allowed: Set[str] = set()

        for ts_name in active:
            ts = all_ts.get(ts_name)
            if ts:
                for schema in ts.get("tools", []):
                    name = (
                        schema.get("name")
                        or schema.get("function", {}).get("name", "")
                    )
                    if name:
                        allowed.add(name)

        if not allowed:
            # No schemas registered for active toolsets — pass everything through
            return list(tools)

        return [t for t in tools if self._get_tool_name(t) in allowed]

    @staticmethod
    def _get_tool_name(t: Any) -> str:
        """Extract a tool's name from a BaseTool instance or dict schema."""
        if hasattr(t, "name") and not isinstance(t, dict):
            return t.name or ""
        if isinstance(t, dict):
            return (
                t.get("name")
                or t.get("function", {}).get("name", "")
                or ""
            )
        return ""

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _process_tool_calls(self, response: Any) -> None:
        """Inspect an AIMessage response for use_toolset tool calls.

        LangChain returns ``AIMessage.tool_calls`` as a list of dicts::

            [{"name": "use_toolset", "args": {"action": "add",
              "toolsets": "browser"}, "id": "...", "type": "tool_call"}]

        Both dict and object forms are handled defensively.
        """
        try:
            tool_calls = getattr(response, "tool_calls", None) or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args", {}) or {}
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", {}) or {}

                if name == "use_toolset":
                    action = args.get("action", "list")
                    toolsets = args.get("toolsets", "")
                    self._filter.use_toolset(action, toolsets)
        except (AttributeError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Proxy classes (returned by wrap_chat_model / bind_tools)
# ---------------------------------------------------------------------------

class _LangChainModelWrapper:
    """Proxy for a BaseChatModel that filters tools on bind_tools / invoke.

    Uses ``object.__setattr__`` / ``object.__getattribute__`` to avoid
    triggering Pydantic's ``__setattr__`` on the wrapped model.
    """

    def __init__(self, model: Any, adapter: LangChainAdapter) -> None:
        object.__setattr__(self, "_lc_model", model)
        object.__setattr__(self, "_lc_adapter", adapter)

    # -- Intercepted methods ------------------------------------------

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "_LangChainBoundWrapper":
        """Filter tools then delegate to the real bind_tools."""
        adapter: LangChainAdapter = object.__getattribute__(self, "_lc_adapter")
        model: Any = object.__getattribute__(self, "_lc_model")

        adapter._filter.apply()
        filtered = adapter._filter_tools_list(tools)

        if adapter._auto_inject:
            if not any(adapter._get_tool_name(t) == "use_toolset" for t in filtered):
                filtered.append(adapter._get_bootstrap_tool())

        bound = model.bind_tools(filtered, **kwargs)
        return _LangChainBoundWrapper(bound, adapter)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Delegate to the real invoke, then process tool calls."""
        adapter: LangChainAdapter = object.__getattribute__(self, "_lc_adapter")
        model: Any = object.__getattribute__(self, "_lc_model")

        adapter._filter.apply()
        response = model.invoke(input, config=config, **kwargs)
        adapter._process_tool_calls(response)
        return response

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Async invoke — delegates then processes tool calls."""
        adapter: LangChainAdapter = object.__getattribute__(self, "_lc_adapter")
        model: Any = object.__getattribute__(self, "_lc_model")

        adapter._filter.apply()
        response = await model.ainvoke(input, config=config, **kwargs)
        adapter._process_tool_calls(response)
        return response

    # -- Delegation ---------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        model = object.__getattribute__(self, "_lc_model")
        return getattr(model, name)

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_lc_model")
        return f"_LangChainModelWrapper({model!r})"


class _LangChainBoundWrapper:
    """Proxy for the Runnable returned by BaseChatModel.bind_tools.

    Intercepts ``invoke`` / ``ainvoke`` to process tool calls in the
    response, then delegates everything else to the underlying Runnable.
    """

    def __init__(self, bound: Any, adapter: LangChainAdapter) -> None:
        object.__setattr__(self, "_lc_bound", bound)
        object.__setattr__(self, "_lc_adapter", adapter)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        adapter: LangChainAdapter = object.__getattribute__(self, "_lc_adapter")
        bound: Any = object.__getattribute__(self, "_lc_bound")
        response = bound.invoke(input, config=config, **kwargs)
        adapter._process_tool_calls(response)
        return response

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        adapter: LangChainAdapter = object.__getattribute__(self, "_lc_adapter")
        bound: Any = object.__getattribute__(self, "_lc_bound")
        response = await bound.ainvoke(input, config=config, **kwargs)
        adapter._process_tool_calls(response)
        return response

    def __getattr__(self, name: str) -> Any:
        bound = object.__getattribute__(self, "_lc_bound")
        return getattr(bound, name)

    def __repr__(self) -> str:
        bound = object.__getattribute__(self, "_lc_bound")
        return f"_LangChainBoundWrapper({bound!r})"
