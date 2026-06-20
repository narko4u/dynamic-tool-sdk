"""
Dynamic Tool SDK — LiteLLM adapter.

LiteLLM routes to 100+ LLM providers through a single function call.
This adapter wraps ``litellm.completion()`` (and its async and streaming
variants) to filter tool schemas transparently.

Usage::
    from dynamic_tool_sdk.adapters.litellm import LiteLLMAdapter
    import litellm

    adapter = LiteLLMAdapter()
    adapter.wrap_module(litellm)
    # All subsequent litellm.completion() calls filter tool schemas
"""
from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Optional, TypeVar

from ..core import AdapterNotInstalled
from .base import BaseAdapter

_Fn = TypeVar("_Fn")


class LiteLLMAdapter(BaseAdapter):
    """Adapter for LiteLLM proxy — wraps completion() to filter tools."""

    _USE_TOOLSET_SCHEMA: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "use_toolset",
            "description": (
                "Dynamically load or unload tool groups for the next turn. "
                "Actions: set, add, remove, reset, list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "add", "remove", "reset", "list"],
                    },
                    "toolsets": {
                        "type": "string",
                        "description": "Comma-separated toolset names.",
                    },
                },
                "required": ["action"],
            },
        },
    }

    @classmethod
    def _check_deps(cls) -> None:
        """Verify litellm is installed."""
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise AdapterNotInstalled(
                "litellm package is required for LiteLLMAdapter. "
                "Install it with: pip install litellm"
            )

    def __init__(self, core_toolset: str = "core", auto_inject: bool = True):
        self._check_deps()
        super().__init__(core_toolset=core_toolset)
        self._auto_inject = auto_inject

    def get_use_toolset_schema(self) -> Dict[str, Any]:
        return dict(self._USE_TOOLSET_SCHEMA)

    # ------------------------------------------------------------------
    # Public API (mirrors OpenAIAdapter — used directly by tests)
    # ------------------------------------------------------------------

    def _filter_tools(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Public alias for ``_filter_and_process``."""
        return self._filter_and_process(kwargs)

    def wrap_completion(self, mod: Any = None) -> None:
        """Patch completion in the given module (default: ``litellm``)."""
        if mod is None:
            import litellm
            mod = litellm
        self.wrap_module(mod)

    def wrap_acompletion(self, mod: Any = None) -> None:
        """Patch acompletion in the given module (default: ``litellm``)."""
        if mod is None:
            import litellm
            mod = litellm
        self.wrap_module(mod)

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def wrap_module(self, mod: Any) -> Any:
        """Wrap a LiteLLM module's completion/acompletion in place."""
        if hasattr(mod, "completion"):
            mod.completion = self._wrap_completion(mod.completion)
        if hasattr(mod, "acompletion"):
            mod.acompletion = self._wrap_acompletion(mod.acompletion)
        return mod

    def wrap_client(self, client: Any) -> Any:
        """Alias for wrap_module."""
        return self.wrap_module(client)

    def _wrap_completion(self, func) -> Any:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs = self._filter_and_process(kwargs)
            response = func(*args, **kwargs)
            if not kwargs.get("stream"):
                self._process_tool_calls(response)
            return response
        return wrapped

    def _wrap_acompletion(self, func) -> Any:
        @functools.wraps(func)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs = self._filter_and_process(kwargs)
            response = await func(*args, **kwargs)
            if not kwargs.get("stream"):
                self._process_tool_calls(response)
            return response
        return wrapped

    # ------------------------------------------------------------------
    # Tool filtering + tool_call processing
    # ------------------------------------------------------------------

    def _filter_and_process(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        tools = kwargs.get("tools", [])

        self._filter.apply()

        if not tools:
            if self._auto_inject:
                kwargs["tools"] = [self._USE_TOOLSET_SCHEMA]
            return kwargs

        filtered = self._filter.filter(tools)

        if self._auto_inject:
            has_bootstrap = any(
                t.get("function", {}).get("name") == "use_toolset"
                or t.get("name") == "use_toolset"
                for t in filtered
            )
            if not has_bootstrap:
                filtered.append(self._USE_TOOLSET_SCHEMA)

        kwargs["tools"] = filtered
        return kwargs

    def _process_tool_calls(self, response: Any) -> None:
        """Inspect the response for use_toolset tool calls.

        LiteLLM returns OpenAI-compatible responses, so we reuse
        the same logic: scan choices[*].message.tool_calls.
        Handles both OpenAI response objects and plain dicts.
        """
        is_dict = isinstance(response, dict)
        try:
            choices = response.get("choices", []) if is_dict else getattr(response, "choices", []) or []
            for choice in choices:
                msg = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
                if not msg:
                    continue
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", []) or []
                for tc in tool_calls:
                    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                    if fn and (fn.get("name") if isinstance(fn, dict) else fn.name) == "use_toolset":
                        raw_args = fn.get("arguments") if isinstance(fn, dict) else fn.arguments
                        args = json.loads(raw_args)
                        action = args.get("action", "list")
                        toolsets = args.get("toolsets", "")
                        self._filter.use_toolset(action, toolsets)
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError):
            pass  # Malformed responses are silently ignored
