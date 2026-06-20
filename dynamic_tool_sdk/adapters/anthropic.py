"""
Dynamic Tool SDK — Anthropic Python client adapter.

Wraps ``anthropic.Anthropic`` and ``anthropic.AsyncAnthropic`` so tool
schemas are auto-filtered on every API call.  The ``use_toolset`` tool
declaration is injected as a bootstrap tool.

Anthropic uses ``input_schema`` instead of ``parameters``.

Usage::
    from dynamic_tool_sdk.adapters.anthropic import AnthropicAdapter
    import anthropic

    client = AnthropicAdapter().wrap_client(anthropic.Anthropic(api_key="sk-ant-..."))
"""
from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Optional, TypeVar

from ..core import AdapterNotInstalled
from .base import BaseAdapter

_T = TypeVar("_T")


class AnthropicAdapter(BaseAdapter):
    """Adapter that wraps an Anthropic client to filter tool schemas."""

    _USE_TOOLSET_SCHEMA: Dict[str, Any] = {
        "name": "use_toolset",
        "description": (
            "Dynamically load or unload tool groups for the next turn. "
            "Actions: set (replace all), add (add to current), "
            "remove (remove from current), reset (back to core), "
            "list (show available). "
            "Examples: 'add browser', 'set terminal,web', 'reset'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "load", "replace", "add", "enable",
                             "include", "remove", "disable", "drop",
                             "reset", "default", "list"],
                },
                "toolsets": {
                    "type": "string",
                    "description": "Comma-separated toolset names. "
                                   "Omit for 'reset' and 'list'.",
                },
            },
            "required": ["action"],
        },
    }

    @classmethod
    def _check_deps(cls) -> None:
        """Verify anthropic is installed. Raises AdapterNotInstalled if not."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise AdapterNotInstalled(
                "anthropic package is required for AnthropicAdapter. "
                "Install it with: pip install anthropic"
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

    # ------------------------------------------------------------------
    # Client wrapping
    # ------------------------------------------------------------------

    def wrap_client(self, client: _T) -> _T:
        import anthropic
        if isinstance(client, anthropic.AsyncAnthropic):
            return self._wrap_async(client)  # type: ignore
        return self._wrap_sync(client)

    def _wrap_sync(self, client) -> Any:
        original_create = client.messages.create

        @functools.wraps(original_create)
        def patched_create(*args: Any, **kwargs: Any) -> Any:
            kwargs = self._filter_and_process(kwargs)
            response = original_create(*args, **kwargs)
            self._process_tool_calls(response)
            return response

        client.messages.create = patched_create
        return client

    def _wrap_async(self, client) -> Any:
        original_create = client.messages.create

        @functools.wraps(original_create)
        async def patched_create(*args: Any, **kwargs: Any) -> Any:
            kwargs = self._filter_and_process(kwargs)
            response = await original_create(*args, **kwargs)
            self._process_tool_calls(response)
            return response

        client.messages.create = patched_create
        return client

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
                t.get("name") == "use_toolset" for t in filtered
            )
            if not has_bootstrap:
                filtered.append(self._USE_TOOLSET_SCHEMA)

        kwargs["tools"] = filtered
        return kwargs

    def _process_tool_calls(self, response: Any) -> None:
        """Inspect the response for use_toolset tool calls.

        Anthropic returns tool uses in ``content`` blocks with
        ``type == "tool_use"``.
        """
        try:
            content = getattr(response, "content", []) or []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use" and block.get("name") == "use_toolset":
                        args = block.get("input", {})
                        action = args.get("action", "list")
                        toolsets = args.get("toolsets", "")
                        self._filter.use_toolset(action, toolsets)
                elif hasattr(block, "type") and block.type == "tool_use":
                    if getattr(block, "name", None) == "use_toolset":
                        args = getattr(block, "input", {}) or {}
                        action = args.get("action", "list")
                        toolsets = args.get("toolsets", "")
                        self._filter.use_toolset(action, toolsets)
        except (AttributeError, TypeError, json.JSONDecodeError):
            pass
