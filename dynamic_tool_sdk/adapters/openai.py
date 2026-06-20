"""
Dynamic Tool SDK — OpenAI Python client adapter.

Wraps ``openai.OpenAI`` and ``openai.AsyncOpenAI`` so tool schemas are
auto-filtered on every API call.  The ``use_toolset`` schema is injected
as a bootstrap tool so the model can dynamically request toolset changes.

Usage::
    from dynamic_tool_sdk.adapters.openai import OpenAIAdapter
    import openai

    client = OpenAIAdapter().wrap_client(openai.OpenAI(api_key="sk-..."))
    # Client now auto-filters tool schemas and supports use_toolset()
"""
from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Optional, TypeVar

from ..core import AdapterNotInstalled
from .base import BaseAdapter

_C = TypeVar("_C")


class OpenAIAdapter(BaseAdapter):
    """Adapter that wraps an OpenAI client to filter tool schemas."""

    # The bootstrap use_toolset schema sent on every request
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
        },
    }

    @classmethod
    def _check_deps(cls) -> None:
        """Verify openai is installed. Raises AdapterNotInstalled if not."""
        try:
            import openai  # noqa: F401
        except ImportError:
            raise AdapterNotInstalled(
                "openai package is required for OpenAIAdapter. "
                "Install it with: pip install openai>=1.0.0"
            )

    def __init__(
        self,
        core_toolset: str = "core",
        auto_inject: bool = True,
    ):
        """Initialise the OpenAI adapter.

        Args:
            core_toolset: Name of the always-present toolset.
            auto_inject: If True, inject the use_toolset bootstrap schema
                         into every request automatically.
        """
        self._check_deps()
        super().__init__(core_toolset=core_toolset)
        self._auto_inject = auto_inject

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def get_use_toolset_schema(self) -> Dict[str, Any]:
        return dict(self._USE_TOOLSET_SCHEMA)

    # ------------------------------------------------------------------
    # Client wrapping
    # ------------------------------------------------------------------

    def wrap_client(self, client: _C) -> _C:
        """Wrap an OpenAI or AsyncOpenAI client.

        Returns the same client type with chat.completions.create patched.
        """
        import openai

        if isinstance(client, openai.AsyncOpenAI):
            return self._wrap_async(client)  # type: ignore
        return self._wrap_sync(client)

    def _wrap_sync(self, client) -> Any:
        import openai
        original_create = client.chat.completions.create

        @functools.wraps(original_create)
        def patched_create(*args: Any, **kwargs: Any) -> Any:
            kwargs = self._filter_and_process(kwargs)
            response = original_create(*args, **kwargs)
            self._process_tool_calls(response)
            return response

        client.chat.completions.create = patched_create
        return client

    def _wrap_async(self, client) -> Any:
        import openai
        original_create = client.chat.completions.create

        @functools.wraps(original_create)
        async def patched_create(*args: Any, **kwargs: Any) -> Any:
            kwargs = self._filter_and_process(kwargs)
            response = await original_create(*args, **kwargs)
            self._process_tool_calls(response)
            return response

        client.chat.completions.create = patched_create
        return client

    # ------------------------------------------------------------------
    # Public API (used directly by tests)
    # ------------------------------------------------------------------

    def _filter_tools(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Public alias for _filter_and_process. Used by tests directly."""
        return self._filter_and_process(kwargs)

    # ------------------------------------------------------------------
    # Tool filtering + tool_call processing
    # ------------------------------------------------------------------

    def _filter_and_process(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter tool schemas and apply any pending changes."""
        tools = kwargs.get("tools")

        # Apply any pending toolset change before this turn
        self._filter.apply()

        if tools is None:
            if self._auto_inject:
                kwargs["tools"] = [self._USE_TOOLSET_SCHEMA]
            return kwargs

        filtered = self._filter.filter(tools)

        if self._auto_inject:
            has_bootstrap = any(
                t.get("function", {}).get("name") == "use_toolset"
                for t in filtered
            )
            if not has_bootstrap:
                filtered.append(self._USE_TOOLSET_SCHEMA)

        kwargs["tools"] = filtered
        return kwargs

    def _process_tool_calls(self, response: Any) -> None:
        """Inspect the model's response for use_toolset tool calls.

        If the model requested a toolset change, apply it now so the
        change takes effect on the *next* turn.  Handles both OpenAI
        response objects and plain dicts (test mocks).
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
