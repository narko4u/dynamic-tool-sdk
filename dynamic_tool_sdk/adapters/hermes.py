"""
Dynamic Tool SDK — Hermes Agent adapter.

Reference implementation based on the working Hermes patches.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseAdapter


class HermesAdapter(BaseAdapter):
    """Adapter for NousResearch Hermes Agent.

    Integration points (for the Hermes codebase):
    - ``tools/usetoolset_tool.py`` — replace with this class
    - ``agent/conversation_loop.py`` — call ``apply()`` before each API call
    - ``agent/tool_executor.py`` — use filtered schemas
    - ``agent/iteration_budget.py`` — optional ToolCallBudget integration
    - ``model_tools.py`` — bootstrap ``use_toolset`` schema always present
    - ``toolsets.py`` — register ``use_toolset`` and ``get_tool_call_budget``
    """

    # The ``use_toolset`` schema that Hermes injects as a bootstrap tool
    _USE_TOOLSET_SCHEMA = {
        "name": "use_toolset",
        "description": (
            "Dynamically load or unload tool groups for the next turn. "
            "Start each session with core tools only. "
            "Load specialised toolsets on demand: e.g. 'add browser' for "
            "browsing, 'add vision' for image analysis, 'add web' for search. "
            "Actions: set/replace (replace all), add/enable/include (add to current), "
            "remove/disable/drop (remove from current), reset (back to core), list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "set", "load", "replace", "overwrite",
                        "add", "enable", "include",
                        "remove", "disable", "drop",
                        "reset", "default", "list",
                    ],
                    "description": (
                        "What to do: 'set/load/replace' = replace entirely, "
                        "'add/enable/include' = add to current, "
                        "'remove/disable/drop' = remove from current, "
                        "'reset/default' = back to core tools only, "
                        "'list' = print available toolsets"
                    ),
                },
                "toolsets": {
                    "type": "string",
                    "description": (
                        "Comma-separated toolset names. "
                        "Examples: 'terminal,web' or 'browser' or 'vision'. "
                        "Not used with 'reset' or 'list' actions."
                    ),
                },
            },
            "required": ["action"],
        },
    }

    # The ``get_tool_call_budget`` schema for Hermes
    _BUDGET_SCHEMA = {
        "name": "get_tool_call_budget",
        "description": (
            "Introspect the remaining tool-call budget for this turn. "
            "Returns the number of tool calls consumed and remaining. "
            "Use this to decide whether to continue the current approach "
            "or wrap up early."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    def wrap_client(self, client: Any) -> Any:
        """For Hermes, wrapping happens at the agent lifecycle level.

        This method is a no-op for Hermes because the integration is
        deeper — it patches conversation_loop.py and tool_executor.py,
        not the HTTP client.

        Returns the client unchanged.
        """
        return client

    def get_use_toolset_schema(self) -> Dict[str, Any]:
        return dict(self._USE_TOOLSET_SCHEMA)

    def get_budget_schema(self) -> Dict[str, Any]:
        """Return the schema for ``get_tool_call_budget`` tool."""
        return dict(self._BUDGET_SCHEMA)

    def get_bootstrap_schemas(self) -> List[Dict[str, Any]]:
        """Return schemas that must always be present (bootstrap tools)."""
        return [self._USE_TOOLSET_SCHEMA, self._BUDGET_SCHEMA]
