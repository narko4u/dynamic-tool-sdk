"""
Dynamic Tool SDK — BaseAdapter protocol.

All platform adapters inherit from BaseAdapter.  Each adapter:
1. Hooks into the platform's request lifecycle
2. Intercepts tool schemas before they're sent to the LLM
3. Filters them to only the active toolsets
4. Applies any pending toolset changes at the turn boundary
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core import SchemaFilter


class BaseAdapter(ABC):
    """Base class for all Dynamic Tool SDK platform adapters.

    Subclasses must implement:
    - ``wrap_client()`` — wrap the platform's client with our filtering
    - ``use_toolset()`` — expose the toolset-change function as a callable
    """

    def __init__(self, core_toolset: str = "core"):
        self._filter = SchemaFilter(core_toolset=core_toolset)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def schema_filter(self) -> SchemaFilter:
        return self._filter

    def use_toolset(self, action: str, toolsets: str = "") -> Dict[str, Any]:
        """Queue a toolset change for the next API round-trip.

        Returns a JSON-ready response dict for the model to read.
        """
        return self._filter.use_toolset(action, toolsets)

    def get_active_toolsets(self) -> List[str]:
        """Return list of currently active toolset names."""
        return [s for s in self._filter.state.active_list if s != "core"]

    def get_toolset_count(self) -> int:
        """Return the number of active toolsets (including core)."""
        return len(self._filter.state.active_list)

    # ------------------------------------------------------------------
    # Hook points (implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def wrap_client(self, client: Any) -> Any:
        """Wrap a platform client so tool schemas are auto-filtered.

        Example::
            adapter = OpenAIAdapter()
            wrapped = adapter.wrap_client(openai.OpenAI(api_key="..."))
            # All subsequent calls filter tool schemas automatically
        """
        ...

    @abstractmethod
    def get_use_toolset_schema(self) -> Dict[str, Any]:
        """Return the OpenAI-compatible function schema for ``use_toolset``.

        This schema is injected into every API call so the model can
        dynamically load/unload tool groups.
        """
        ...
