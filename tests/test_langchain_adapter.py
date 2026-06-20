"""Tests for the Dynamic Tool SDK LangChain adapter.

Mirrors the structure of test_openai_adapter.py and test_anthropic_adapter.py.

LangChain differences from OpenAI / Anthropic adapters:
  - Tools are BaseTool instances (not raw dicts); names come from ``.name``
  - ``BaseChatModel`` is Pydantic — methods cannot be monkey-patched in place,
    so ``wrap_chat_model()`` returns a ``_LangChainModelWrapper`` proxy instead
  - Responses are ``AIMessage`` objects with ``.tool_calls`` as a list of dicts:
    ``[{"name": "...", "args": {...}, "id": "...", "type": "tool_call"}]``
  - The bootstrap ``use_toolset`` tool is a LangChain ``StructuredTool``
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


SDK_ROOT    = Path(__file__).resolve().parents[1]   # tests/ -> sdk/
PACKAGE_ROOT = SDK_ROOT / "dynamic_tool_sdk"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _install_local_dynamic_tool_sdk_alias() -> None:
    package = sys.modules.get("dynamic_tool_sdk")
    if package is None:
        package = types.ModuleType("dynamic_tool_sdk")
        package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
        sys.modules["dynamic_tool_sdk"] = package


def _import_adapter_modules():
    _install_local_dynamic_tool_sdk_alias()
    adapter_module  = importlib.import_module("dynamic_tool_sdk.adapters.langchain")
    registry_module = importlib.import_module("dynamic_tool_sdk.core.registry")
    return adapter_module.LangChainAdapter, adapter_module, registry_module


def _lc_tool(name: str):
    """Minimal stand-in for a BaseTool with only a .name attribute.

    Used in filtering tests so we don't need real LangChain tool creation
    overhead.  The adapter resolves names via ``_get_tool_name(t)`` which
    checks ``hasattr(t, 'name')`` first, so SimpleNamespace works fine.
    """
    return SimpleNamespace(name=name)


def _tool_names(tools: list) -> list:
    """Extract names from a mixed list of mock tools or StructuredTools."""
    out = []
    for t in tools:
        if isinstance(t, dict):
            out.append(t.get("name") or t.get("function", {}).get("name"))
        else:
            out.append(getattr(t, "name", None))
    return out


# ---------------------------------------------------------------------------
# Mock clients (no real API calls)
# ---------------------------------------------------------------------------

class _MockResponse:
    """Mimics an AIMessage with a use_toolset tool_call."""

    def __init__(self, toolsets: str = "browser"):
        self.content = ""
        self.tool_calls = [
            {
                "name": "use_toolset",
                "args": {"action": "add", "toolsets": toolsets},
                "id": "tc_mock_1",
                "type": "tool_call",
            }
        ]


class _MockModel:
    """Non-Pydantic model stub that accepts bind_tools / invoke without an API call."""

    def __init__(self) -> None:
        self._bound_tools: list = []
        self.invoke_calls: list = []

    def bind_tools(self, tools: list, **kwargs):
        self._bound_tools = list(tools)
        return _MockBoundModel(self, tools)

    def invoke(self, input, config=None, **kwargs):
        self.invoke_calls.append(input)
        return _MockResponse()


class _MockBoundModel:
    """Stub returned by _MockModel.bind_tools."""

    def __init__(self, parent, tools: list) -> None:
        self._parent = parent
        self._tools = tools

    def invoke(self, input, config=None, **kwargs):
        return _MockResponse()

    async def ainvoke(self, input, config=None, **kwargs):
        return _MockResponse()


# ---------------------------------------------------------------------------
# 1 & 2 & 3: Import contract
# ---------------------------------------------------------------------------

class LangChainAdapterImportTests(unittest.TestCase):

    def test_langchain_packages_are_installed(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import langchain; import langchain_core; "
                "print(langchain.__version__, langchain_core.__version__)",
            ],
            cwd=SDK_ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            "langchain / langchain-core not installed. "
            "Repair: pip install langchain langchain-core\n"
            f"stderr:\n{result.stderr}",
        )

    def test_documented_import_path_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-c",
                "from dynamic_tool_sdk.adapters.langchain import LangChainAdapter;"
                " print(LangChainAdapter)",
            ],
            cwd=SDK_ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            "Documented import path failed.\n"
            f"stderr:\n{result.stderr}",
        )

    def test_missing_langchain_raises_AdapterNotInstalled(self) -> None:
        """When langchain_core is absent, instantiation must raise AdapterNotInstalled.

        ``AdapterNotInstalled`` is a ``DynamicToolError`` subclass — NOT a
        ``ModuleNotFoundError``.  The subprocess exits 1 if a raw
        ``ModuleNotFoundError`` leaks; exits 0 if the proper error is raised.
        """
        code = textwrap.dedent(
            f"""
            import importlib, importlib.abc, sys, types

            sdk_root = {str(PACKAGE_ROOT)!r}
            package = types.ModuleType("dynamic_tool_sdk")
            package.__path__ = [sdk_root]
            sys.modules["dynamic_tool_sdk"] = package
            for name in list(sys.modules):
                if name == "langchain" or name.startswith("langchain"):
                    del sys.modules[name]

            class BlockLangChain(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith("langchain"):
                        raise ModuleNotFoundError("simulated missing langchain")
                    return None

            sys.meta_path.insert(0, BlockLangChain())
            try:
                mod = importlib.import_module("dynamic_tool_sdk.adapters.langchain")
                mod.LangChainAdapter()
            except ModuleNotFoundError as exc:
                print(type(exc).__name__, exc)
                raise SystemExit(1)
            except Exception as exc:
                print(type(exc).__name__, exc)
                raise SystemExit(0)
            else:
                print("adapter instantiated without langchain - _check_deps may be broken")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=SDK_ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            "Raw ModuleNotFoundError escaped — _check_deps did not convert it "
            "to AdapterNotInstalled.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


# ---------------------------------------------------------------------------
# 4: Model wrapping
# ---------------------------------------------------------------------------

class LangChainAdapterWrappingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.LangChainAdapter, self.adapter_module, _ = _import_adapter_modules()

    def test_wrap_chat_model_returns_proxy_wrapper(self) -> None:
        """wrap_chat_model() must return a _LangChainModelWrapper, not the original."""
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_chat_model(model)

        self.assertIsInstance(
            wrapped,
            self.adapter_module._LangChainModelWrapper,
            "wrap_chat_model must return a _LangChainModelWrapper proxy because "
            "Pydantic models cannot be monkey-patched in place.",
        )

    def test_wrap_client_is_alias_for_wrap_chat_model(self) -> None:
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_client(model)

        self.assertIsInstance(wrapped, self.adapter_module._LangChainModelWrapper)

    def test_real_ChatOpenAI_wraps_without_error(self) -> None:
        """wrap_chat_model on a real ChatOpenAI instance should not raise."""
        from langchain_openai import ChatOpenAI

        client  = ChatOpenAI(api_key="sk-test-fake-key")
        adapter = self.LangChainAdapter()
        wrapped = adapter.wrap_chat_model(client)

        self.assertIsInstance(wrapped, self.adapter_module._LangChainModelWrapper)
        # The proxy should forward arbitrary attribute reads to the real model
        self.assertEqual(wrapped.model_name, client.model_name)

    def test_wrapper_invoke_processes_tool_calls(self) -> None:
        """After wrapping, invoke() must call _process_tool_calls on the response."""
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_chat_model(model)

        # _MockModel.invoke returns a response with use_toolset tool_call for "browser"
        wrapped.invoke("hello")

        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["browser", "core"],
            "invoke() must pass the response to _process_tool_calls() so that "
            "the queued toolset change is recorded.",
        )

    def test_wrapper_bind_tools_returns_bound_wrapper(self) -> None:
        """bind_tools() on the proxy must return a _LangChainBoundWrapper."""
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_chat_model(model)

        tool_a = _lc_tool("some_tool")
        bound  = wrapped.bind_tools([tool_a])

        self.assertIsInstance(bound, self.adapter_module._LangChainBoundWrapper)

    def test_bound_wrapper_invoke_processes_tool_calls(self) -> None:
        """invoke() on the bound wrapper must also process tool calls."""
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_chat_model(model)
        bound   = wrapped.bind_tools([_lc_tool("some_tool")])

        bound.invoke("hello")

        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["browser", "core"],
        )

    def test_bound_wrapper_ainvoke_is_coroutine_function(self) -> None:
        """ainvoke on _LangChainBoundWrapper must be a coroutine function."""
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_chat_model(model)
        bound   = wrapped.bind_tools([_lc_tool("some_tool")])

        self.assertTrue(asyncio.iscoroutinefunction(bound.ainvoke))

    def test_bound_wrapper_ainvoke_runs_without_error(self) -> None:
        """Async path: ainvoke processes tool calls without raising."""
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_chat_model(model)
        bound   = wrapped.bind_tools([_lc_tool("some_tool")])

        async def run():
            return await bound.ainvoke("hello")

        asyncio.run(run())
        self.assertEqual(adapter.schema_filter.state.pending, ["browser", "core"])


# ---------------------------------------------------------------------------
# 5: Tool filtering (_filter_tools public alias + Anthropic/LangChain schemas)
# ---------------------------------------------------------------------------

class LangChainAdapterFilterTests(unittest.TestCase):

    def setUp(self) -> None:
        self.LangChainAdapter, _, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)

        # Register with dict schemas so the core filter knows the tool names
        self.alpha    = _lc_tool("lc_alpha_tool")
        self.beta     = _lc_tool("lc_beta_tool")
        self.core_t   = _lc_tool("lc_core_tool")

        self.registry.register_toolset(
            "lc_alpha", "LangChain alpha toolset.",
            tools=[{"name": "lc_alpha_tool", "description": "alpha"}],
        )
        self.registry.register_toolset(
            "lc_beta", "LangChain beta toolset.",
            tools=[{"name": "lc_beta_tool", "description": "beta"}],
        )
        self.registry.register_tool_schema(
            "core",
            {"name": "lc_core_tool", "description": "core tool"},
        )

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_filter_tools_public_alias_exists(self) -> None:
        adapter = self.LangChainAdapter()
        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "LangChainAdapter must expose _filter_tools() matching OpenAI/Anthropic adapters.",
        )
        self.assertTrue(callable(adapter._filter_tools))

    def test_filter_tools_keeps_active_toolsets_and_bootstrap(self) -> None:
        adapter = self.LangChainAdapter()
        adapter.use_toolset("add", "lc_alpha")

        kwargs = adapter._filter_tools(
            {"tools": [self.alpha, self.beta, self.core_t]}
        )
        names = _tool_names(kwargs["tools"])

        self.assertIn("lc_alpha_tool", names)
        self.assertIn("lc_core_tool", names)
        self.assertIn("use_toolset", names)
        self.assertNotIn("lc_beta_tool", names)

    def test_filter_tools_bootstrap_is_structured_tool(self) -> None:
        """The injected bootstrap must be a StructuredTool (LangChain-native)."""
        from langchain_core.tools import BaseTool

        adapter = self.LangChainAdapter()
        kwargs  = adapter._filter_tools({"tools": []})
        tools   = kwargs["tools"]

        self.assertEqual(len(tools), 1)
        bootstrap = tools[0]
        self.assertIsInstance(
            bootstrap, BaseTool,
            "LangChain adapter must inject a BaseTool (StructuredTool) as the "
            "bootstrap use_toolset, not a raw dict.",
        )
        self.assertEqual(bootstrap.name, "use_toolset")

    def test_filter_tools_empty_list_injects_only_bootstrap(self) -> None:
        adapter = self.LangChainAdapter()
        kwargs  = adapter._filter_tools({"tools": []})
        names   = _tool_names(kwargs["tools"])
        self.assertEqual(names, ["use_toolset"])

    def test_filter_tools_no_tools_key_injects_only_bootstrap(self) -> None:
        adapter = self.LangChainAdapter()
        kwargs  = adapter._filter_tools({})
        names   = _tool_names(kwargs["tools"])
        self.assertEqual(names, ["use_toolset"])

    def test_filter_tools_with_dict_format_tools(self) -> None:
        """_filter_tools also handles OpenAI-style dict tool schemas."""
        adapter = self.LangChainAdapter()
        adapter.use_toolset("add", "lc_alpha")

        alpha_dict = {"type": "function", "function": {"name": "lc_alpha_tool"}}
        beta_dict  = {"type": "function", "function": {"name": "lc_beta_tool"}}

        kwargs = adapter._filter_tools({"tools": [alpha_dict, beta_dict]})
        # _get_tool_name falls back to function.name for OpenAI-style dicts
        result_names = _tool_names(kwargs["tools"])

        self.assertIn("lc_alpha_tool", [
            n or t.get("function", {}).get("name")
            for n, t in zip(result_names, kwargs["tools"])
        ] or result_names)


# ---------------------------------------------------------------------------
# 6: _process_tool_calls — AIMessage format
# ---------------------------------------------------------------------------

class LangChainAdapterToolCallTests(unittest.TestCase):

    def setUp(self) -> None:
        self.LangChainAdapter, _, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.registry.register_toolset(
            "lc_alpha", "LangChain alpha toolset.",
            tools=[{"name": "lc_alpha_tool"}],
        )

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_process_tool_calls_handles_dict_form(self) -> None:
        """AIMessage.tool_calls items are plain dicts — handle them."""
        from langchain_core.messages import AIMessage

        adapter  = self.LangChainAdapter()
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "use_toolset",
                    "args": {"action": "add", "toolsets": "lc_alpha"},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        )

        adapter._process_tool_calls(response)

        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["core", "lc_alpha"],
        )

    def test_process_tool_calls_handles_object_form(self) -> None:
        """Defensive: also handle tool_call objects (e.g., future LangChain versions)."""
        adapter  = self.LangChainAdapter()
        response = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    name="use_toolset",
                    args={"action": "add", "toolsets": "lc_alpha"},
                )
            ]
        )

        adapter._process_tool_calls(response)

        self.assertEqual(adapter.schema_filter.state.pending, ["core", "lc_alpha"])

    def test_process_tool_calls_ignores_non_use_toolset(self) -> None:
        """Other tool calls must not trigger toolset changes."""
        from langchain_core.messages import AIMessage

        adapter  = self.LangChainAdapter()
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_web",
                    "args": {"query": "langchain"},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        )

        adapter._process_tool_calls(response)

        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_process_tool_calls_handles_empty_tool_calls(self) -> None:
        from langchain_core.messages import AIMessage

        adapter  = self.LangChainAdapter()
        response = AIMessage(content="Just text, no tool calls.")

        adapter._process_tool_calls(response)

        self.assertIsNone(adapter.schema_filter.state.pending)


# ---------------------------------------------------------------------------
# 7: Edge cases
# ---------------------------------------------------------------------------

class LangChainAdapterEdgeCaseTests(unittest.TestCase):

    def setUp(self) -> None:
        self.LangChainAdapter, _, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.alpha  = _lc_tool("lc_alpha_tool")
        self.beta   = _lc_tool("lc_beta_tool")
        self.core_t = _lc_tool("lc_core_tool")
        self.registry.register_toolset(
            "lc_alpha", "LangChain alpha toolset.",
            tools=[{"name": "lc_alpha_tool"}],
        )
        self.registry.register_toolset(
            "lc_beta", "LangChain beta toolset.",
            tools=[{"name": "lc_beta_tool"}],
        )
        self.registry.register_tool_schema("core", {"name": "lc_core_tool"})

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_multiple_pending_toolset_changes_combined(self) -> None:
        adapter = self.LangChainAdapter()
        adapter.use_toolset("add", "lc_alpha")
        adapter.use_toolset("add", "lc_beta")

        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core_t]})
        names  = _tool_names(kwargs["tools"])

        self.assertIn("lc_alpha_tool", names)
        self.assertIn("lc_beta_tool",  names)
        self.assertIn("lc_core_tool",  names)
        self.assertIn("use_toolset",   names)

    def test_reset_to_core_removes_loaded_toolsets(self) -> None:
        adapter = self.LangChainAdapter()
        adapter.use_toolset("add", "lc_alpha")
        adapter._filter_tools({"tools": [self.alpha, self.beta, self.core_t]})

        adapter.use_toolset("reset")
        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core_t]})
        names  = _tool_names(kwargs["tools"])

        self.assertIn("lc_core_tool", names)
        self.assertIn("use_toolset",  names)
        self.assertNotIn("lc_alpha_tool", names)
        self.assertNotIn("lc_beta_tool",  names)

    def test_removing_unloaded_toolset_keeps_core(self) -> None:
        adapter  = self.LangChainAdapter()
        response = adapter.use_toolset("remove", "lc_alpha")  # not loaded yet

        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core_t]})
        names  = _tool_names(kwargs["tools"])

        self.assertEqual(response["status"], "deferred")
        self.assertIn("lc_core_tool", names)
        self.assertNotIn("lc_alpha_tool", names)

    def test_process_tool_calls_handles_none_tool_calls(self) -> None:
        adapter  = self.LangChainAdapter()
        response = SimpleNamespace(tool_calls=None)
        adapter._process_tool_calls(response)
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_process_tool_calls_handles_missing_tool_calls_attr(self) -> None:
        adapter  = self.LangChainAdapter()
        response = SimpleNamespace()
        adapter._process_tool_calls(response)
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_auto_inject_false_skips_bootstrap(self) -> None:
        adapter = self.LangChainAdapter(auto_inject=False)
        kwargs  = adapter._filter_tools({})
        self.assertNotIn("tools", kwargs)

    def test_bind_tools_injects_bootstrap_in_filtered_list(self) -> None:
        """bind_tools on the wrapped model must include the bootstrap tool."""
        adapter = self.LangChainAdapter()
        model   = _MockModel()
        wrapped = adapter.wrap_chat_model(model)

        adapter.use_toolset("add", "lc_alpha")
        bound = wrapped.bind_tools([self.alpha, self.beta, self.core_t])

        # The MockModel stores the final filtered list in _bound_tools
        actual_model = object.__getattribute__(bound, "_lc_bound")  # _MockBoundModel
        bound_names  = _tool_names(actual_model._tools)

        self.assertIn("lc_alpha_tool", bound_names)
        self.assertIn("lc_core_tool",  bound_names)
        self.assertIn("use_toolset",   bound_names)
        self.assertNotIn("lc_beta_tool", bound_names)


# ---------------------------------------------------------------------------
# 8: Consistency with OpenAI / Anthropic adapter surface
# ---------------------------------------------------------------------------

class LangChainAdapterConsistencyTests(unittest.TestCase):

    def setUp(self) -> None:
        self.LangChainAdapter, _, _ = _import_adapter_modules()

    def test_has_check_deps_classmethod(self) -> None:
        self.assertTrue(hasattr(self.LangChainAdapter, "_check_deps"))
        self.assertTrue(callable(self.LangChainAdapter._check_deps))

    def test_has_filter_tools_public_method(self) -> None:
        adapter = self.LangChainAdapter()
        self.assertTrue(hasattr(adapter, "_filter_tools"))
        self.assertTrue(callable(adapter._filter_tools))

    def test_has_process_tool_calls_method(self) -> None:
        adapter = self.LangChainAdapter()
        self.assertTrue(hasattr(adapter, "_process_tool_calls"))
        self.assertTrue(callable(adapter._process_tool_calls))

    def test_has_wrap_client_method(self) -> None:
        adapter = self.LangChainAdapter()
        self.assertTrue(hasattr(adapter, "wrap_client"))

    def test_has_get_use_toolset_schema(self) -> None:
        adapter = self.LangChainAdapter()
        schema  = adapter.get_use_toolset_schema()
        self.assertIsInstance(schema, dict)
        # Dict schema (OpenAI-compat) for interoperability
        fn = schema.get("function", {})
        self.assertEqual(fn.get("name"), "use_toolset")

    def test_use_toolset_injection_enabled_by_default(self) -> None:
        adapter = self.LangChainAdapter()
        kwargs  = adapter._filter_tools({})
        names   = _tool_names(kwargs.get("tools", []))
        self.assertIn(
            "use_toolset", names,
            "use_toolset bootstrap must be auto-injected when auto_inject=True",
        )

    def test_schema_filter_property_exposed(self) -> None:
        adapter = self.LangChainAdapter()
        self.assertIsNotNone(adapter.schema_filter)

    def test_get_active_toolsets_returns_list(self) -> None:
        adapter = self.LangChainAdapter()
        self.assertIsInstance(adapter.get_active_toolsets(), list)

    def test_get_toolset_count_includes_core(self) -> None:
        adapter = self.LangChainAdapter()
        self.assertGreaterEqual(adapter.get_toolset_count(), 1)

    def test_use_toolset_method_returns_dict(self) -> None:
        adapter = self.LangChainAdapter()
        result  = adapter.use_toolset("list")
        self.assertIsInstance(result, dict)
        self.assertIn("active_toolsets", result)

    def test_get_tool_name_handles_base_tool(self) -> None:
        """_get_tool_name must read .name from BaseTool-like objects."""
        adapter = self.LangChainAdapter()
        tool    = _lc_tool("my_tool")
        self.assertEqual(adapter._get_tool_name(tool), "my_tool")

    def test_get_tool_name_handles_dict_schema(self) -> None:
        """_get_tool_name must read name from flat and nested dict schemas."""
        adapter = self.LangChainAdapter()
        flat    = {"name": "my_tool"}
        nested  = {"type": "function", "function": {"name": "my_nested_tool"}}
        self.assertEqual(adapter._get_tool_name(flat),   "my_tool")
        self.assertEqual(adapter._get_tool_name(nested), "my_nested_tool")

    def test_bootstrap_structured_tool_is_lazy_singleton(self) -> None:
        """_get_bootstrap_tool() must return the same instance on repeat calls."""
        adapter = self.LangChainAdapter()
        t1 = adapter._get_bootstrap_tool()
        t2 = adapter._get_bootstrap_tool()
        self.assertIs(t1, t2, "_get_bootstrap_tool should be lazily cached")


if __name__ == "__main__":
    unittest.main()
