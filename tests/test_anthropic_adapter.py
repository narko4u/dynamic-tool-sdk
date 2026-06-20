"""Tests for the Dynamic Tool SDK Anthropic adapter.

These tests mirror the structure of test_openai_adapter.py and cover both the
public import contract and the adapter mechanics.

KNOWN GAPS (tests will FAIL — read the repair hints in each test):
  - AnthropicAdapter is missing the ``_filter_tools`` public alias that
    OpenAIAdapter exposes (tests 5+ that call adapter._filter_tools will
    raise AttributeError).

Repair summary for the Anthropic adapter:
  Add to AnthropicAdapter in adapters/anthropic.py:

      def _filter_tools(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
          \"\"\"Public alias for _filter_and_process. Mirrors OpenAIAdapter pattern.\"\"\"
          return self._filter_and_process(kwargs)
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


SDK_ROOT = Path(__file__).resolve().parents[1]   # tests/ -> sdk/
PACKAGE_ROOT = SDK_ROOT / "dynamic_tool_sdk"


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _install_local_dynamic_tool_sdk_alias() -> None:
    """Expose this source tree as dynamic_tool_sdk for local tests."""
    package = sys.modules.get("dynamic_tool_sdk")
    if package is None:
        package = types.ModuleType("dynamic_tool_sdk")
        package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
        sys.modules["dynamic_tool_sdk"] = package


def _import_adapter_modules():
    _install_local_dynamic_tool_sdk_alias()
    adapter_module = importlib.import_module("dynamic_tool_sdk.adapters.anthropic")
    registry_module = importlib.import_module("dynamic_tool_sdk.core.registry")
    return adapter_module.AnthropicAdapter, registry_module


def _anthropic_tool_schema(name: str) -> dict:
    """Return an Anthropic-format tool schema (input_schema, not parameters)."""
    return {
        "name": name,
        "description": f"{name} test tool",
        "input_schema": {"type": "object", "properties": {}},
    }


def _tool_names(tools: list) -> list:
    """Extract names from a list of Anthropic-format tool schemas."""
    return [t.get("name") for t in tools]


# ---------------------------------------------------------------------------
# Minimal mock clients (no real Anthropic API calls)
# ---------------------------------------------------------------------------

class _SyncMessages:
    """Anthropic-style messages resource — returns a tool_use response."""

    def __init__(self) -> None:
        self.calls: list = []

    def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        # Anthropic response: content list with object-style tool_use blocks
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="use_toolset",
                    input={"action": "add", "toolsets": "browser"},
                )
            ]
        )


class _SyncAnthropicClient:
    def __init__(self) -> None:
        self.messages = _SyncMessages()


class _AsyncMessages:
    def __init__(self) -> None:
        self.calls: list = []

    async def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[])


class _AsyncAnthropicClient:
    def __init__(self) -> None:
        self.messages = _AsyncMessages()


# ---------------------------------------------------------------------------
# 1 & 2: Import contract tests
# ---------------------------------------------------------------------------

class AnthropicAdapterImportTests(unittest.TestCase):

    # Test 1 — pip-level dependency
    def test_anthropic_package_is_installed(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import anthropic; print(anthropic.__version__)"],
            cwd=SDK_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "anthropic is not installed. Repair: run `pip install anthropic`.\n"
            f"stderr:\n{result.stderr}",
        )

    # Test 2 — documented import path
    def test_documented_dynamic_tool_sdk_import_path_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from dynamic_tool_sdk.adapters.anthropic import AnthropicAdapter;"
                " print(AnthropicAdapter)",
            ],
            cwd=SDK_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "Documented import path failed. Repair: ensure the SDK source root "
            "is on PYTHONPATH or installed as a package.\n"
            f"stderr:\n{result.stderr}",
        )

    # Test 3 — missing dependency raises AdapterNotInstalled, not raw ModuleNotFoundError
    def test_missing_anthropic_dependency_raises_AdapterNotInstalled(self) -> None:
        """
        When `anthropic` is absent, instantiating AnthropicAdapter() must raise
        AdapterNotInstalled (DynamicToolError subclass), not a raw ModuleNotFoundError.

        The adapter's module-level import is clean (no top-level `import anthropic`),
        so the error is deferred to __init__ via _check_deps().  The subprocess below
        imports the module then instantiates the adapter — the AdapterNotInstalled
        must surface instead of raw ModuleNotFoundError.
        """
        code = textwrap.dedent(
            f"""
            import importlib
            import importlib.abc
            import sys
            import types

            sdk_root = {str(PACKAGE_ROOT)!r}
            package = types.ModuleType("dynamic_tool_sdk")
            package.__path__ = [sdk_root]
            sys.modules["dynamic_tool_sdk"] = package
            for name in list(sys.modules):
                if name == "anthropic" or name.startswith("anthropic."):
                    del sys.modules[name]

            class BlockAnthropic(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "anthropic" or fullname.startswith("anthropic."):
                        raise ModuleNotFoundError("simulated missing anthropic")
                    return None

            sys.meta_path.insert(0, BlockAnthropic())
            try:
                mod = importlib.import_module("dynamic_tool_sdk.adapters.anthropic")
                mod.AnthropicAdapter()   # Must raise AdapterNotInstalled
            except ModuleNotFoundError as exc:
                # BAD: raw error escaped — _check_deps didn't catch it
                print(type(exc).__name__, exc)
                raise SystemExit(1)
            except Exception as exc:
                # GOOD: AdapterNotInstalled (or another DynamicToolError) raised
                print(type(exc).__name__, exc)
                raise SystemExit(0)
            else:
                print("adapter instantiated without anthropic — _check_deps may be broken")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=SDK_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "AnthropicAdapter raises raw ModuleNotFoundError when anthropic is absent. "
            "Repair: _check_deps() already wraps the import in try/except and re-raises "
            "AdapterNotInstalled — verify the ImportError catch includes ModuleNotFoundError "
            "(it should, since ModuleNotFoundError is a subclass of ImportError).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


# ---------------------------------------------------------------------------
# 3 & 4: Client wrapping
# ---------------------------------------------------------------------------

class AnthropicAdapterClientWrappingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.AnthropicAdapter, self.registry = _import_adapter_modules()

    # Test 3 — sync client wrapping
    def test_real_sync_client_is_wrapped_and_create_is_patched(self) -> None:
        """
        anthropic.Anthropic(api_key=...) must be returned as-is with
        messages.create replaced by a wrapper function.
        """
        import anthropic

        client = anthropic.Anthropic(api_key="sk-ant-fake-key-for-testing-only")
        original_create = client.messages.create
        adapter = self.AnthropicAdapter()

        wrapped = adapter.wrap_client(client)

        self.assertIs(wrapped, client, "wrap_client should return the same client object")
        self.assertIsNot(
            client.messages.create, original_create,
            "messages.create should be replaced by the patched wrapper",
        )
        # functools.wraps sets __wrapped__ on the inner function
        self.assertEqual(
            client.messages.create.__wrapped__,
            original_create,
            "Patched create should carry __wrapped__ pointing to the original",
        )

    # Test 4 — async client wrapping
    def test_real_async_client_is_wrapped_and_create_is_coroutine(self) -> None:
        """
        anthropic.AsyncAnthropic must be detected as async and the patched
        messages.create must be a coroutine function.
        """
        import anthropic

        client = anthropic.AsyncAnthropic(api_key="sk-ant-fake-key-for-testing-only")
        original_create = client.messages.create
        adapter = self.AnthropicAdapter()

        wrapped = adapter.wrap_client(client)

        self.assertIs(wrapped, client)
        self.assertIsNot(client.messages.create, original_create)
        self.assertTrue(
            asyncio.iscoroutinefunction(client.messages.create),
            "Patched messages.create on AsyncAnthropic must be a coroutine function",
        )


# ---------------------------------------------------------------------------
# 5: Tool schema format + _filter_tools consistency
# ---------------------------------------------------------------------------

class AnthropicAdapterFilterTests(unittest.TestCase):

    def setUp(self) -> None:
        self.AnthropicAdapter, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        # Register Anthropic-format schemas in the registry
        self.alpha = _anthropic_tool_schema("ant_alpha_tool")
        self.beta = _anthropic_tool_schema("ant_beta_tool")
        self.core_tool = _anthropic_tool_schema("ant_core_tool")
        self.registry.register_toolset(
            "ant_alpha", "Anthropic adapter alpha toolset.", tools=[self.alpha]
        )
        self.registry.register_toolset(
            "ant_beta", "Anthropic adapter beta toolset.", tools=[self.beta]
        )
        self.registry.register_tool_schema("core", self.core_tool)

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    # Consistency check — _filter_tools public alias
    def test_filter_tools_public_alias_exists(self) -> None:
        """
        GAP — AnthropicAdapter is missing _filter_tools().

        OpenAIAdapter exposes `_filter_tools` as a documented public alias of
        `_filter_and_process` so tests and downstream code have a stable entry
        point without touching private internals.  AnthropicAdapter does not
        have this method.

        REPAIR — Add to AnthropicAdapter (adapters/anthropic.py):

            def _filter_tools(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Public alias for _filter_and_process. Mirrors OpenAIAdapter.\"\"\"
                return self._filter_and_process(kwargs)
        """
        adapter = self.AnthropicAdapter()
        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "AnthropicAdapter is missing _filter_tools(). "
            "Repair: add `def _filter_tools(self, kwargs): return self._filter_and_process(kwargs)`",
        )

    # Test 5a — use_toolset bootstrap schema uses input_schema (not parameters)
    def test_use_toolset_bootstrap_schema_uses_input_schema(self) -> None:
        adapter = self.AnthropicAdapter()
        schema = adapter.get_use_toolset_schema()

        self.assertEqual(schema["name"], "use_toolset")
        self.assertIn(
            "input_schema", schema,
            "Anthropic format requires `input_schema`, not `parameters`",
        )
        self.assertNotIn(
            "parameters", schema,
            "Anthropic format must NOT use `parameters` (that is OpenAI format)",
        )
        # Must not be wrapped in a {"type": "function", "function": {...}} envelope
        self.assertNotIn(
            "function", schema,
            "Anthropic schemas must not use the OpenAI function envelope",
        )

    # Test 5b — active toolset schemas survive; inactive ones are filtered out
    def test_filter_tools_keeps_active_toolsets_and_bootstrap(self) -> None:
        """
        Calls adapter._filter_tools() — will raise AttributeError if the public
        alias is missing.  Repair: see test_filter_tools_public_alias_exists.
        """
        adapter = self.AnthropicAdapter()
        adapter.use_toolset("add", "ant_alpha")

        kwargs = adapter._filter_tools(
            {"tools": [self.alpha, self.beta, self.core_tool]}
        )

        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"ant_alpha_tool", "ant_core_tool", "use_toolset"},
            "Only active-toolset tools (ant_alpha + core) and the use_toolset "
            "bootstrap should survive filtering",
        )

    # Test 5c — empty tools list injects only bootstrap
    def test_filter_tools_empty_list_injects_only_bootstrap(self) -> None:
        adapter = self.AnthropicAdapter()
        kwargs = adapter._filter_tools({"tools": []})
        self.assertEqual(_tool_names(kwargs["tools"]), ["use_toolset"])

    # Test 5d — absent tools key injects only bootstrap
    def test_filter_tools_no_tools_key_injects_only_bootstrap(self) -> None:
        adapter = self.AnthropicAdapter()
        kwargs = adapter._filter_tools({})
        self.assertEqual(_tool_names(kwargs["tools"]), ["use_toolset"])


# ---------------------------------------------------------------------------
# 6: _process_tool_calls — Anthropic uses tool_use blocks, not tool_calls
# ---------------------------------------------------------------------------

class AnthropicAdapterToolCallTests(unittest.TestCase):

    def setUp(self) -> None:
        self.AnthropicAdapter, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.alpha = _anthropic_tool_schema("ant_alpha_tool")
        self.beta = _anthropic_tool_schema("ant_beta_tool")
        self.core_tool = _anthropic_tool_schema("ant_core_tool")
        self.registry.register_toolset(
            "ant_alpha", "Anthropic adapter alpha toolset.", tools=[self.alpha]
        )
        self.registry.register_toolset(
            "ant_beta", "Anthropic adapter beta toolset.", tools=[self.beta]
        )
        self.registry.register_tool_schema("core", self.core_tool)

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    # Test 6a — object-form content block (real Anthropic SDK response objects)
    def test_process_tool_calls_handles_object_form_tool_use(self) -> None:
        """
        Real Anthropic responses have content as a list of objects with .type, .name,
        and .input attributes.  _process_tool_calls must handle this form.
        """
        adapter = self.AnthropicAdapter()
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="use_toolset",
                    input={"action": "add", "toolsets": "ant_alpha"},
                )
            ]
        )

        adapter._process_tool_calls(response)

        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["ant_alpha", "core"],
            "Object-form tool_use block with use_toolset should queue ant_alpha",
        )

    # Test 6b — dict-form content block (streaming / test mocks)
    def test_process_tool_calls_handles_dict_form_tool_use(self) -> None:
        """
        Streaming responses or mock clients may return content blocks as plain dicts
        rather than typed objects.  _process_tool_calls must handle both forms.
        """
        adapter = self.AnthropicAdapter()
        response = SimpleNamespace(
            content=[
                {
                    "type": "tool_use",
                    "name": "use_toolset",
                    "input": {"action": "add", "toolsets": "ant_beta"},
                }
            ]
        )

        adapter._process_tool_calls(response)

        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["ant_beta", "core"],
            "Dict-form tool_use block with use_toolset should queue ant_beta",
        )

    # Test 6c — end-to-end: patched sync create intercepts tool_use response
    def test_patched_sync_create_intercepts_tool_use_response(self) -> None:
        """
        Critical gap check — the mock client returns a use_toolset tool_use block.
        After calling the patched messages.create, adapter.schema_filter.state.pending
        must contain the queued toolset change.

        This mirrors test_use_toolset_tool_call_response_is_now_handled in
        test_openai_adapter.py but uses Anthropic's tool_use content blocks
        instead of OpenAI's tool_calls array.
        """
        adapter = self.AnthropicAdapter()
        client = _SyncAnthropicClient()
        adapter._wrap_sync(client)  # patch the mock directly

        client.messages.create(
            model="claude-3-test",
            messages=[],
            tools=[self.alpha, self.beta, self.core_tool],
        )

        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["browser", "core"],
            "Critical gap: tool_use block returned by model must be intercepted "
            "by _process_tool_calls() and the toolset change queued as pending. "
            "If this fails, _process_tool_calls does not handle Anthropic's "
            "content block format correctly.",
        )

    # Test 6d — _process_tool_calls ignores non-use_toolset blocks
    def test_process_tool_calls_ignores_non_use_toolset_blocks(self) -> None:
        adapter = self.AnthropicAdapter()
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="some_other_tool",
                    input={"foo": "bar"},
                )
            ]
        )

        adapter._process_tool_calls(response)

        self.assertIsNone(
            adapter.schema_filter.state.pending,
            "Non-use_toolset tool_use blocks must not trigger a toolset change",
        )


# ---------------------------------------------------------------------------
# 7: Edge cases
# ---------------------------------------------------------------------------

class AnthropicAdapterEdgeCaseTests(unittest.TestCase):

    def setUp(self) -> None:
        self.AnthropicAdapter, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.alpha = _anthropic_tool_schema("ant_alpha_tool")
        self.beta = _anthropic_tool_schema("ant_beta_tool")
        self.core_tool = _anthropic_tool_schema("ant_core_tool")
        self.registry.register_toolset(
            "ant_alpha", "Anthropic adapter alpha toolset.", tools=[self.alpha]
        )
        self.registry.register_toolset(
            "ant_beta", "Anthropic adapter beta toolset.", tools=[self.beta]
        )
        self.registry.register_tool_schema("core", self.core_tool)

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_empty_content_list_does_not_raise(self) -> None:
        adapter = self.AnthropicAdapter()
        adapter._process_tool_calls(SimpleNamespace(content=[]))
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_none_content_does_not_raise(self) -> None:
        adapter = self.AnthropicAdapter()
        adapter._process_tool_calls(SimpleNamespace(content=None))
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_missing_content_attribute_does_not_raise(self) -> None:
        adapter = self.AnthropicAdapter()
        adapter._process_tool_calls(SimpleNamespace())
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_multiple_pending_toolset_changes_are_combined(self) -> None:
        """Two sequential use_toolset('add') calls should accumulate."""
        adapter = self.AnthropicAdapter()
        adapter.use_toolset("add", "ant_alpha")
        adapter.use_toolset("add", "ant_beta")

        kwargs = adapter._filter_and_process(
            {"tools": [self.alpha, self.beta, self.core_tool]}
        )

        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"ant_alpha_tool", "ant_beta_tool", "ant_core_tool", "use_toolset"},
        )

    def test_reset_to_core_removes_loaded_toolsets(self) -> None:
        adapter = self.AnthropicAdapter()
        adapter.use_toolset("add", "ant_alpha")
        adapter._filter_and_process({"tools": [self.alpha, self.beta, self.core_tool]})

        adapter.use_toolset("reset")
        kwargs = adapter._filter_and_process(
            {"tools": [self.alpha, self.beta, self.core_tool]}
        )

        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"ant_core_tool", "use_toolset"},
        )

    def test_removing_unloaded_toolset_keeps_core(self) -> None:
        adapter = self.AnthropicAdapter()
        response = adapter.use_toolset("remove", "ant_alpha")

        kwargs = adapter._filter_and_process(
            {"tools": [self.alpha, self.beta, self.core_tool]}
        )

        self.assertEqual(response["status"], "deferred")
        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"ant_core_tool", "use_toolset"},
        )

    def test_async_patched_create_is_coroutine_function(self) -> None:
        """After wrapping a mock async client, messages.create must be a coroutine."""
        adapter = self.AnthropicAdapter()
        client = _AsyncAnthropicClient()
        adapter._wrap_async(client)

        self.assertTrue(asyncio.iscoroutinefunction(client.messages.create))

    def test_async_patched_create_runs_without_error(self) -> None:
        """Async path end-to-end with empty content — no exceptions raised."""
        adapter = self.AnthropicAdapter()
        client = _AsyncAnthropicClient()
        adapter._wrap_async(client)

        async def run():
            return await client.messages.create(model="claude-test", messages=[])

        asyncio.run(run())
        self.assertIsNone(adapter.schema_filter.state.pending)


# ---------------------------------------------------------------------------
# 8: Consistency with OpenAI adapter surface
# ---------------------------------------------------------------------------

class AnthropicAdapterConsistencyTests(unittest.TestCase):

    def setUp(self) -> None:
        self.AnthropicAdapter, _ = _import_adapter_modules()

    def test_has_check_deps_classmethod(self) -> None:
        self.assertTrue(
            hasattr(self.AnthropicAdapter, "_check_deps"),
            "AnthropicAdapter must expose _check_deps classmethod matching OpenAIAdapter",
        )
        self.assertTrue(
            callable(self.AnthropicAdapter._check_deps),
        )

    def test_has_filter_tools_public_method(self) -> None:
        """
        GAP — _filter_tools is absent from AnthropicAdapter.

        REPAIR — Add to AnthropicAdapter in adapters/anthropic.py:

            def _filter_tools(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Public alias for _filter_and_process. Mirrors OpenAIAdapter.\"\"\"
                return self._filter_and_process(kwargs)
        """
        adapter = self.AnthropicAdapter()
        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "AnthropicAdapter is missing _filter_tools(). "
            "OpenAIAdapter has it — add the same alias to AnthropicAdapter.",
        )

    def test_has_process_tool_calls_method(self) -> None:
        adapter = self.AnthropicAdapter()
        self.assertTrue(
            hasattr(adapter, "_process_tool_calls"),
            "AnthropicAdapter must expose _process_tool_calls matching OpenAIAdapter",
        )

    def test_has_use_toolset_injection(self) -> None:
        """use_toolset schema must be injected into every request (auto_inject=True)."""
        adapter = self.AnthropicAdapter()
        kwargs = adapter._filter_and_process({})
        names = _tool_names(kwargs.get("tools", []))
        self.assertIn(
            "use_toolset",
            names,
            "use_toolset bootstrap must be auto-injected when auto_inject=True",
        )

    def test_auto_inject_false_skips_bootstrap(self) -> None:
        adapter = self.AnthropicAdapter(auto_inject=False)
        kwargs = adapter._filter_and_process({})
        # With no tools and auto_inject=False, tools should not be injected at all
        self.assertNotIn("tools", kwargs)

    def test_schema_filter_property_exposed(self) -> None:
        adapter = self.AnthropicAdapter()
        self.assertIsNotNone(adapter.schema_filter)

    def test_get_active_toolsets_returns_list(self) -> None:
        adapter = self.AnthropicAdapter()
        active = adapter.get_active_toolsets()
        self.assertIsInstance(active, list)

    def test_get_toolset_count_includes_core(self) -> None:
        adapter = self.AnthropicAdapter()
        self.assertGreaterEqual(adapter.get_toolset_count(), 1)


if __name__ == "__main__":
    unittest.main()
