"""Tests for the Dynamic Tool SDK OpenAI adapter.

These tests intentionally cover both the public import contract and the
adapter mechanics. If the import-contract tests fail, the repair is in
packaging/import guards rather than the filtering logic.
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


SDK_ROOT = Path(__file__).resolve().parents[1]  # tests/ -> sdk/
PACKAGE_ROOT = SDK_ROOT / "dynamic_tool_sdk"


def _install_local_dynamic_tool_sdk_alias() -> None:
    """Expose this source tree as dynamic_tool_sdk for local tests."""
    package = sys.modules.get("dynamic_tool_sdk")
    if package is None:
        package = types.ModuleType("dynamic_tool_sdk")
        package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
        sys.modules["dynamic_tool_sdk"] = package


def _import_adapter_modules():
    _install_local_dynamic_tool_sdk_alias()
    adapter_module = importlib.import_module("dynamic_tool_sdk.adapters.openai")
    registry_module = importlib.import_module("dynamic_tool_sdk.core.registry")
    return adapter_module.OpenAIAdapter, registry_module


def _tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_names(tools: list[dict]) -> list[str]:
    return [tool.get("function", {}).get("name") for tool in tools]


class _SyncCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, *args, **kwargs):  # noqa: ANN001
        self.calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "use_toolset",
                                    "arguments": '{"action":"add","toolsets":"browser"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }


class _SyncClient:
    def __init__(self) -> None:
        self.chat = types.SimpleNamespace(completions=_SyncCompletions())


class OpenAIAdapterImportTests(unittest.TestCase):
    def test_openai_package_is_installed(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import openai; print(openai.__version__)"],
            cwd=SDK_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "openai is not installed. Repair: run `pip install openai` in this environment.",
        )

    def test_documented_dynamic_tool_sdk_import_path_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from dynamic_tool_sdk.adapters.openai import OpenAIAdapter; print(OpenAIAdapter)",
            ],
            cwd=SDK_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "Documented import failed. Repair: package the SDK under a "
            "`dynamic_tool_sdk/` package, add the SDK source root to packaging "
            "metadata, or provide a compatibility package that points at `sdk/`.\n"
            f"stderr:\n{result.stderr}",
        )

    def test_missing_openai_dependency_is_handled_gracefully(self) -> None:
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
                if name == "openai" or name.startswith("openai."):
                    del sys.modules[name]

            class BlockOpenAI(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "openai" or fullname.startswith("openai."):
                        raise ModuleNotFoundError("simulated missing openai")
                    return None

            sys.meta_path.insert(0, BlockOpenAI())
            try:
                importlib.import_module("dynamic_tool_sdk.adapters.openai")
            except ModuleNotFoundError as exc:
                print(type(exc).__name__, exc)
                raise SystemExit(1)
            except Exception as exc:
                print(type(exc).__name__, exc)
                raise SystemExit(0)
            else:
                print("import ok without openai")
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
            "Adapter raises raw ModuleNotFoundError when openai is missing. "
            "Repair: wrap OpenAI imports in try/except and raise AdapterNotInstalled "
            "from wrap_client/__init__, or defer OpenAI type imports behind "
            "`typing.TYPE_CHECKING`.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


class OpenAIAdapterBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.OpenAIAdapter, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.alpha = _tool_schema("adapter_alpha_tool")
        self.beta = _tool_schema("adapter_beta_tool")
        self.core = _tool_schema("adapter_core_tool")
        self.registry.register_toolset(
            "adapter_alpha", "Adapter alpha test toolset.", tools=[self.alpha]
        )
        self.registry.register_toolset(
            "adapter_beta", "Adapter beta test toolset.", tools=[self.beta]
        )
        self.registry.register_tool_schema("core", self.core)

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_real_openai_client_is_wrapped_and_create_is_patched(self) -> None:
        import openai

        client = openai.OpenAI(api_key="sk-test")
        original_create = client.chat.completions.create
        adapter = self.OpenAIAdapter()

        wrapped = adapter.wrap_client(client)

        self.assertIs(wrapped, client)
        self.assertIsNot(client.chat.completions.create, original_create)
        # functools.wraps preserves the original function name
        self.assertEqual(client.chat.completions.create.__wrapped__, original_create)

    def test_tool_filtering_keeps_active_toolsets_and_bootstrap(self) -> None:
        adapter = self.OpenAIAdapter()
        adapter.use_toolset("add", "adapter_alpha")

        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core]})

        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"adapter_alpha_tool", "adapter_core_tool", "use_toolset"},
        )

    def test_empty_tools_list_injects_only_bootstrap(self) -> None:
        adapter = self.OpenAIAdapter()

        kwargs = adapter._filter_tools({"tools": []})

        self.assertEqual(_tool_names(kwargs["tools"]), ["use_toolset"])

    def test_tools_none_injects_only_bootstrap(self) -> None:
        adapter = self.OpenAIAdapter()

        kwargs = adapter._filter_tools({})

        self.assertEqual(_tool_names(kwargs["tools"]), ["use_toolset"])

    def test_async_client_is_wrapped_and_create_is_patched(self) -> None:
        import openai

        client = openai.AsyncOpenAI(api_key="sk-test")
        original_create = client.chat.completions.create
        adapter = self.OpenAIAdapter()

        wrapped = adapter.wrap_client(client)

        self.assertIs(wrapped, client)
        self.assertIsNot(client.chat.completions.create, original_create)
        self.assertTrue(asyncio.iscoroutinefunction(client.chat.completions.create))

    def test_multiple_pending_toolset_changes_are_combined(self) -> None:
        adapter = self.OpenAIAdapter()

        adapter.use_toolset("add", "adapter_alpha")
        adapter.use_toolset("add", "adapter_beta")
        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core]})

        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"adapter_alpha_tool", "adapter_beta_tool", "adapter_core_tool", "use_toolset"},
        )

    def test_reset_to_core_filters_out_loaded_toolsets(self) -> None:
        adapter = self.OpenAIAdapter()
        adapter.use_toolset("add", "adapter_alpha")
        adapter._filter_tools({"tools": [self.alpha, self.beta, self.core]})

        adapter.use_toolset("reset")
        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core]})

        self.assertEqual(set(_tool_names(kwargs["tools"])), {"adapter_core_tool", "use_toolset"})

    def test_removing_toolset_that_is_not_loaded_keeps_core(self) -> None:
        adapter = self.OpenAIAdapter()

        response = adapter.use_toolset("remove", "adapter_alpha")
        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core]})

        self.assertEqual(response["status"], "deferred")
        self.assertEqual(set(_tool_names(kwargs["tools"])), {"adapter_core_tool", "use_toolset"})

    def test_use_toolset_tool_call_response_is_now_handled(self) -> None:
        """Patched create processes use_toolset tool_calls from the response."""
        adapter = self.OpenAIAdapter()
        client = _SyncClient()

        adapter.wrap_client(client)
        response = client.chat.completions.create(
            model="gpt-test",
            messages=[],
            tools=[self.alpha, self.beta, self.core],
        )

        # After the fix, the tool_call should have been processed
        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["browser", "core"],
            "Critical gap resolved: tool_call returned by model should have been "
            "intercepted and adapter.use_toolset() called.",
        )


if __name__ == "__main__":
    unittest.main()

