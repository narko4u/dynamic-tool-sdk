"""Tests for the Dynamic Tool SDK LiteLLM adapter.

These tests mirror the OpenAI/Anthropic adapter suites and cover both the
public import contract and the module-level LiteLLM wrapping pattern.

Known repair targets are encoded as failing assertions with inline repair hints.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.abc
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


SDK_ROOT = Path(__file__).resolve().parents[1]  # tests/ -> sdk/
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
    adapter_module = importlib.import_module("dynamic_tool_sdk.adapters.litellm")
    registry_module = importlib.import_module("dynamic_tool_sdk.core.registry")
    return adapter_module.LiteLLMAdapter, registry_module


def _openai_tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_names(tools: list[dict]) -> list[str]:
    return [tool.get("function", {}).get("name") or tool.get("name") for tool in tools]


def _object_tool_call_response(toolset: str = "lite_alpha") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="use_toolset",
                                arguments=f'{{"action":"add","toolsets":"{toolset}"}}',
                            )
                        )
                    ]
                )
            )
        ]
    )


def _dict_tool_call_response(toolset: str = "lite_alpha") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "use_toolset",
                                "arguments": f'{{"action":"add","toolsets":"{toolset}"}}',
                            },
                        }
                    ]
                }
            }
        ]
    }


class _LiteLLMModuleMock:
    def __init__(self, response=None) -> None:
        self.calls: list[dict] = []
        self.async_calls: list[dict] = []
        self.response = response if response is not None else _object_tool_call_response()

    def completion(self, *args, **kwargs):  # noqa: ANN001
        self.calls.append(kwargs)
        return self.response

    async def acompletion(self, *args, **kwargs):  # noqa: ANN001
        self.async_calls.append(kwargs)
        return self.response


class _CountingLiteLLMAdapterMixin:
    def _make_counting_adapter(self):
        adapter = self.LiteLLMAdapter()
        adapter.process_count = 0
        original = adapter._process_tool_calls

        def counting_process(response):
            adapter.process_count += 1
            return original(response)

        adapter._process_tool_calls = counting_process
        return adapter


# ---------------------------------------------------------------------------
# 1 & 2: Import contract tests
# ---------------------------------------------------------------------------


class LiteLLMAdapterImportTests(unittest.TestCase):
    def test_litellm_package_is_installed(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import litellm, importlib.metadata; print(importlib.metadata.version('litellm'))"],
            cwd=SDK_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "litellm is not installed. Repair: run `pip install litellm`.\n"
            f"stderr:\n{result.stderr}",
        )

    def test_documented_dynamic_tool_sdk_import_path_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from dynamic_tool_sdk.adapters.litellm import LiteLLMAdapter; print(LiteLLMAdapter)",
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

    def test_missing_litellm_dependency_raises_AdapterNotInstalled(self) -> None:
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
                if name == "litellm" or name.startswith("litellm."):
                    del sys.modules[name]

            class BlockLiteLLM(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "litellm" or fullname.startswith("litellm."):
                        raise ModuleNotFoundError("simulated missing litellm")
                    return None

            sys.meta_path.insert(0, BlockLiteLLM())
            try:
                mod = importlib.import_module("dynamic_tool_sdk.adapters.litellm")
                mod.LiteLLMAdapter()
            except ModuleNotFoundError as exc:
                print(type(exc).__name__, exc)
                raise SystemExit(1)
            except Exception as exc:
                print(type(exc).__name__, exc)
                if type(exc).__name__ == "AdapterNotInstalled":
                    raise SystemExit(0)
                raise SystemExit(2)
            else:
                print("adapter instantiated without litellm — _check_deps may be broken")
                raise SystemExit(3)
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
            "LiteLLMAdapter must raise AdapterNotInstalled, not raw ModuleNotFoundError, "
            "when litellm is absent. Repair: keep imports deferred and catch ImportError "
            "inside _check_deps().\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


# ---------------------------------------------------------------------------
# 3: Module-level wrapping pattern
# ---------------------------------------------------------------------------


class LiteLLMAdapterWrappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.LiteLLMAdapter, self.registry = _import_adapter_modules()

    def test_wrap_module_patches_completion_and_acompletion_with_wraps(self) -> None:
        adapter = self.LiteLLMAdapter()
        mod = _LiteLLMModuleMock(response=SimpleNamespace(choices=[]))
        original_completion = mod.completion
        original_acompletion = mod.acompletion

        wrapped_mod = adapter.wrap_module(mod)

        self.assertIs(wrapped_mod, mod)
        self.assertIsNot(mod.completion, original_completion)
        self.assertIsNot(mod.acompletion, original_acompletion)
        self.assertEqual(mod.completion.__wrapped__, original_completion)
        self.assertEqual(mod.acompletion.__wrapped__, original_acompletion)
        self.assertTrue(asyncio.iscoroutinefunction(mod.acompletion))

    def test_public_wrap_completion_patches_litellm_completion(self) -> None:
        import litellm

        adapter = self.LiteLLMAdapter()
        original = litellm.completion
        self.addCleanup(setattr, litellm, "completion", original)

        self.assertTrue(
            hasattr(adapter, "wrap_completion"),
            "LiteLLMAdapter is missing public wrap_completion(). Repair: add "
            "`def wrap_completion(self, mod=None):` that defaults to the litellm "
            "module and patches mod.completion via _wrap_completion().",
        )
        adapter.wrap_completion(litellm)

        self.assertIsNot(litellm.completion, original)
        self.assertEqual(litellm.completion.__wrapped__, original)

    def test_public_wrap_acompletion_patches_litellm_acompletion(self) -> None:
        import litellm

        adapter = self.LiteLLMAdapter()
        original = litellm.acompletion
        self.addCleanup(setattr, litellm, "acompletion", original)

        self.assertTrue(
            hasattr(adapter, "wrap_acompletion"),
            "LiteLLMAdapter is missing public wrap_acompletion(). Repair: add "
            "`def wrap_acompletion(self, mod=None):` that defaults to the litellm "
            "module and patches mod.acompletion via _wrap_acompletion().",
        )
        adapter.wrap_acompletion(litellm)

        self.assertIsNot(litellm.acompletion, original)
        self.assertEqual(litellm.acompletion.__wrapped__, original)
        self.assertTrue(asyncio.iscoroutinefunction(litellm.acompletion))


# ---------------------------------------------------------------------------
# 4: OpenAI-format tool schemas and bootstrap injection
# ---------------------------------------------------------------------------


class LiteLLMAdapterFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.LiteLLMAdapter, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.alpha = _openai_tool_schema("lite_alpha_tool")
        self.beta = _openai_tool_schema("lite_beta_tool")
        self.core_tool = _openai_tool_schema("lite_core_tool")
        self.registry.register_toolset(
            "lite_alpha", "LiteLLM adapter alpha toolset.", tools=[self.alpha]
        )
        self.registry.register_toolset(
            "lite_beta", "LiteLLM adapter beta toolset.", tools=[self.beta]
        )
        self.registry.register_tool_schema("core", self.core_tool)

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_filter_tools_public_alias_exists(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "LiteLLMAdapter is missing _filter_tools(). Repair: add the same "
            "public alias used by OpenAIAdapter: `return self._filter_and_process(kwargs)`.",
        )

    def test_openai_format_filtering_keeps_active_toolsets_and_bootstrap(self) -> None:
        adapter = self.LiteLLMAdapter()
        adapter.use_toolset("add", "lite_alpha")

        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "LiteLLMAdapter is missing _filter_tools(). Repair: add `def _filter_tools(self, kwargs): return self._filter_and_process(kwargs)`.",
        )
        kwargs = adapter._filter_tools({"tools": [self.alpha, self.beta, self.core_tool]})

        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"lite_alpha_tool", "lite_core_tool", "use_toolset"},
        )

    def test_empty_tools_list_injects_only_bootstrap(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "LiteLLMAdapter is missing _filter_tools(). Repair: add `def _filter_tools(self, kwargs): return self._filter_and_process(kwargs)`.",
        )
        kwargs = adapter._filter_tools({"tools": []})
        self.assertEqual(_tool_names(kwargs["tools"]), ["use_toolset"])

    def test_tools_none_injects_only_bootstrap(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "LiteLLMAdapter is missing _filter_tools(). Repair: add `def _filter_tools(self, kwargs): return self._filter_and_process(kwargs)`.",
        )
        kwargs = adapter._filter_tools({})
        self.assertEqual(_tool_names(kwargs["tools"]), ["use_toolset"])

    def test_auto_inject_false_skips_bootstrap(self) -> None:
        adapter = self.LiteLLMAdapter(auto_inject=False)
        kwargs = adapter._filter_and_process({})
        self.assertNotIn("tools", kwargs)


# ---------------------------------------------------------------------------
# 5: Streaming behavior
# ---------------------------------------------------------------------------


class LiteLLMAdapterStreamingTests(_CountingLiteLLMAdapterMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.LiteLLMAdapter, self.registry = _import_adapter_modules()

    def test_stream_true_skips_process_tool_calls_for_sync_completion(self) -> None:
        adapter = self._make_counting_adapter()
        mod = _LiteLLMModuleMock(response=(chunk for chunk in []))
        adapter.wrap_module(mod)

        response = mod.completion(model="gpt-test", messages=[], stream=True)

        self.assertTrue(hasattr(response, "__iter__"))
        self.assertEqual(
            adapter.process_count,
            0,
            "Streaming LiteLLM responses are generators and cannot be inspected for "
            "final tool_calls here. Repair: in _wrap_completion, call "
            "_process_tool_calls(response) only when kwargs.get('stream') is not True.",
        )

    def test_stream_false_processes_tool_calls_for_sync_completion(self) -> None:
        adapter = self._make_counting_adapter()
        mod = _LiteLLMModuleMock(response=SimpleNamespace(choices=[]))
        adapter.wrap_module(mod)

        mod.completion(model="gpt-test", messages=[], stream=False)

        self.assertEqual(adapter.process_count, 1)

    def test_stream_true_skips_process_tool_calls_for_async_completion(self) -> None:
        adapter = self._make_counting_adapter()
        mod = _LiteLLMModuleMock(response=(chunk for chunk in []))
        adapter.wrap_module(mod)

        async def run():
            return await mod.acompletion(model="gpt-test", messages=[], stream=True)

        response = asyncio.run(run())

        self.assertTrue(hasattr(response, "__iter__"))
        self.assertEqual(
            adapter.process_count,
            0,
            "Streaming LiteLLM async responses should also skip _process_tool_calls. "
            "Repair: guard _wrap_acompletion with kwargs.get('stream') is not True.",
        )

    def test_stream_default_processes_tool_calls_for_async_completion(self) -> None:
        adapter = self._make_counting_adapter()
        mod = _LiteLLMModuleMock(response=SimpleNamespace(choices=[]))
        adapter.wrap_module(mod)

        async def run():
            return await mod.acompletion(model="gpt-test", messages=[])

        asyncio.run(run())

        self.assertEqual(adapter.process_count, 1)


# ---------------------------------------------------------------------------
# 6: tool_call response handling
# ---------------------------------------------------------------------------


class LiteLLMAdapterToolCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.LiteLLMAdapter, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.alpha = _openai_tool_schema("lite_alpha_tool")
        self.beta = _openai_tool_schema("lite_beta_tool")
        self.core_tool = _openai_tool_schema("lite_core_tool")
        self.registry.register_toolset(
            "lite_alpha", "LiteLLM adapter alpha toolset.", tools=[self.alpha]
        )
        self.registry.register_toolset(
            "lite_beta", "LiteLLM adapter beta toolset.", tools=[self.beta]
        )
        self.registry.register_tool_schema("core", self.core_tool)

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_process_tool_calls_handles_object_form_openai_response(self) -> None:
        adapter = self.LiteLLMAdapter()

        adapter._process_tool_calls(_object_tool_call_response("lite_alpha"))

        self.assertEqual(adapter.schema_filter.state.pending, ["core", "lite_alpha"])

    def test_process_tool_calls_handles_dict_form_openai_response(self) -> None:
        adapter = self.LiteLLMAdapter()

        adapter._process_tool_calls(_dict_tool_call_response("lite_beta"))

        self.assertEqual(
            adapter.schema_filter.state.pending,
            ["core", "lite_beta"],
            "LiteLLM responses and tests may be plain OpenAI-format dicts. Repair: "
            "mirror OpenAIAdapter._process_tool_calls by supporting both dict and object "
            "choices/messages/tool_calls/functions.",
        )

    def test_patched_completion_intercepts_use_toolset_response(self) -> None:
        adapter = self.LiteLLMAdapter()
        mod = _LiteLLMModuleMock(response=_object_tool_call_response("lite_alpha"))
        adapter.wrap_module(mod)

        mod.completion(
            model="gpt-test",
            messages=[],
            tools=[self.alpha, self.beta, self.core_tool],
        )

        self.assertEqual(adapter.schema_filter.state.pending, ["core", "lite_alpha"])

    def test_process_tool_calls_ignores_non_use_toolset(self) -> None:
        adapter = self.LiteLLMAdapter()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="some_other_tool",
                                    arguments='{"foo":"bar"}',
                                )
                            )
                        ]
                    )
                )
            ]
        )

        adapter._process_tool_calls(response)

        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_malformed_tool_call_arguments_do_not_raise(self) -> None:
        adapter = self.LiteLLMAdapter()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="use_toolset",
                                    arguments="{not-json",
                                )
                            )
                        ]
                    )
                )
            ]
        )

        adapter._process_tool_calls(response)

        self.assertIsNone(adapter.schema_filter.state.pending)


# ---------------------------------------------------------------------------
# 7: Consistency with OpenAI/Anthropic adapter surface
# ---------------------------------------------------------------------------


class LiteLLMAdapterConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.LiteLLMAdapter, _ = _import_adapter_modules()

    def test_has_check_deps_classmethod(self) -> None:
        self.assertTrue(hasattr(self.LiteLLMAdapter, "_check_deps"))
        self.assertTrue(callable(self.LiteLLMAdapter._check_deps))

    def test_has_filter_tools_public_alias(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertTrue(
            hasattr(adapter, "_filter_tools"),
            "LiteLLMAdapter must expose _filter_tools matching OpenAIAdapter/AnthropicAdapter.",
        )

    def test_has_process_tool_calls_method(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertTrue(hasattr(adapter, "_process_tool_calls"))

    def test_schema_filter_property_exposed(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertIsNotNone(adapter.schema_filter)

    def test_use_toolset_injection_enabled_by_default(self) -> None:
        adapter = self.LiteLLMAdapter()
        kwargs = adapter._filter_and_process({})
        self.assertIn("use_toolset", _tool_names(kwargs["tools"]))

    def test_get_active_toolsets_returns_list(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertIsInstance(adapter.get_active_toolsets(), list)

    def test_get_toolset_count_includes_core(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertGreaterEqual(adapter.get_toolset_count(), 1)

    def test_public_wrap_completion_methods_exist(self) -> None:
        adapter = self.LiteLLMAdapter()
        self.assertTrue(
            hasattr(adapter, "wrap_completion"),
            "LiteLLMAdapter should expose wrap_completion() for module-level litellm.completion.",
        )
        self.assertTrue(
            hasattr(adapter, "wrap_acompletion"),
            "LiteLLMAdapter should expose wrap_acompletion() for module-level litellm.acompletion.",
        )


# ---------------------------------------------------------------------------
# 8: Edge cases
# ---------------------------------------------------------------------------


class LiteLLMAdapterEdgeCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.LiteLLMAdapter, self.registry = _import_adapter_modules()
        self._toolsets_before = copy.deepcopy(self.registry._TOOLSETS)
        self.alpha = _openai_tool_schema("lite_alpha_tool")
        self.beta = _openai_tool_schema("lite_beta_tool")
        self.core_tool = _openai_tool_schema("lite_core_tool")
        self.registry.register_toolset(
            "lite_alpha", "LiteLLM adapter alpha toolset.", tools=[self.alpha]
        )
        self.registry.register_toolset(
            "lite_beta", "LiteLLM adapter beta toolset.", tools=[self.beta]
        )
        self.registry.register_tool_schema("core", self.core_tool)

    def tearDown(self) -> None:
        self.registry._TOOLSETS.clear()
        self.registry._TOOLSETS.update(self._toolsets_before)

    def test_empty_choices_does_not_raise(self) -> None:
        adapter = self.LiteLLMAdapter()
        adapter._process_tool_calls(SimpleNamespace(choices=[]))
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_none_choices_does_not_raise(self) -> None:
        adapter = self.LiteLLMAdapter()
        adapter._process_tool_calls(SimpleNamespace(choices=None))
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_missing_choices_attribute_does_not_raise(self) -> None:
        adapter = self.LiteLLMAdapter()
        adapter._process_tool_calls(SimpleNamespace())
        self.assertIsNone(adapter.schema_filter.state.pending)

    def test_multiple_pending_toolset_changes_are_combined(self) -> None:
        adapter = self.LiteLLMAdapter()
        adapter.use_toolset("add", "lite_alpha")
        adapter.use_toolset("add", "lite_beta")

        kwargs = adapter._filter_and_process(
            {"tools": [self.alpha, self.beta, self.core_tool]}
        )

        self.assertEqual(
            set(_tool_names(kwargs["tools"])),
            {"lite_alpha_tool", "lite_beta_tool", "lite_core_tool", "use_toolset"},
        )

    def test_reset_to_core_removes_loaded_toolsets(self) -> None:
        adapter = self.LiteLLMAdapter()
        adapter.use_toolset("add", "lite_alpha")
        adapter._filter_and_process({"tools": [self.alpha, self.beta, self.core_tool]})

        adapter.use_toolset("reset")
        kwargs = adapter._filter_and_process(
            {"tools": [self.alpha, self.beta, self.core_tool]}
        )

        self.assertEqual(set(_tool_names(kwargs["tools"])), {"lite_core_tool", "use_toolset"})

    def test_removing_unloaded_toolset_keeps_core(self) -> None:
        adapter = self.LiteLLMAdapter()
        response = adapter.use_toolset("remove", "lite_alpha")

        kwargs = adapter._filter_and_process(
            {"tools": [self.alpha, self.beta, self.core_tool]}
        )

        self.assertEqual(response["status"], "deferred")
        self.assertEqual(set(_tool_names(kwargs["tools"])), {"lite_core_tool", "use_toolset"})

    def test_async_wrapped_acompletion_runs_without_error(self) -> None:
        adapter = self.LiteLLMAdapter()
        mod = _LiteLLMModuleMock(response=SimpleNamespace(choices=[]))
        adapter.wrap_module(mod)

        async def run():
            return await mod.acompletion(model="gpt-test", messages=[])

        asyncio.run(run())
        self.assertIsNone(adapter.schema_filter.state.pending)


if __name__ == "__main__":
    unittest.main()

