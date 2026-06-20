# License Overview

The Dynamic Tool SDK repository contains two distinct layers with different licensing terms:

## MIT Licensed — Public Surface

The following files are open source under the MIT License (see `LICENSE`):

- **Adapters:** `dynamic_tool_sdk/adapters/*.py` — Platform-specific wrapping code (OpenAI, Anthropic, LiteLLM, LangChain)
- **Tests:** `tests/*.py` — Full test suites demonstrating adapter behaviour
- **TypeScript adapter:** `adapters/vercel-ai.ts` — Vercel AI SDK integration
- **TypeScript tests:** `tests/vercel-ai.test.ts`
- **Configuration:** `pyproject.toml`, `package.json`, `tsconfig.json`
- **This document**

You are free to use, modify, and distribute the adapter code under the terms of the MIT License.

## Proprietary — Core Engine

The following directories contain the **proprietary core technology** of the Dynamic Tool SDK and are **not** covered by the MIT License:

- `dynamic_tool_sdk/core/` — Schema registry, state machine, filter pipeline, deferred-execution architecture
- `dynamic_tool_sdk/registry/` — Toolset registry implementation

These components are included in this repository for **evaluation purposes only**. They may be:
- ✅ Inspected to understand the SDK's architecture
- ✅ Tested via the included test suites
- ❌ NOT used in production without a license agreement
- ❌ NOT copied, modified, or redistributed independently
- ❌ NOT incorporated into other projects

## Licensing Inquiries

For production licenses, enterprise agreements, or investment discussions:

**Empire Labs Pty Ltd**
contact@empirelabs.com.au
https://empirelabs.com.au
