/**
 * Tests for the Dynamic Tool SDK Vercel AI adapter.
 *
 * Mirrors the structure of the Python adapter tests:
 *   - Import / dependency checks
 *   - filterTools() (mirrors Python _filter_tools())
 *   - processToolCalls() (mirrors Python _process_tool_calls())
 *   - wrapGenerateText() / wrapStreamText()
 *   - Edge cases
 *   - Consistency checks (same public surface as OpenAI / Anthropic adapters)
 *
 * No real LLM API calls are made.  generateText / streamText are mocked as
 * plain async functions returning controlled VercelToolCall fixtures.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  VercelAIAdapter,
  USE_TOOLSET_TOOL,
  type VercelTool,
  type VercelToolSet,
  type VercelToolCall,
  type GenerateParams,
  type GenerateTextResult,
  type StreamTextResult,
} from '../adapters/vercel-ai.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal VercelTool stub for tests. */
function mockTool(description = 'test tool'): VercelTool {
  return {
    description,
    parameters: { type: 'object', properties: {} },
  };
}

/** Extract the names of an active toolset from a filtered tools record. */
function toolNames(tools: VercelToolSet): string[] {
  return Object.keys(tools).sort();
}

/** Build a mock use_toolset tool call response item. */
function useToolsetCall(action: string, toolsets = ''): VercelToolCall {
  return { toolCallId: 'tc-1', toolName: 'use_toolset', args: { action, toolsets } };
}

/** Build a mock generateText function that records calls and returns a preset response. */
function mockGenerateText(toolCalls: VercelToolCall[] = []) {
  return vi.fn(async (params: GenerateParams): Promise<GenerateTextResult> => ({
    text: '',
    toolCalls,
    _capturedParams: params,
  }));
}

/** Build a mock streamText function whose toolCalls resolve asynchronously. */
function mockStreamText(toolCalls: VercelToolCall[] = []) {
  return vi.fn(async (params: GenerateParams): Promise<StreamTextResult> => ({
    _capturedParams: params,
    toolCalls: Promise.resolve(toolCalls),
  }));
}

// ---------------------------------------------------------------------------
// 1. Adapter construction + dependency check
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — construction', () => {
  it('instantiates without options', () => {
    const adapter = new VercelAIAdapter();
    expect(adapter).toBeInstanceOf(VercelAIAdapter);
  });

  it('accepts coreToolset and autoInject options', () => {
    const adapter = new VercelAIAdapter({ coreToolset: 'core', autoInject: false });
    expect(adapter.toolsetCount).toBe(1);
  });

  it('zod is installed and the use_toolset tool has a Zod schema', async () => {
    const { z } = await import('zod');
    expect(USE_TOOLSET_TOOL.parameters).toBeInstanceOf(z.ZodObject);
  });

  it('ai package is importable', async () => {
    const ai = await import('ai');
    expect(typeof ai.generateText).toBe('function');
    expect(typeof ai.streamText).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// 2. Toolset registration
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — registerToolset()', () => {
  it('registers a toolset and reflects it in listRegisteredToolsets()', () => {
    const adapter = new VercelAIAdapter();
    adapter.registerToolset('browser', ['navigate', 'click'], 'Browser tools');
    const list = adapter.listRegisteredToolsets();
    expect(list['browser']).toBe('Browser tools');
  });

  it('can register multiple toolsets', () => {
    const adapter = new VercelAIAdapter();
    adapter.registerToolset('browser', ['navigate']);
    adapter.registerToolset('email', ['send_email', 'read_email']);
    const list = adapter.listRegisteredToolsets();
    expect(Object.keys(list)).toContain('browser');
    expect(Object.keys(list)).toContain('email');
  });
});

// ---------------------------------------------------------------------------
// 3. useToolset() — state management
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — useToolset()', () => {
  let adapter: VercelAIAdapter;

  beforeEach(() => {
    adapter = new VercelAIAdapter();
    adapter.registerToolset('browser', ['navigate', 'click']);
    adapter.registerToolset('email',   ['send_email', 'read_email']);
  });

  it('add queues a pending change and returns deferred status', () => {
    const result = adapter.useToolset('add', 'browser');
    expect(result.status).toBe('deferred');
    expect(result.requested_toolsets).toContain('browser');
    expect(result.requested_toolsets).toContain('core');
  });

  it('add for unknown toolset returns an error', () => {
    const result = adapter.useToolset('add', 'nonexistent');
    expect(result.error).toBeDefined();
  });

  it('set replaces the active set (pending)', () => {
    adapter.useToolset('add', 'browser');
    const result = adapter.useToolset('set', 'email');
    expect(result.requested_toolsets).toContain('email');
    expect(result.requested_toolsets).not.toContain('browser');
  });

  it('remove queues removal even if toolset not currently active', () => {
    const result = adapter.useToolset('remove', 'browser');
    expect(result.status).toBe('deferred');
    // core must always remain
    expect(result.requested_toolsets).toContain('core');
  });

  it('reset clears pending and active back to core only', () => {
    adapter.useToolset('add', 'browser');
    const result = adapter.useToolset('reset');
    expect(result.status).toBe('ok');
    expect(result.active_toolsets).toEqual(['core']);
    expect(adapter.pendingToolsets).toBeNull();
  });

  it('list returns active toolsets', () => {
    const result = adapter.useToolset('list');
    expect(result.active_toolsets).toContain('core');
  });

  it('unknown action returns error', () => {
    const result = adapter.useToolset('explode', 'browser');
    expect(result.error).toBeDefined();
  });

  it('multiple add calls accumulate toolsets', () => {
    adapter.useToolset('add', 'browser');
    const result = adapter.useToolset('add', 'email');
    expect(result.requested_toolsets).toContain('browser');
    expect(result.requested_toolsets).toContain('email');
    expect(result.requested_toolsets).toContain('core');
  });
});

// ---------------------------------------------------------------------------
// 4. filterTools() — the main filtering entry point
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — filterTools()', () => {
  let adapter: VercelAIAdapter;

  beforeEach(() => {
    adapter = new VercelAIAdapter();
    adapter.registerToolset('alpha', ['alpha_search', 'alpha_read']);
    adapter.registerToolset('beta',  ['beta_write', 'beta_delete']);
    // Register core tools
    adapter.registerToolset('core',  ['core_ping']);
  });

  it('injects only use_toolset when no tools are provided', () => {
    const result = adapter.filterTools({});
    expect(result.tools).toBeDefined();
    expect(result.tools!['use_toolset']).toBeDefined();
    expect(Object.keys(result.tools!)).toHaveLength(1);
  });

  it('injects only use_toolset for an empty tools record', () => {
    const result = adapter.filterTools({ tools: {} });
    expect(Object.keys(result.tools!)).toEqual(['use_toolset']);
  });

  it('keeps tools from active toolsets + injects bootstrap', () => {
    adapter.useToolset('add', 'alpha');

    const tools: VercelToolSet = {
      alpha_search: mockTool(),
      alpha_read:   mockTool(),
      beta_write:   mockTool(),
      core_ping:    mockTool(),
    };

    const result = adapter.filterTools({ tools });
    const names  = toolNames(result.tools!);

    expect(names).toContain('alpha_search');
    expect(names).toContain('alpha_read');
    expect(names).toContain('core_ping');
    expect(names).toContain('use_toolset');
    expect(names).not.toContain('beta_write');
  });

  it('applies the pending change at filter time (deferred activation)', () => {
    // After useToolset('add', 'alpha'), the change is pending.
    // filterTools() calls applyPending() before filtering.
    adapter.useToolset('add', 'alpha');
    expect(adapter.pendingToolsets).toContain('alpha');

    adapter.filterTools({ tools: { alpha_search: mockTool() } });

    // Pending should be cleared after apply
    expect(adapter.pendingToolsets).toBeNull();
    expect(adapter.activeToolsets).toContain('alpha');
  });

  it('does not inject bootstrap when autoInject is false', () => {
    const noInject = new VercelAIAdapter({ autoInject: false });
    noInject.registerToolset('alpha', ['alpha_search']);

    const result = noInject.filterTools({});
    // With no tools and autoInject=false, tools key should not be added
    expect(result.tools).toBeUndefined();
  });

  it('skips bootstrap injection when use_toolset already present', () => {
    const tools: VercelToolSet = { use_toolset: mockTool('existing bootstrap') };
    const result = adapter.filterTools({ tools });
    // Should not have a duplicate use_toolset or replace the existing one
    expect(Object.keys(result.tools!).filter(k => k === 'use_toolset')).toHaveLength(1);
  });

  it('passes through all tools when no toolset schemas registered', () => {
    const fresh = new VercelAIAdapter();
    // No registerToolset calls — filter should pass everything through

    const tools: VercelToolSet = {
      tool_a: mockTool(),
      tool_b: mockTool(),
    };
    const result = fresh.filterTools({ tools });

    // Both tools should survive (pass-through when no registry info)
    expect(result.tools!['tool_a']).toBeDefined();
    expect(result.tools!['tool_b']).toBeDefined();
    expect(result.tools!['use_toolset']).toBeDefined(); // bootstrap still injected
  });

  it('preserves non-tools keys in params', () => {
    const result = adapter.filterTools({ model: 'gpt-4o', messages: [], tools: {} });
    expect((result as Record<string, unknown>)['model']).toBe('gpt-4o');
    expect((result as Record<string, unknown>)['messages']).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 5. processToolCalls() — response inspection
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — processToolCalls()', () => {
  let adapter: VercelAIAdapter;

  beforeEach(() => {
    adapter = new VercelAIAdapter();
    adapter.registerToolset('browser', ['navigate']);
    adapter.registerToolset('email',   ['send_email']);
  });

  it('queues add action from a use_toolset tool call', () => {
    adapter.processToolCalls([useToolsetCall('add', 'browser')]);
    expect(adapter.pendingToolsets).toContain('browser');
    expect(adapter.pendingToolsets).toContain('core');
  });

  it('applies reset from a use_toolset tool call', () => {
    adapter.useToolset('add', 'browser');
    // Apply and then get a reset tool call
    adapter.filterTools({});
    adapter.processToolCalls([useToolsetCall('reset')]);
    expect(adapter.activeToolsets).toEqual([]);
    expect(adapter.pendingToolsets).toBeNull();
  });

  it('ignores tool calls with names other than use_toolset', () => {
    adapter.processToolCalls([{ toolCallId: 'tc-2', toolName: 'navigate', args: { url: 'https://example.com' } }]);
    expect(adapter.pendingToolsets).toBeNull();
  });

  it('handles an empty tool calls array gracefully', () => {
    expect(() => adapter.processToolCalls([])).not.toThrow();
    expect(adapter.pendingToolsets).toBeNull();
  });

  it('handles malformed args without throwing', () => {
    const bad: VercelToolCall = { toolName: 'use_toolset', args: {} }; // missing action
    expect(() => adapter.processToolCalls([bad])).not.toThrow();
  });

  it('processes multiple tool calls in sequence', () => {
    adapter.processToolCalls([
      useToolsetCall('add', 'browser'),
      useToolsetCall('add', 'email'),
    ]);
    // Second add accumulates on top of first
    expect(adapter.pendingToolsets).toContain('browser');
    expect(adapter.pendingToolsets).toContain('email');
  });
});

// ---------------------------------------------------------------------------
// 6. wrapGenerateText()
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — wrapGenerateText()', () => {
  let adapter: VercelAIAdapter;

  beforeEach(() => {
    adapter = new VercelAIAdapter();
    adapter.registerToolset('browser', ['navigate', 'click']);
    adapter.registerToolset('email',   ['send_email']);
  });

  it('returns a function', () => {
    const wrapped = adapter.wrapGenerateText(mockGenerateText());
    expect(typeof wrapped).toBe('function');
  });

  it('calls the original function with filtered tools', async () => {
    adapter.useToolset('add', 'browser');

    const inner = mockGenerateText();
    const wrapped = adapter.wrapGenerateText(inner);

    await wrapped({
      tools: {
        navigate:   mockTool(),
        click:      mockTool(),
        send_email: mockTool(),
      },
    });

    expect(inner).toHaveBeenCalledOnce();
    const calledWith = inner.mock.calls[0][0] as GenerateParams;
    const names = toolNames(calledWith.tools ?? {});
    expect(names).toContain('navigate');
    expect(names).toContain('click');
    expect(names).not.toContain('send_email');
    expect(names).toContain('use_toolset');
  });

  it('processes use_toolset tool calls from the response', async () => {
    const inner   = mockGenerateText([useToolsetCall('add', 'browser')]);
    const wrapped = adapter.wrapGenerateText(inner);

    await wrapped({});

    expect(adapter.pendingToolsets).toContain('browser');
  });

  it('returns the original result object from the inner function', async () => {
    const inner  = mockGenerateText();
    const wrapped = adapter.wrapGenerateText(inner);
    const result  = await wrapped({});

    expect(result.text).toBe('');
    expect(Array.isArray(result.toolCalls)).toBe(true);
  });

  it('critical: use_toolset call in response queues change for next turn', async () => {
    const inner = mockGenerateText([useToolsetCall('add', 'email')]);
    const wrapped = adapter.wrapGenerateText(inner);

    // First call — response contains use_toolset('add', 'email')
    await wrapped({ tools: { navigate: mockTool() } });

    // Pending should now include 'email'
    expect(adapter.pendingToolsets).toContain('email');

    // Second call — filter applies the pending change, email tools should appear
    const inner2  = mockGenerateText();
    const wrapped2 = adapter.wrapGenerateText(inner2);

    await wrapped2({ tools: { navigate: mockTool(), send_email: mockTool() } });

    const calledWith = inner2.mock.calls[0][0] as GenerateParams;
    const names = toolNames(calledWith.tools ?? {});
    expect(names).toContain('send_email');
  });
});

// ---------------------------------------------------------------------------
// 7. wrapStreamText()
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — wrapStreamText()', () => {
  let adapter: VercelAIAdapter;

  beforeEach(() => {
    adapter = new VercelAIAdapter();
    adapter.registerToolset('browser', ['navigate']);
  });

  it('returns a function', () => {
    const wrapped = adapter.wrapStreamText(mockStreamText());
    expect(typeof wrapped).toBe('function');
  });

  it('calls the original streamText with filtered tools', async () => {
    adapter.useToolset('add', 'browser');
    const inner  = mockStreamText();
    const wrapped = adapter.wrapStreamText(inner);

    await wrapped({
      tools: {
        navigate:   mockTool(),
        send_email: mockTool(),
      },
    });

    expect(inner).toHaveBeenCalledOnce();
    const calledWith = inner.mock.calls[0][0] as GenerateParams;
    const names = toolNames(calledWith.tools ?? {});
    expect(names).toContain('navigate');
    expect(names).not.toContain('send_email');
  });

  it('processes use_toolset tool calls from the async stream result', async () => {
    const inner   = mockStreamText([useToolsetCall('add', 'browser')]);
    const wrapped = adapter.wrapStreamText(inner);

    await wrapped({});

    // Give the async Promise.resolve chain time to execute
    await new Promise(r => setTimeout(r, 0));

    expect(adapter.pendingToolsets).toContain('browser');
  });
});

// ---------------------------------------------------------------------------
// 8. Edge cases
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — edge cases', () => {
  let adapter: VercelAIAdapter;

  beforeEach(() => {
    adapter = new VercelAIAdapter();
    adapter.registerToolset('alpha', ['alpha_tool']);
    adapter.registerToolset('beta',  ['beta_tool']);
    adapter.registerToolset('core',  ['core_tool']);
  });

  it('reset then add works correctly across two turns', () => {
    adapter.useToolset('add', 'alpha');
    adapter.filterTools({});  // apply alpha

    adapter.useToolset('reset');
    adapter.filterTools({});  // apply reset

    adapter.useToolset('add', 'beta');
    const result = adapter.filterTools({
      tools: { alpha_tool: mockTool(), beta_tool: mockTool(), core_tool: mockTool() },
    });
    const names = toolNames(result.tools!);

    expect(names).toContain('beta_tool');
    expect(names).not.toContain('alpha_tool');
  });

  it('removing a toolset not in active set keeps core', () => {
    // alpha not active; remove it anyway
    const r = adapter.useToolset('remove', 'alpha');
    expect(r.status).toBe('deferred');
    adapter.filterTools({});
    expect(adapter.activeToolsets).not.toContain('alpha');
    expect(adapter.toolsetCount).toBeGreaterThanOrEqual(1); // core survives
  });

  it('getUseToolsetSchema() returns a tool with description', () => {
    const schema = adapter.getUseToolsetSchema();
    expect(schema.description).toBeDefined();
    expect(schema.parameters).toBeDefined();
  });

  it('processToolCalls handles non-array input gracefully', () => {
    // @ts-expect-error — intentionally passing wrong type for robustness check
    expect(() => adapter.processToolCalls(null)).not.toThrow();
  });

  it('activeToolsets excludes core from the list', () => {
    // activeToolsets is the non-core toolsets
    adapter.useToolset('add', 'alpha');
    adapter.filterTools({});
    expect(adapter.activeToolsets).toContain('alpha');
    expect(adapter.activeToolsets).not.toContain('core');
  });

  it('toolsetCount includes core', () => {
    expect(adapter.toolsetCount).toBe(1); // only core initially
    adapter.useToolset('add', 'alpha');
    adapter.filterTools({});
    expect(adapter.toolsetCount).toBe(2); // core + alpha
  });

  it('filterTools is idempotent on the same turn', () => {
    adapter.useToolset('add', 'alpha');
    const tools = { alpha_tool: mockTool(), beta_tool: mockTool() };

    const r1 = adapter.filterTools({ tools });
    // No pending change after first call — second call should give same result
    const r2 = adapter.filterTools({ tools });

    expect(toolNames(r1.tools!)).toEqual(toolNames(r2.tools!));
  });
});

// ---------------------------------------------------------------------------
// 9. Consistency — public surface mirrors Python adapters
// ---------------------------------------------------------------------------

describe('VercelAIAdapter — API surface consistency', () => {
  it('exposes filterTools() (mirrors Python _filter_tools)', () => {
    const adapter = new VercelAIAdapter();
    expect(typeof adapter.filterTools).toBe('function');
  });

  it('exposes processToolCalls() (mirrors Python _process_tool_calls)', () => {
    const adapter = new VercelAIAdapter();
    expect(typeof adapter.processToolCalls).toBe('function');
  });

  it('exposes useToolset() (mirrors Python BaseAdapter.use_toolset)', () => {
    const adapter = new VercelAIAdapter();
    expect(typeof adapter.useToolset).toBe('function');
  });

  it('exposes getUseToolsetSchema() (mirrors Python get_use_toolset_schema)', () => {
    const adapter = new VercelAIAdapter();
    expect(typeof adapter.getUseToolsetSchema).toBe('function');
  });

  it('exposes wrapGenerateText() (mirrors Python wrap_client)', () => {
    const adapter = new VercelAIAdapter();
    expect(typeof adapter.wrapGenerateText).toBe('function');
  });

  it('exposes wrapStreamText()', () => {
    const adapter = new VercelAIAdapter();
    expect(typeof adapter.wrapStreamText).toBe('function');
  });

  it('exposes activeToolsets property (mirrors Python get_active_toolsets)', () => {
    const adapter = new VercelAIAdapter();
    expect(Array.isArray(adapter.activeToolsets)).toBe(true);
  });

  it('exposes pendingToolsets property', () => {
    const adapter = new VercelAIAdapter();
    expect(adapter.pendingToolsets).toBeNull();
  });

  it('exposes toolsetCount (mirrors Python get_toolset_count)', () => {
    const adapter = new VercelAIAdapter();
    expect(typeof adapter.toolsetCount).toBe('number');
    expect(adapter.toolsetCount).toBeGreaterThanOrEqual(1);
  });

  it('USE_TOOLSET_TOOL is exported (mirrors Python _USE_TOOLSET_SCHEMA)', () => {
    // In Vercel AI SDK, the tool name is the record key — not a field on the tool
    // itself. The description describes what the tool does, not its name.
    expect(USE_TOOLSET_TOOL).toBeDefined();
    expect(USE_TOOLSET_TOOL.description).toBeTruthy();
    expect(USE_TOOLSET_TOOL.description).toContain('load');   // "Dynamically load…"
    expect(USE_TOOLSET_TOOL.parameters).toBeDefined();
  });

  it('useToolset returns result with status field (deferred or ok)', () => {
    const adapter = new VercelAIAdapter();
    adapter.registerToolset('browser', ['navigate']);

    const add   = adapter.useToolset('add', 'browser');
    const reset = adapter.useToolset('reset');

    expect(add.status).toBe('deferred');
    expect(reset.status).toBe('ok');
  });
});
