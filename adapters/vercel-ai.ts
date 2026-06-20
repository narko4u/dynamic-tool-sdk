/**
 * Dynamic Tool SDK — Vercel AI SDK adapter.
 *
 * Wraps `generateText()` and `streamText()` from the `ai` package so tool
 * schemas are auto-filtered on every call.  The `use_toolset` bootstrap tool
 * is injected into every request so the model can dynamically load/unload
 * tool groups between turns.
 *
 * Architecture mirrors the Python adapters (openai.py / anthropic.py) but is
 * idiomatic TypeScript.  Key difference: Vercel AI tools are keyed by name
 * in a `Record<string, Tool>` object, not an array, so filtering removes keys
 * rather than array items.
 *
 * @example
 * ```typescript
 * import { generateText } from 'ai';
 * import { openai } from '@ai-sdk/openai';
 * import { VercelAIAdapter } from './adapters/vercel-ai';
 *
 * const adapter = new VercelAIAdapter();
 * adapter.registerToolset('browser', ['navigate', 'click', 'type']);
 *
 * const wrappedGenerate = adapter.wrapGenerateText(generateText);
 * const result = await wrappedGenerate({
 *   model: openai('gpt-4o'),
 *   tools: { navigate: ..., click: ..., type: ... },
 *   messages: [...],
 * });
 * ```
 */

import { z } from 'zod';

// ---------------------------------------------------------------------------
// Local types (subset of ai package types; avoids hard coupling to version)
// ---------------------------------------------------------------------------

/** A single tool definition as passed to generateText/streamText. */
export interface VercelTool {
  description?: string;
  parameters: z.ZodTypeAny | Record<string, unknown>;
  execute?: (...args: unknown[]) => Promise<unknown>;
}

/** The tools record passed to generateText/streamText. */
export type VercelToolSet = Record<string, VercelTool>;

/** A single resolved tool call in the generateText response. */
export interface VercelToolCall {
  type?: string;
  toolCallId?: string;
  toolName: string;
  args: Record<string, unknown>;
}

/** Minimal shape of the generateText result we need to inspect. */
export interface GenerateTextResult {
  text: string;
  toolCalls?: VercelToolCall[] | Promise<VercelToolCall[]>;
  [key: string]: unknown;
}

/** Minimal shape of the streamText result (tool calls are async). */
export interface StreamTextResult {
  toolCalls?: Promise<VercelToolCall[]>;
  [key: string]: unknown;
}

/** Parameters accepted by generateText / streamText. */
export interface GenerateParams {
  tools?: VercelToolSet;
  [key: string]: unknown;
}

/** Wrapped function type — same signature, auto-filtered. */
export type GenerateTextFn = (params: GenerateParams) => Promise<GenerateTextResult>;
export type StreamTextFn   = (params: GenerateParams) => Promise<StreamTextResult>;

// ---------------------------------------------------------------------------
// UseToolset result type (mirrors Python SchemaFilter.use_toolset())
// ---------------------------------------------------------------------------

export interface UseToolsetResult {
  action: string;
  status?: 'ok' | 'deferred';
  active_toolsets?: string[];
  requested_toolsets?: string[];
  message?: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// Internal state management (TypeScript port of state.py + filter.py)
// ---------------------------------------------------------------------------

/**
 * Tracks active and pending toolsets.
 * Thread-safety is not a concern in Node.js (single-threaded event loop).
 */
class ToolsetState {
  private _active: Set<string>;
  private _pending: string[] | null = null;
  private readonly _core: string;

  constructor(coreToolset = 'core') {
    this._core = coreToolset;
    this._active = new Set([coreToolset]);
  }

  get active(): Set<string> {
    return new Set(this._active);
  }

  get activeList(): string[] {
    return [...this._active].sort();
  }

  get pending(): string[] | null {
    return this._pending ? [...this._pending] : null;
  }

  hasPending(): boolean {
    return this._pending !== null;
  }

  applyPending(): Set<string> | null {
    if (this._pending === null) return null;
    this._active = new Set(this._pending);
    this._pending = null;
    return new Set(this._active);
  }

  reset(): void {
    this._active = new Set([this._core]);
    this._pending = null;
  }

  setToolsets(names: string[]): string[] {
    const next = new Set([...names, this._core]);
    this._pending = [...next].sort();
    return [...this._pending];
  }

  addToolsets(names: string[]): string[] {
    const current = new Set([...this._active, ...(this._pending ?? [])]);
    const next    = new Set([...current, ...names, this._core]);
    this._pending = [...next].sort();
    return [...this._pending];
  }

  removeToolsets(names: string[]): string[] {
    const current  = new Set([...this._active, ...(this._pending ?? [])]);
    const toRemove = new Set(names);
    const next     = new Set([...current].filter(n => !toRemove.has(n)));
    next.add(this._core);
    this._pending = [...next].sort();
    return [...this._pending];
  }
}

// ---------------------------------------------------------------------------
// ToolsetRegistry — maps toolset name → allowed tool names
// ---------------------------------------------------------------------------

export interface ToolsetEntry {
  description: string;
  tools: string[];  // tool names belonging to this toolset
}

// ---------------------------------------------------------------------------
// Bootstrap use_toolset tool
// ---------------------------------------------------------------------------

const USE_TOOLSET_DESCRIPTION =
  'Dynamically load or unload tool groups for the next turn. ' +
  'Actions: set (replace all), add (add to current), ' +
  'remove (remove from current), reset (back to core), ' +
  'list (show available). ' +
  "Examples: 'add browser', 'set terminal,web', 'reset'.";

const useToolsetParameters = z.object({
  action: z.enum([
    'set', 'load', 'replace',
    'add', 'enable', 'include',
    'remove', 'disable', 'drop',
    'reset', 'default',
    'list',
  ]).describe('The toolset management action to perform.'),
  toolsets: z
    .string()
    .optional()
    .describe("Comma-separated toolset names. Omit for 'reset' and 'list'."),
});

/** The use_toolset bootstrap tool injected into every request. */
export const USE_TOOLSET_TOOL: VercelTool = {
  description: USE_TOOLSET_DESCRIPTION,
  parameters: useToolsetParameters,
};

// ---------------------------------------------------------------------------
// VercelAIAdapter
// ---------------------------------------------------------------------------

export interface VercelAIAdapterOptions {
  /** Name of the always-present toolset (default: 'core'). */
  coreToolset?: string;
  /** If true, inject use_toolset into every request automatically (default: true). */
  autoInject?: boolean;
}

/**
 * Adapter that wraps Vercel AI SDK's `generateText` / `streamText` to
 * auto-filter tool schemas and intercept `use_toolset` tool calls.
 */
export class VercelAIAdapter {
  private readonly _state:    ToolsetState;
  private readonly _registry: Map<string, ToolsetEntry>;
  private readonly _autoInject: boolean;

  constructor(options: VercelAIAdapterOptions = {}) {
    this._state      = new ToolsetState(options.coreToolset ?? 'core');
    this._registry   = new Map();
    this._autoInject = options.autoInject ?? true;
  }

  // ------------------------------------------------------------------
  // Registry
  // ------------------------------------------------------------------

  /**
   * Register a toolset so the adapter knows which tool names belong to it.
   *
   * @param name       Toolset name (e.g. 'browser', 'email')
   * @param toolNames  Array of tool names in this toolset
   * @param description Human-readable description (optional)
   */
  registerToolset(name: string, toolNames: string[], description = ''): void {
    this._registry.set(name, { description, tools: toolNames });
  }

  /**
   * List all registered toolsets with their descriptions.
   */
  listRegisteredToolsets(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [name, entry] of this._registry) {
      out[name] = entry.description;
    }
    return out;
  }

  // ------------------------------------------------------------------
  // Schema helpers
  // ------------------------------------------------------------------

  /**
   * Return the use_toolset bootstrap tool definition.
   * This is always injected into every request when autoInject is true.
   */
  getUseToolsetSchema(): VercelTool {
    return { ...USE_TOOLSET_TOOL };
  }

  // ------------------------------------------------------------------
  // State accessors (mirrors Python BaseAdapter API)
  // ------------------------------------------------------------------

  get activeToolsets(): string[] {
    return this._state.activeList.filter(s => s !== 'core');
  }

  get pendingToolsets(): string[] | null {
    return this._state.pending;
  }

  get toolsetCount(): number {
    return this._state.activeList.length;
  }

  // ------------------------------------------------------------------
  // Toolset management (mirrors Python BaseAdapter.use_toolset())
  // ------------------------------------------------------------------

  /**
   * Queue a toolset change for the next API round-trip.
   * Changes are deferred — they take effect on the next `filterTools()` call.
   *
   * @param action   'set' | 'add' | 'remove' | 'reset' | 'list' (and aliases)
   * @param toolsets Comma-separated toolset names (omit for reset/list)
   */
  useToolset(action: string, toolsets = ''): UseToolsetResult {
    const a = action.trim().toLowerCase();

    if (a === 'reset' || a === 'default') {
      this._state.reset();
      return {
        action: 'reset',
        status: 'ok',
        active_toolsets: this._state.activeList,
        message: 'Reset to core tools. Change takes effect on the next turn.',
      };
    }

    if (a === 'list') {
      return {
        action: 'list',
        active_toolsets: this._state.activeList,
        message: `Active: ${this._state.activeList.join(', ')}. Registered: ${[...this._registry.keys()].join(', ')}`,
      };
    }

    const names = toolsets
      .split(',')
      .map(n => n.trim())
      .filter(Boolean);

    if (['set', 'load', 'replace'].includes(a)) {
      const valid = names.filter(n => this._registry.has(n));
      if (!valid.length) {
        return { action, error: `No valid toolsets in: "${toolsets}". Registered: ${[...this._registry.keys()].join(', ')}` };
      }
      const pending = this._state.setToolsets(valid);
      return { action, status: 'deferred', requested_toolsets: pending, message: `Toolset change queued: ${pending.join(', ')}` };
    }

    if (['add', 'enable', 'include'].includes(a)) {
      const valid = names.filter(n => this._registry.has(n));
      if (!valid.length) {
        return { action, error: `No valid toolsets: "${toolsets}". Registered: ${[...this._registry.keys()].join(', ')}` };
      }
      const pending = this._state.addToolsets(valid);
      return { action, status: 'deferred', requested_toolsets: pending, message: `Toolset change queued: ${pending.join(', ')}` };
    }

    if (['remove', 'disable', 'drop'].includes(a)) {
      const pending = this._state.removeToolsets(names);
      return { action, status: 'deferred', requested_toolsets: pending, message: `Toolset change queued: ${pending.join(', ')}` };
    }

    return { action, error: `Unknown action '${action}'. Use: set, add, remove, reset, list.` };
  }

  // ------------------------------------------------------------------
  // Core filtering (mirrors Python _filter_and_process)
  // ------------------------------------------------------------------

  /**
   * Filter the `tools` key in a params object by active toolsets, apply any
   * pending changes, and inject the use_toolset bootstrap tool.
   *
   * This is the public entry point used directly by tests and by the wrapped
   * functions — mirrors `_filter_tools()` in the Python adapters.
   */
  filterTools(params: GenerateParams): GenerateParams {
    // Apply any pending toolset change at the turn boundary
    this._state.applyPending();

    const { tools, ...rest } = params;

    if (!tools || Object.keys(tools).length === 0) {
      if (this._autoInject) {
        return { ...rest, tools: { use_toolset: USE_TOOLSET_TOOL } };
      }
      return params;
    }

    const filtered = this._filterToolSet(tools);

    if (this._autoInject && !('use_toolset' in filtered)) {
      filtered['use_toolset'] = USE_TOOLSET_TOOL;
    }

    return { ...rest, tools: filtered };
  }

  /**
   * Filter a VercelToolSet record to only include tools from active toolsets.
   * Falls through (keeps all) if no tool names are registered for the active set.
   */
  private _filterToolSet(tools: VercelToolSet): VercelToolSet {
    const active  = this._state.active;
    const allowed = this._buildAllowedSet(active);

    if (allowed.size === 0) {
      // Nothing registered — pass all tools through (same as Python behaviour)
      return { ...tools };
    }

    return Object.fromEntries(
      Object.entries(tools).filter(([name]) => allowed.has(name))
    );
  }

  /** Build the set of allowed tool names from currently active toolsets. */
  private _buildAllowedSet(active: Set<string>): Set<string> {
    const allowed = new Set<string>();
    for (const tsName of active) {
      const entry = this._registry.get(tsName);
      if (entry) {
        entry.tools.forEach(t => allowed.add(t));
      }
    }
    return allowed;
  }

  // ------------------------------------------------------------------
  // Response processing (mirrors Python _process_tool_calls)
  // ------------------------------------------------------------------

  /**
   * Inspect an array of tool calls for `use_toolset` invocations.
   * Applies the requested action immediately (same as Python adapters).
   */
  processToolCalls(toolCalls: VercelToolCall[]): void {
    if (!Array.isArray(toolCalls)) return;
    for (const tc of toolCalls) {
      try {
        if (tc.toolName === 'use_toolset') {
          const action   = (tc.args['action']   as string | undefined) ?? 'list';
          const toolsets = (tc.args['toolsets'] as string | undefined) ?? '';
          this.useToolset(action, toolsets);
        }
      } catch {
        // Malformed tool call — silently ignored, same as Python adapters
      }
    }
  }

  // ------------------------------------------------------------------
  // Function wrapping
  // ------------------------------------------------------------------

  /**
   * Wrap `generateText` from the `ai` package.
   *
   * The returned function has the same signature.  On every call it:
   *  1. Applies any pending toolset change
   *  2. Filters the tools record
   *  3. Calls the original `generateText`
   *  4. Processes tool calls in the response
   */
  wrapGenerateText(fn: GenerateTextFn): GenerateTextFn {
    return async (params: GenerateParams): Promise<GenerateTextResult> => {
      const filtered = this.filterTools(params);
      const result   = await fn(filtered);

      // Resolve tool calls — in v4 they're synchronous on the result object
      const toolCalls = await Promise.resolve(result.toolCalls ?? []);
      this.processToolCalls(toolCalls as VercelToolCall[]);

      return result;
    };
  }

  /**
   * Wrap `streamText` from the `ai` package.
   *
   * Tool calls are available as a promise on the stream result; we await them
   * after returning so the caller gets the stream immediately but the adapter
   * still processes tool calls when they resolve.
   */
  wrapStreamText(fn: StreamTextFn): StreamTextFn {
    return async (params: GenerateParams): Promise<StreamTextResult> => {
      const filtered = this.filterTools(params);
      const result   = await fn(filtered);

      // Process tool calls asynchronously as the stream resolves
      const toolCallsPromise = result.toolCalls ?? Promise.resolve([]);
      Promise.resolve(toolCallsPromise).then((tcs) => {
        this.processToolCalls((tcs as VercelToolCall[]) ?? []);
      }).catch(() => {/* ignore stream errors in tool-call processing */});

      return result;
    };
  }
}
