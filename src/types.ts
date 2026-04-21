/**
 * Core types for sage-llm.
 *
 * Design principles:
 * - Provider-agnostic: users write these types, providers translate to/from
 *   OpenAI / Anthropic wire formats.
 * - No runtime deps: Zod is a peer dep, only imported as type.
 * - Tool execution is the caller's job. We parse tool_calls out of the
 *   response; we never invoke handlers ourselves.
 */

import type { ZodType } from "zod";

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/**
 * Unified message type. Each provider translates this to its own wire format:
 *
 * - OpenAI: `{ role, content, tool_calls?, tool_call_id? }`
 * - Anthropic: content blocks (`text` / `tool_use` / `tool_result`)
 */
export type Message =
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | ToolResultMessage;

export interface SystemMessage {
  role: "system";
  content: string;
}

export interface UserMessage {
  role: "user";
  /** Plain string, or multi-part content (future: images). */
  content: string | UserContentPart[];
}

export type UserContentPart =
  | { type: "text"; text: string }
  | { type: "image"; source: string /* data URL or https URL */ };

export interface AssistantMessage {
  role: "assistant";
  /** null when the response is pure tool_calls with no text. */
  content: string | null;
  /** Present when the model wants to invoke tools. */
  toolCalls?: ToolCall[];
}

export interface ToolResultMessage {
  role: "tool";
  /** Links back to the `ToolCall.id` this message is responding to. */
  toolCallId: string;
  /** Stringified result. Objects should be JSON.stringify'd by caller. */
  content: string;
  /** Optional flag so providers can render errors as `is_error: true` (Anthropic). */
  isError?: boolean;
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/**
 * Tool definition. `inputSchema` is a Zod schema — we convert it to JSON
 * Schema via `z.toJSONSchema()` (Zod 4) at invoke time.
 *
 * Note: no `execute` here. The library returns parsed tool_calls; the caller
 * dispatches to their own handler (e.g. across the sage-extension bridge).
 */
export interface ToolDefinition<TInput = unknown> {
  name: string;
  description?: string;
  inputSchema: ZodType<TInput>;
}

/** A tool invocation parsed from an assistant response. */
export interface ToolCall {
  /** Provider-assigned id. Echoed back in the matching `ToolResultMessage`. */
  id: string;
  name: string;
  /** Parsed JSON args. `unknown` because caller should validate with Zod. */
  arguments: unknown;
  /** Raw JSON string, kept for debugging / logging. */
  argumentsRaw: string;
}

// ---------------------------------------------------------------------------
// Usage / finish reason
// ---------------------------------------------------------------------------

export interface LLMUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  /** OpenAI `prompt_tokens_details.cached_tokens`. */
  cachedTokens?: number;
  /** OpenAI o1-series `completion_tokens_details.reasoning_tokens`. */
  reasoningTokens?: number;
  /** Anthropic `cache_creation_input_tokens`. */
  cacheCreationTokens?: number;
  /** Anthropic `cache_read_input_tokens`. */
  cacheReadTokens?: number;
}

export type FinishReason =
  | "stop"
  | "tool_calls"
  | "length"
  | "content_filter"
  | "unknown";

// ---------------------------------------------------------------------------
// Invoke (non-streaming)
// ---------------------------------------------------------------------------

export interface InvokeOptions {
  messages: Message[];
  /** Zero or more tools. If present, the model may call them. */
  tools?: ToolDefinition[];
  /**
   * Tool-choice policy:
   * - `auto` (default): model decides whether to call a tool
   * - `required`: must call some tool
   * - `none`: disallow tool calls
   * - `{ name }`: force a specific tool
   */
  toolChoice?: "auto" | "required" | "none" | { name: string };
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
}

export interface InvokeResult {
  content: string | null;
  toolCalls: ToolCall[];
  finishReason: FinishReason;
  usage: LLMUsage;
  /** Raw provider response for debugging. */
  raw?: unknown;
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/**
 * Streaming chunks. Consumers do `for await (const chunk of provider.stream(...))`.
 *
 * Tool calls stream as: `tool-call-start` → N × `tool-call-delta` → `tool-call-end`.
 * Text streams as: N × `text-delta`.
 * The final chunk is always `done` (or `error` if something broke mid-stream).
 */
export type StreamChunk =
  | { type: "text-delta"; text: string }
  | { type: "tool-call-start"; index: number; id: string; name: string }
  | { type: "tool-call-delta"; index: number; argumentsDelta: string }
  | { type: "tool-call-end"; index: number }
  | { type: "done"; finishReason: FinishReason; usage: LLMUsage }
  | { type: "error"; error: Error };
