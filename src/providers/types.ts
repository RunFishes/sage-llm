/**
 * LLMProvider — the abstraction every concrete provider implements.
 *
 * Two providers planned:
 * - OpenAIProvider: OpenAI / DeepSeek / Qwen / local vllm (any OpenAI-compat endpoint)
 * - AnthropicProvider: native /v1/messages with prompt caching + thinking blocks
 *
 * A provider is a translator: public `Message` / `ToolDefinition` in → wire
 * format out, wire response in → `InvokeResult` / `StreamChunk` out.
 * Retry and agent-loop logic live *above* this interface.
 */

import type {
  InvokeOptions,
  InvokeResult,
  StreamChunk,
} from "../types.ts";

export interface LLMProvider {
  /** Human-readable tag, e.g. `"openai"` / `"anthropic"`. For logging. */
  readonly name: string;

  /** Resolved model id, e.g. `"gpt-4o"` / `"claude-sonnet-4-5"`. */
  readonly model: string;

  /**
   * Non-streaming request. Returns once the full response is available.
   * Single round-trip — no retry, no agent loop. Caller wraps with `withRetry`.
   */
  invoke(options: InvokeOptions): Promise<InvokeResult>;

  /**
   * Streaming request. Yields chunks as they arrive.
   * Final chunk is always `{ type: "done" }` on success or `{ type: "error" }` on failure.
   */
  stream(options: InvokeOptions): AsyncGenerator<StreamChunk, void, void>;
}

/**
 * Base config shared by all providers. Each concrete provider extends this
 * with its own fields (e.g. `apiVersion` for Anthropic).
 */
export interface ProviderConfig {
  /** API endpoint base, e.g. `"https://api.openai.com/v1"`. */
  baseURL: string;
  /** Model id sent in request body. */
  model: string;
  /** Sent as `Authorization: Bearer <key>` (OpenAI) or `x-api-key` (Anthropic). */
  apiKey?: string;

  /** Default temperature; can be overridden per invoke. */
  temperature?: number;
  /** Default max output tokens; can be overridden per invoke. */
  maxTokens?: number;

  /**
   * Custom fetch. Use to inject headers, proxies, or a polyfill.
   * Defaults to `globalThis.fetch` bound to globalThis.
   */
  fetch?: typeof globalThis.fetch;

  /** Extra headers sent with every request (merged with auth headers). */
  headers?: Record<string, string>;
}
