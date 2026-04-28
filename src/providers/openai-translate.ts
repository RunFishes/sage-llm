/**
 * Translation layer between sage-llm's unified types and OpenAI's wire format.
 *
 * Pure functions only — no fetch, no IO. Easy to unit-test.
 *
 * Sections:
 * 1. Outgoing: Message / ToolDefinition / ToolChoice → request body
 * 2. Incoming (non-streaming): JSON response → InvokeResult
 * 3. Incoming (streaming): chunk-by-chunk state machine → StreamChunk[]
 * 4. Error: error body → (status, code, message) for LLMError.fromHTTPError
 */

import { z } from "zod/v4";

import type {
  AssistantMessage,
  FinishReason,
  InvokeOptions,
  InvokeResult,
  LLMUsage,
  Message,
  StreamChunk,
  ToolCall,
  ToolDefinition,
} from "../types.ts";

// ---------------------------------------------------------------------------
// 1. Outgoing
// ---------------------------------------------------------------------------

/** OpenAI message shape (what we send). */
type OpenAIMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string | OpenAIUserPart[] }
  | {
      role: "assistant";
      content: string | null;
      tool_calls?: {
        id: string;
        type: "function";
        function: { name: string; arguments: string };
      }[];
    }
  | { role: "tool"; tool_call_id: string; content: string };

type OpenAIUserPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

/** Convert a list of unified messages to OpenAI wire format. */
export function messagesToOpenAI(messages: Message[]): OpenAIMessage[] {
  return messages.map(messageToOpenAI);
}

function messageToOpenAI(m: Message): OpenAIMessage {
  switch (m.role) {
    case "system":
      return { role: "system", content: m.content };
    case "user": {
      if (typeof m.content === "string") {
        return { role: "user", content: m.content };
      }
      const parts: OpenAIUserPart[] = m.content.map((p) =>
        p.type === "text"
          ? { type: "text", text: p.text }
          : { type: "image_url", image_url: { url: p.source } }
      );
      return { role: "user", content: parts };
    }
    case "assistant":
      return assistantToOpenAI(m);
    case "tool":
      return {
        role: "tool",
        tool_call_id: m.toolCallId,
        content: m.content,
      };
  }
}

function assistantToOpenAI(
  m: AssistantMessage
): Extract<OpenAIMessage, { role: "assistant" }> {
  const out: Extract<OpenAIMessage, { role: "assistant" }> = {
    role: "assistant",
    content: m.content,
  };
  if (m.toolCalls && m.toolCalls.length > 0) {
    out.tool_calls = m.toolCalls.map((tc) => ({
      id: tc.id,
      type: "function" as const,
      function: { name: tc.name, arguments: tc.argumentsRaw },
    }));
  }
  return out;
}

/** Convert a tool definition to OpenAI tool format (JSON Schema via Zod). */
export function toolToOpenAI(tool: ToolDefinition) {
  return {
    type: "function" as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: z.toJSONSchema(tool.inputSchema, { target: "draft-7" }),
    },
  };
}

export function toolsToOpenAI(tools: ToolDefinition[]) {
  return tools.map(toolToOpenAI);
}

/** Translate our `toolChoice` to OpenAI's. */
export function toolChoiceToOpenAI(
  choice: InvokeOptions["toolChoice"]
): string | { type: "function"; function: { name: string } } | undefined {
  if (choice === undefined) return undefined;
  if (typeof choice === "string") return choice; // "auto" | "required" | "none"
  return { type: "function", function: { name: choice.name } };
}

// ---------------------------------------------------------------------------
// 2. Incoming (non-streaming)
// ---------------------------------------------------------------------------

/** Parse a non-streaming OpenAI chat.completion response. */
export function parseOpenAIResponse(data: unknown): InvokeResult {
  const d = data as {
    choices?: Array<{
      message?: {
        content?: string | null;
        tool_calls?: Array<{
          id: string;
          function?: { name: string; arguments: string };
        }>;
      };
      finish_reason?: string;
    }>;
    usage?: OpenAIUsage;
  };

  const choice = d.choices?.[0];
  if (!choice) {
    throw new Error("OpenAI response missing choices[0]");
  }
  const msg = choice.message ?? {};

  const toolCalls: ToolCall[] = (msg.tool_calls ?? []).map((tc) => ({
    id: tc.id,
    name: tc.function?.name ?? "",
    arguments: safeJsonParse(tc.function?.arguments ?? ""),
    argumentsRaw: tc.function?.arguments ?? "",
  }));

  return {
    content: msg.content ?? null,
    toolCalls,
    finishReason: mapFinishReason(choice.finish_reason),
    usage: parseUsage(d.usage),
    raw: data,
  };
}

function safeJsonParse(s: string): unknown {
  if (!s) return {};
  try {
    return JSON.parse(s);
  } catch {
    return {};
  }
}

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

export function parseUsage(u: OpenAIUsage | undefined): LLMUsage {
  return {
    promptTokens: u?.prompt_tokens ?? 0,
    completionTokens: u?.completion_tokens ?? 0,
    totalTokens: u?.total_tokens ?? 0,
    ...(u?.prompt_tokens_details?.cached_tokens !== undefined && {
      cachedTokens: u.prompt_tokens_details.cached_tokens,
    }),
    ...(u?.completion_tokens_details?.reasoning_tokens !== undefined && {
      reasoningTokens: u.completion_tokens_details.reasoning_tokens,
    }),
  };
}

export function mapFinishReason(r: string | undefined | null): FinishReason {
  switch (r) {
    case "stop":
      return "stop";
    case "tool_calls":
    case "function_call": // legacy
      return "tool_calls";
    case "length":
      return "length";
    case "content_filter":
      return "content_filter";
    default:
      return "unknown";
  }
}

// ---------------------------------------------------------------------------
// 3. Incoming (streaming)
// ---------------------------------------------------------------------------

interface OpenAIStreamChunk {
  choices?: Array<{
    delta?: {
      content?: string | null;
      tool_calls?: Array<{
        index: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: OpenAIUsage | null;
}

/**
 * Stateful accumulator for OpenAI streaming responses.
 *
 * Usage:
 *   const acc = new OpenAIStreamAccumulator();
 *   for each parsed JSON chunk:
 *     yield* acc.process(chunk);
 *   yield* acc.finish();
 */
export class OpenAIStreamAccumulator {
  private seenToolIndices = new Set<number>();
  private finishReason: FinishReason = "unknown";
  private usage: LLMUsage = {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  };

  /** Process one streaming chunk; returns chunks to yield. */
  process(raw: OpenAIStreamChunk): StreamChunk[] {
    const out: StreamChunk[] = [];
    const choice = raw.choices?.[0];

    const delta = choice?.delta;
    if (delta) {
      if (delta.content) {
        out.push({ type: "text-delta", text: delta.content });
      }
      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index;
          if (!this.seenToolIndices.has(idx)) {
            this.seenToolIndices.add(idx);
            out.push({
              type: "tool-call-start",
              index: idx,
              id: tc.id ?? "",
              name: tc.function?.name ?? "",
            });
          }
          if (tc.function?.arguments) {
            out.push({
              type: "tool-call-delta",
              index: idx,
              argumentsDelta: tc.function.arguments,
            });
          }
        }
      }
    }

    if (choice?.finish_reason) {
      this.finishReason = mapFinishReason(choice.finish_reason);
    }
    if (raw.usage) {
      this.usage = parseUsage(raw.usage);
    }

    return out;
  }

  /** Called once stream ends — emits tool-call-end per index, then done. */
  finish(): StreamChunk[] {
    const out: StreamChunk[] = [];
    // Sort indices for determinism in tests.
    const indices = [...this.seenToolIndices].sort((a, b) => a - b);
    for (const idx of indices) {
      out.push({ type: "tool-call-end", index: idx });
    }
    out.push({
      type: "done",
      finishReason: this.finishReason,
      usage: this.usage,
    });
    return out;
  }
}

// ---------------------------------------------------------------------------
// 4. Error extraction
// ---------------------------------------------------------------------------

/**
 * Extract `{ code, message }` from an OpenAI error response body.
 * Falls back to status text when body is not in the expected shape.
 */
export function extractOpenAIError(
  body: unknown,
  fallbackMessage: string
): { code: string | undefined; message: string } {
  const b = body as
    | { error?: { code?: string; type?: string; message?: string } }
    | undefined;
  return {
    code: b?.error?.code ?? b?.error?.type,
    message: b?.error?.message ?? fallbackMessage,
  };
}
