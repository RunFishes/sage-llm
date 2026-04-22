// sage-llm — public API.
// Provider implementations are added in later steps.

export type {
  Message,
  SystemMessage,
  UserMessage,
  UserContentPart,
  AssistantMessage,
  ToolResultMessage,
  ToolDefinition,
  ToolCall,
  LLMUsage,
  FinishReason,
  InvokeOptions,
  InvokeResult,
  StreamChunk,
} from "./types.ts";

export { LLMError, LLMErrorType } from "./errors.ts";

export type { LLMProvider, ProviderConfig } from "./providers/index.ts";

export { parseSSE } from "./sse.ts";
export type { SSEEvent } from "./sse.ts";

export const VERSION = "0.0.0";
