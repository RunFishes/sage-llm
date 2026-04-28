/**
 * Typed errors for sage-llm.
 *
 * `retryable` drives the `withRetry` wrapper in retry.ts — network/5xx/rate_limit
 * can retry, auth/context_length/content_filter/aborted cannot.
 */

export const LLMErrorType = {
  /** fetch() threw — DNS, offline, connection reset, etc. Retryable. */
  NETWORK: "network",
  /** 401 / 403. Not retryable. */
  AUTH: "auth",
  /** 429. Retryable. */
  RATE_LIMIT: "rate_limit",
  /** 5xx. Retryable. */
  SERVER: "server",
  /** 400-ish: prompt too long. Not retryable. */
  CONTEXT_LENGTH: "context_length",
  /** Safety system blocked the response. Not retryable. */
  CONTENT_FILTER: "content_filter",
  /** Response body didn't match the shape we expected. Retryable (model hiccup). */
  INVALID_RESPONSE: "invalid_response",
  /** AbortSignal fired. Not retryable. */
  ABORTED: "aborted",
  /** Catch-all. Retryable by default. */
  UNKNOWN: "unknown",
} as const;

export type LLMErrorType = (typeof LLMErrorType)[keyof typeof LLMErrorType];

const RETRYABLE_TYPES: ReadonlySet<LLMErrorType> = new Set([
  LLMErrorType.NETWORK,
  LLMErrorType.RATE_LIMIT,
  LLMErrorType.SERVER,
  LLMErrorType.INVALID_RESPONSE,
  LLMErrorType.UNKNOWN,
]);

/**
 * Error codes from OpenAI / Anthropic that signal "prompt too long".
 * Add aliases here as we encounter them in real responses.
 */
const CONTEXT_LENGTH_CODES: ReadonlySet<string> = new Set([
  "context_length_exceeded", // OpenAI
  "string_above_max_length", // OpenAI (rare)
  "tokens_exceeded_error", // some compatible providers
]);

/**
 * Error codes that signal "content filtered by safety system".
 */
const CONTENT_FILTER_CODES: ReadonlySet<string> = new Set([
  "content_filter", // OpenAI
  "content_policy_violation", // OpenAI
  "safety", // generic
]);

/** Fallback regex for messages that look like context-length errors. */
const CONTEXT_LENGTH_PATTERN =
  /context.{0,10}length|too long|maximum.{0,10}token|exceeds.{0,20}token/i;

export class LLMError extends Error {
  readonly type: LLMErrorType;
  readonly retryable: boolean;
  readonly statusCode?: number;
  /** Original error (e.g. the TypeError from a failed fetch). */
  readonly cause?: unknown;
  /** Parsed response body, if we got one. Useful for debugging. */
  readonly raw?: unknown;

  constructor(
    type: LLMErrorType,
    message: string,
    options?: { statusCode?: number; cause?: unknown; raw?: unknown }
  ) {
    super(message);
    this.name = "LLMError";
    this.type = type;
    this.statusCode = options?.statusCode;
    this.cause = options?.cause;
    this.raw = options?.raw;
    this.retryable = RETRYABLE_TYPES.has(type);
  }

  /**
   * Unified HTTP error classifier — used by all providers.
   *
   * Provider's job is just to extract `{ status, errorCode?, message }` from
   * its own error body shape. This function maps that to an `LLMErrorType`.
   *
   * Classification:
   * 1. Status code (coarse): 401/403 → AUTH, 429 → RATE_LIMIT, 5xx → SERVER
   * 2. errorCode (fine, for 4xx): see code sets below
   * 3. Message regex (fallback): matches "context length"-ish phrases
   */
  static fromHTTPError(
    status: number,
    errorCode: string | undefined,
    message: string,
    raw?: unknown
  ): LLMError {
    // Coarse classification by status code.
    if (status === 401 || status === 403) {
      return new LLMError(LLMErrorType.AUTH, message, {
        statusCode: status,
        raw,
      });
    }
    if (status === 429) {
      return new LLMError(LLMErrorType.RATE_LIMIT, message, {
        statusCode: status,
        raw,
      });
    }
    if (status >= 500) {
      return new LLMError(LLMErrorType.SERVER, message, {
        statusCode: status,
        raw,
      });
    }

    // Fine classification by error code (covers OpenAI `code` and Anthropic `type`).
    const code = errorCode?.toLowerCase();
    if (code && CONTEXT_LENGTH_CODES.has(code)) {
      return new LLMError(LLMErrorType.CONTEXT_LENGTH, message, {
        statusCode: status,
        raw,
      });
    }
    if (code && CONTENT_FILTER_CODES.has(code)) {
      return new LLMError(LLMErrorType.CONTENT_FILTER, message, {
        statusCode: status,
        raw,
      });
    }

    // Fallback: pattern-match the message for context-length signals.
    if (CONTEXT_LENGTH_PATTERN.test(message)) {
      return new LLMError(LLMErrorType.CONTEXT_LENGTH, message, {
        statusCode: status,
        raw,
      });
    }

    return new LLMError(LLMErrorType.UNKNOWN, message, {
      statusCode: status,
      raw,
    });
  }

  /** Wrap a thrown fetch error. */
  static fromFetchError(err: unknown): LLMError {
    const isAbort =
      err instanceof Error &&
      (err.name === "AbortError" || err.message.includes("aborted"));
    if (isAbort) {
      return new LLMError(LLMErrorType.ABORTED, "Request aborted", { cause: err });
    }
    return new LLMError(LLMErrorType.NETWORK, "Network request failed", {
      cause: err,
    });
  }
}
