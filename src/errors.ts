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

  /** Classify an HTTP response status code into an LLMErrorType. */
  static fromStatus(
    status: number,
    message: string,
    raw?: unknown
  ): LLMError {
    let type: LLMErrorType;
    if (status === 401 || status === 403) type = LLMErrorType.AUTH;
    else if (status === 429) type = LLMErrorType.RATE_LIMIT;
    else if (status >= 500) type = LLMErrorType.SERVER;
    else if (status === 400 && /context|length|token/i.test(message))
      type = LLMErrorType.CONTEXT_LENGTH;
    else type = LLMErrorType.UNKNOWN;
    return new LLMError(type, message, { statusCode: status, raw });
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
