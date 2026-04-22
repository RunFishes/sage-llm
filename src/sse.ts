/**
 * SSE (Server-Sent Events) parser.
 *
 * Design:
 * - Input: a `Response` with an event-stream body.
 * - Output: async iterator of `SSEEvent` — `{ event?, data, id?, retry? }`.
 * - Only parses the SSE wire protocol. Does NOT interpret `data: [DONE]` or
 *   parse JSON — those are the caller's job.
 *
 * Handles:
 * - Chunks split mid-line (TCP doesn't respect line boundaries).
 * - Chunks split mid-UTF-8-character (via `TextDecoder({ stream: true })`).
 * - `\n` / `\r\n` / `\r` line endings (spec requires all three).
 * - Multi-line `data:` fields (joined with `\n` per spec).
 * - Comment lines starting with `:` (skipped).
 * - `event:` / `data:` / `id:` / `retry:` fields.
 *
 * Reference: https://html.spec.whatwg.org/multipage/server-sent-events.html
 */

import { LLMError, LLMErrorType } from "./errors.ts";

export interface SSEEvent {
  /** The `event:` field, if present. OpenAI omits, Anthropic uses it. */
  event?: string;
  /** Concatenated `data:` field(s). Raw string — caller decides how to parse. */
  data: string;
  /** The `id:` field, if present. */
  id?: string;
  /** The `retry:` field (ms), if present and numeric. */
  retry?: number;
}

/**
 * Parse a streaming `Response` body as SSE events.
 *
 * Yields one event per `\n\n`-terminated block. Trailing content that never
 * got a terminator is dropped (strict per spec — don't yield half-events).
 */
export async function* parseSSE(
  response: Response,
  signal?: AbortSignal
): AsyncGenerator<SSEEvent, void, void> {
  if (!response.body) {
    throw new LLMError(LLMErrorType.INVALID_RESPONSE, "Response has no body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        throw new LLMError(LLMErrorType.ABORTED, "SSE stream aborted");
      }

      const { value, done } = await reader.read();
      if (done) {
        // Flush any final bytes still buffered in the decoder.
        buffer += decoder.decode();
        // If the last event lacked a terminating blank line, drop it.
        // (HTML spec: only dispatch on blank-line terminator.)
        return;
      }

      buffer += decoder.decode(value, { stream: true });

      // Pull out every complete event currently in the buffer.
      while (true) {
        const boundary = findEventBoundary(buffer);
        if (boundary === -1) break;
        const rawEvent = buffer.slice(0, boundary.end);
        buffer = buffer.slice(boundary.end + boundary.skip);
        const parsed = parseEvent(rawEvent);
        if (parsed) yield parsed;
      }
    }
  } finally {
    // Always release the lock so the caller can cancel / re-read if needed.
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

interface Boundary {
  /** Offset where the event body ends (exclusive). */
  end: number;
  /** Number of characters to skip after `end` (the blank-line separator). */
  skip: number;
}

/**
 * Find the end of the first complete event in `buf`.
 * Events end at a blank line: `\n\n`, `\r\n\r\n`, or `\r\r`.
 * Returns `-1` if no complete event yet, or a `Boundary` giving the slice offsets.
 */
function findEventBoundary(buf: string): -1 | Boundary {
  // Check all three separators; take the earliest one found.
  const candidates: Array<{ idx: number; skip: number }> = [];

  const lfLf = buf.indexOf("\n\n");
  if (lfLf !== -1) candidates.push({ idx: lfLf, skip: 2 });

  const crlfCrlf = buf.indexOf("\r\n\r\n");
  if (crlfCrlf !== -1) candidates.push({ idx: crlfCrlf, skip: 4 });

  const crCr = buf.indexOf("\r\r");
  if (crCr !== -1) candidates.push({ idx: crCr, skip: 2 });

  if (candidates.length === 0) return -1;

  // Pick the earliest boundary.
  candidates.sort((a, b) => a.idx - b.idx);
  const first = candidates[0]!;
  return { end: first.idx, skip: first.skip };
}

/**
 * Parse a single event block (everything between two blank lines).
 * Returns `null` if the block is empty or contains no dispatchable fields.
 */
function parseEvent(raw: string): SSEEvent | null {
  if (raw.length === 0) return null;

  // Split into lines. Spec accepts \r\n, \n, or \r as line terminator within
  // an event block. A single regex split covers all three.
  const lines = raw.split(/\r\n|\n|\r/);

  let eventName: string | undefined;
  let dataParts: string[] = [];
  let id: string | undefined;
  let retry: number | undefined;
  let hasField = false;

  for (const line of lines) {
    if (line.length === 0) continue;
    // Comment line.
    if (line.startsWith(":")) continue;

    // Split at first colon. If no colon, whole line is field name with empty value.
    const colon = line.indexOf(":");
    let field: string;
    let value: string;
    if (colon === -1) {
      field = line;
      value = "";
    } else {
      field = line.slice(0, colon);
      value = line.slice(colon + 1);
      // Spec: if value begins with a single space, strip it.
      if (value.startsWith(" ")) value = value.slice(1);
    }

    switch (field) {
      case "event":
        eventName = value;
        hasField = true;
        break;
      case "data":
        dataParts.push(value);
        hasField = true;
        break;
      case "id":
        // Spec: ignore id fields containing NUL.
        if (!value.includes("\0")) id = value;
        hasField = true;
        break;
      case "retry": {
        const n = Number(value);
        if (Number.isInteger(n) && n >= 0) retry = n;
        hasField = true;
        break;
      }
      default:
        // Unknown field — ignore per spec.
        break;
    }
  }

  if (!hasField) return null;

  // Per spec, data lines are joined with \n (no trailing \n).
  const data = dataParts.join("\n");

  // If there's no data but also no event name, skip (spec says don't dispatch
  // a message event with empty data *unless* id/retry set something).
  // But for our purposes, yield any event that had at least one field —
  // providers can decide what to do.
  return {
    ...(eventName !== undefined && { event: eventName }),
    data,
    ...(id !== undefined && { id }),
    ...(retry !== undefined && { retry }),
  };
}
