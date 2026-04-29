/**
 * Tests for parseSSE.
 *
 * We build `Response` objects around hand-crafted `ReadableStream`s so we can
 * control exactly how bytes are chunked — including nasty cases like UTF-8
 * characters split across chunks.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseSSE, type SSEEvent } from "../src/sse.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a Response whose body yields the given byte chunks, in order. */
function responseFromChunks(chunks: (string | Uint8Array)[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) {
        controller.enqueue(typeof c === "string" ? encoder.encode(c) : c);
      }
      controller.close();
    },
  });
  return new Response(stream);
}

async function collect(response: Response): Promise<SSEEvent[]> {
  const events: SSEEvent[] = [];
  for await (const ev of parseSSE(response)) {
    events.push(ev);
  }
  return events;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

test("single event with single data line", async () => {
  const res = responseFromChunks(["data: hello\n\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "hello" }]);
});

test("multiple events separated by blank line", async () => {
  const res = responseFromChunks([
    "data: one\n\ndata: two\n\ndata: three\n\n",
  ]);
  const events = await collect(res);
  assert.deepEqual(events, [
    { data: "one" },
    { data: "two" },
    { data: "three" },
  ]);
});

test("multi-line data joined with \\n", async () => {
  const res = responseFromChunks(["data: line1\ndata: line2\ndata: line3\n\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "line1\nline2\nline3" }]);
});

test("event field (Anthropic style)", async () => {
  const res = responseFromChunks([
    "event: message_start\ndata: {}\n\nevent: message_stop\ndata: {}\n\n",
  ]);
  const events = await collect(res);
  assert.deepEqual(events, [
    { event: "message_start", data: "{}" },
    { event: "message_stop", data: "{}" },
  ]);
});

test("[DONE] sentinel passes through as plain data", async () => {
  const res = responseFromChunks([
    'data: {"text":"hi"}\n\ndata: [DONE]\n\n',
  ]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: '{"text":"hi"}' }, { data: "[DONE]" }]);
});

test("comment lines are ignored", async () => {
  const res = responseFromChunks([": heartbeat\ndata: real\n\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "real" }]);
});

test("chunk split mid-line", async () => {
  // The \n\n boundary straddles chunks; data field split mid-word.
  const res = responseFromChunks(["data: hel", 'lo wor', "ld\n", "\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "hello world" }]);
});

test("chunk split mid-UTF-8-character (critical)", async () => {
  // "你好" in UTF-8 = E4 BD A0 E5 A5 BD (6 bytes).
  // Split between the 2nd and 3rd byte of "你".
  const fullMsg = "data: 你好\n\n";
  const bytes = new TextEncoder().encode(fullMsg);
  // Find the position right after "data: " (6 bytes) + first 2 bytes of "你".
  // "data: " is 6 ASCII bytes, so split at index 8.
  const first = bytes.slice(0, 8);
  const second = bytes.slice(8);
  const res = responseFromChunks([first, second]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "你好" }]);
});

test("CRLF line endings", async () => {
  const res = responseFromChunks(["data: a\r\ndata: b\r\n\r\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "a\nb" }]);
});

test("single-space strip on field value", async () => {
  // "data:hello" (no space) and "data:  hello" (two spaces, only strip one).
  const res = responseFromChunks(["data:hello\n\ndata:  hello\n\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "hello" }, { data: " hello" }]);
});

test("id and retry fields", async () => {
  const res = responseFromChunks(["id: 42\nretry: 3000\ndata: x\n\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ id: "42", retry: 3000, data: "x" }]);
});

test("unknown field is silently ignored", async () => {
  const res = responseFromChunks(["unknown: foo\ndata: real\n\n"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "real" }]);
});

test("trailing data without blank line is dropped", async () => {
  // Strict per spec: don't yield partial events.
  const res = responseFromChunks(["data: complete\n\ndata: dangling"]);
  const events = await collect(res);
  assert.deepEqual(events, [{ data: "complete" }]);
});

test("empty body closes cleanly", async () => {
  const res = responseFromChunks([]);
  const events = await collect(res);
  assert.deepEqual(events, []);
});

test("abort signal stops iteration", async () => {
  const controller = new AbortController();
  // Build a stream that never completes.
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(new TextEncoder().encode("data: one\n\n"));
      // Never close.
    },
  });
  const res = new Response(stream);
  // Abort after first event.
  const events: SSEEvent[] = [];
  let threw = false;
  try {
    for await (const ev of parseSSE(res, controller.signal)) {
      events.push(ev);
      controller.abort();
    }
  } catch (err: unknown) {
    threw = true;
    assert.equal((err as Error).name, "LLMError");
    // Must be classified as ABORTED (not NETWORK/UNKNOWN), so retry layer skips it.
    assert.equal((err as { type: string }).type, "aborted");
  }
  assert.equal(threw, true);
  assert.equal(events.length, 1);
});
