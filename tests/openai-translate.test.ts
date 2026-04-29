/**
 * Tests for src/providers/openai-translate.ts.
 *
 * Covers the pure-function translation layer between sage-llm's unified
 * Message/Tool/StreamChunk types and OpenAI's wire format:
 *   - messagesToOpenAI / messageToOpenAI (system / user / multi-part / assistant / tool)
 *   - toolToOpenAI / toolsToOpenAI (Zod -> JSON Schema)
 *   - toolChoiceToOpenAI
 *   - parseOpenAIResponse + parseUsage + mapFinishReason
 *   - OpenAIStreamAccumulator (text, tool calls, usage, finish reason, edge cases)
 *   - extractOpenAIError
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod/v4";

import {
  messagesToOpenAI,
  toolToOpenAI,
  toolsToOpenAI,
  toolChoiceToOpenAI,
  parseOpenAIResponse,
  parseUsage,
  mapFinishReason,
  OpenAIStreamAccumulator,
  extractOpenAIError,
} from "../src/providers/openai-translate.ts";
import type { Message, StreamChunk } from "../src/types.ts";

// ---------------------------------------------------------------------------
// messagesToOpenAI
// ---------------------------------------------------------------------------

test("messagesToOpenAI: system message passes through", () => {
  const out = messagesToOpenAI([{ role: "system", content: "you are nice" }]);
  assert.deepEqual(out, [{ role: "system", content: "you are nice" }]);
});

test("messagesToOpenAI: user with string content passes through", () => {
  const out = messagesToOpenAI([{ role: "user", content: "hi" }]);
  assert.deepEqual(out, [{ role: "user", content: "hi" }]);
});

test("messagesToOpenAI: user multi-part content (text + image) becomes image_url nested struct", () => {
  const msgs: Message[] = [
    {
      role: "user",
      content: [
        { type: "text", text: "describe" },
        { type: "image", source: "https://example.com/cat.png" },
      ],
    },
  ];
  const out = messagesToOpenAI(msgs);
  assert.deepEqual(out, [
    {
      role: "user",
      content: [
        { type: "text", text: "describe" },
        {
          type: "image_url",
          image_url: { url: "https://example.com/cat.png" },
        },
      ],
    },
  ]);
});

test("messagesToOpenAI: assistant with content only and no toolCalls -> no tool_calls field", () => {
  const out = messagesToOpenAI([
    { role: "assistant", content: "sure thing" },
  ]);
  assert.deepEqual(out, [{ role: "assistant", content: "sure thing" }]);
  // ensure tool_calls is not even present (not just undefined).
  assert.equal(Object.prototype.hasOwnProperty.call(out[0], "tool_calls"), false);
});

test("messagesToOpenAI: assistant with toolCalls -> camelCase converted, argumentsRaw preserved verbatim", () => {
  const out = messagesToOpenAI([
    {
      role: "assistant",
      content: null,
      toolCalls: [
        {
          id: "call_1",
          name: "search",
          arguments: { q: "cats" },
          // intentionally non-canonical JSON spacing to confirm we use the raw string.
          argumentsRaw: '{"q":   "cats"}',
        },
      ],
    },
  ]);
  assert.deepEqual(out, [
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "search", arguments: '{"q":   "cats"}' },
        },
      ],
    },
  ]);
});

test("messagesToOpenAI: tool result message converts toolCallId -> tool_call_id", () => {
  const out = messagesToOpenAI([
    { role: "tool", toolCallId: "call_xyz", content: "42" },
  ]);
  assert.deepEqual(out, [
    { role: "tool", tool_call_id: "call_xyz", content: "42" },
  ]);
});

// ---------------------------------------------------------------------------
// toolToOpenAI / toolsToOpenAI
// ---------------------------------------------------------------------------

test("toolToOpenAI: simple zod object schema produces correct JSON Schema", () => {
  const out = toolToOpenAI({
    name: "search",
    description: "Search the web",
    inputSchema: z.object({ q: z.string() }),
  });
  assert.equal(out.type, "function");
  assert.equal(out.function.name, "search");
  assert.equal(out.function.description, "Search the web");

  const params = out.function.parameters as {
    type: string;
    properties: { q: { type: string } };
    required: string[];
  };
  assert.equal(params.type, "object");
  assert.equal(params.properties.q.type, "string");
  assert.deepEqual(params.required, ["q"]);
});

test("toolToOpenAI: description is undefined when not provided", () => {
  const out = toolToOpenAI({
    name: "noop",
    inputSchema: z.object({}),
  });
  assert.equal(out.function.description, undefined);
});

test("toolsToOpenAI: maps an array", () => {
  const out = toolsToOpenAI([
    { name: "a", inputSchema: z.object({ x: z.number() }) },
    { name: "b", inputSchema: z.object({ y: z.string() }) },
  ]);
  assert.equal(out.length, 2);
  assert.equal(out[0]!.function.name, "a");
  assert.equal(out[1]!.function.name, "b");
});

// ---------------------------------------------------------------------------
// toolChoiceToOpenAI
// ---------------------------------------------------------------------------

test("toolChoiceToOpenAI: 'auto' passes through as string", () => {
  assert.equal(toolChoiceToOpenAI("auto"), "auto");
});

test("toolChoiceToOpenAI: 'required' passes through as string", () => {
  assert.equal(toolChoiceToOpenAI("required"), "required");
});

test("toolChoiceToOpenAI: 'none' passes through as string", () => {
  assert.equal(toolChoiceToOpenAI("none"), "none");
});

test("toolChoiceToOpenAI: { name } becomes nested function descriptor", () => {
  assert.deepEqual(toolChoiceToOpenAI({ name: "search" }), {
    type: "function",
    function: { name: "search" },
  });
});

test("toolChoiceToOpenAI: undefined -> undefined", () => {
  assert.equal(toolChoiceToOpenAI(undefined), undefined);
});

// ---------------------------------------------------------------------------
// parseOpenAIResponse
// ---------------------------------------------------------------------------

test("parseOpenAIResponse: pure text response", () => {
  const r = parseOpenAIResponse({
    choices: [{ message: { content: "hello" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  });
  assert.equal(r.content, "hello");
  assert.deepEqual(r.toolCalls, []);
  assert.equal(r.finishReason, "stop");
  assert.deepEqual(r.usage, {
    promptTokens: 5,
    completionTokens: 3,
    totalTokens: 8,
  });
});

test("parseOpenAIResponse: pure tool-call response (content null)", () => {
  const r = parseOpenAIResponse({
    choices: [
      {
        message: {
          content: null,
          tool_calls: [
            {
              id: "call_1",
              function: { name: "search", arguments: '{"q":"cats"}' },
            },
          ],
        },
        finish_reason: "tool_calls",
      },
    ],
  });
  assert.equal(r.content, null);
  assert.equal(r.toolCalls.length, 1);
  assert.equal(r.toolCalls[0]!.id, "call_1");
  assert.equal(r.toolCalls[0]!.name, "search");
  assert.deepEqual(r.toolCalls[0]!.arguments, { q: "cats" });
  assert.equal(r.toolCalls[0]!.argumentsRaw, '{"q":"cats"}');
  assert.equal(r.finishReason, "tool_calls");
});

test("parseOpenAIResponse: mixed content + tool_calls", () => {
  const r = parseOpenAIResponse({
    choices: [
      {
        message: {
          content: "let me check",
          tool_calls: [
            { id: "c1", function: { name: "fetch", arguments: "{}" } },
          ],
        },
        finish_reason: "tool_calls",
      },
    ],
  });
  assert.equal(r.content, "let me check");
  assert.equal(r.toolCalls.length, 1);
  assert.deepEqual(r.toolCalls[0]!.arguments, {});
});

test("parseOpenAIResponse: malformed JSON arguments degrade to {} but argumentsRaw kept verbatim", () => {
  const broken = '{"q": "cat'; // truncated
  const r = parseOpenAIResponse({
    choices: [
      {
        message: {
          content: null,
          tool_calls: [
            { id: "c1", function: { name: "search", arguments: broken } },
          ],
        },
        finish_reason: "tool_calls",
      },
    ],
  });
  assert.deepEqual(r.toolCalls[0]!.arguments, {});
  assert.equal(r.toolCalls[0]!.argumentsRaw, broken);
});

test("parseOpenAIResponse: missing choices[0] throws", () => {
  assert.throws(() => parseOpenAIResponse({}), /missing choices\[0\]/);
  assert.throws(() => parseOpenAIResponse({ choices: [] }), /missing choices\[0\]/);
});

test("parseOpenAIResponse: full usage with cached + reasoning details", () => {
  const r = parseOpenAIResponse({
    choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    usage: {
      prompt_tokens: 100,
      completion_tokens: 50,
      total_tokens: 150,
      prompt_tokens_details: { cached_tokens: 30 },
      completion_tokens_details: { reasoning_tokens: 20 },
    },
  });
  assert.deepEqual(r.usage, {
    promptTokens: 100,
    completionTokens: 50,
    totalTokens: 150,
    cachedTokens: 30,
    reasoningTokens: 20,
  });
});

test("parseOpenAIResponse: missing usage -> all zero", () => {
  const r = parseOpenAIResponse({
    choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
  });
  assert.deepEqual(r.usage, {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  });
});

test("parseOpenAIResponse: keeps raw response for debugging", () => {
  const data = {
    choices: [{ message: { content: "hi" }, finish_reason: "stop" }],
  };
  const r = parseOpenAIResponse(data);
  assert.equal(r.raw, data);
});

// ---------------------------------------------------------------------------
// mapFinishReason
// ---------------------------------------------------------------------------

test("mapFinishReason: 'stop' -> 'stop'", () => {
  assert.equal(mapFinishReason("stop"), "stop");
});

test("mapFinishReason: 'tool_calls' -> 'tool_calls'", () => {
  assert.equal(mapFinishReason("tool_calls"), "tool_calls");
});

test("mapFinishReason: legacy 'function_call' -> 'tool_calls'", () => {
  assert.equal(mapFinishReason("function_call"), "tool_calls");
});

test("mapFinishReason: 'length' -> 'length'", () => {
  assert.equal(mapFinishReason("length"), "length");
});

test("mapFinishReason: 'content_filter' -> 'content_filter'", () => {
  assert.equal(mapFinishReason("content_filter"), "content_filter");
});

test("mapFinishReason: unknown / undefined / null -> 'unknown'", () => {
  assert.equal(mapFinishReason("weird"), "unknown");
  assert.equal(mapFinishReason(undefined), "unknown");
  assert.equal(mapFinishReason(null), "unknown");
});

// ---------------------------------------------------------------------------
// parseUsage (direct)
// ---------------------------------------------------------------------------

test("parseUsage: undefined -> zeroes, no optional fields", () => {
  const u = parseUsage(undefined);
  assert.deepEqual(u, {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  });
});

test("parseUsage: only base fields, no details", () => {
  const u = parseUsage({
    prompt_tokens: 10,
    completion_tokens: 20,
    total_tokens: 30,
  });
  assert.deepEqual(u, {
    promptTokens: 10,
    completionTokens: 20,
    totalTokens: 30,
  });
  // optional fields not set when absent
  assert.equal(Object.prototype.hasOwnProperty.call(u, "cachedTokens"), false);
  assert.equal(Object.prototype.hasOwnProperty.call(u, "reasoningTokens"), false);
});

test("parseUsage: cached_tokens=0 still surfaces (since !== undefined)", () => {
  const u = parseUsage({
    prompt_tokens: 1,
    completion_tokens: 1,
    total_tokens: 2,
    prompt_tokens_details: { cached_tokens: 0 },
  });
  assert.equal(u.cachedTokens, 0);
});

// ---------------------------------------------------------------------------
// OpenAIStreamAccumulator
// ---------------------------------------------------------------------------

test("OpenAIStreamAccumulator: pure text stream yields multiple text-delta chunks then done", () => {
  const acc = new OpenAIStreamAccumulator();
  const out: StreamChunk[] = [];
  out.push(...acc.process({ choices: [{ delta: { content: "hel" } }] }));
  out.push(...acc.process({ choices: [{ delta: { content: "lo " } }] }));
  out.push(...acc.process({ choices: [{ delta: { content: "world" } }] }));
  out.push(
    ...acc.process({ choices: [{ delta: {}, finish_reason: "stop" }] })
  );
  out.push(...acc.finish());
  assert.deepEqual(out, [
    { type: "text-delta", text: "hel" },
    { type: "text-delta", text: "lo " },
    { type: "text-delta", text: "world" },
    {
      type: "done",
      finishReason: "stop",
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    },
  ]);
});

test("OpenAIStreamAccumulator: single tool-call stream emits start, deltas, end, done", () => {
  const acc = new OpenAIStreamAccumulator();
  const out: StreamChunk[] = [];
  // chunk 1: id+name only
  out.push(
    ...acc.process({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: "call_1", function: { name: "search", arguments: "" } },
            ],
          },
        },
      ],
    })
  );
  // chunk 2: arguments fragment
  out.push(
    ...acc.process({
      choices: [
        {
          delta: { tool_calls: [{ index: 0, function: { arguments: '{"q":' } }] },
        },
      ],
    })
  );
  // chunk 3: more args
  out.push(
    ...acc.process({
      choices: [
        {
          delta: { tool_calls: [{ index: 0, function: { arguments: '"cats"}' } }] },
        },
      ],
    })
  );
  // chunk 4: finish_reason
  out.push(
    ...acc.process({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })
  );
  out.push(...acc.finish());
  assert.deepEqual(out, [
    { type: "tool-call-start", index: 0, id: "call_1", name: "search" },
    { type: "tool-call-delta", index: 0, argumentsDelta: '{"q":' },
    { type: "tool-call-delta", index: 0, argumentsDelta: '"cats"}' },
    { type: "tool-call-end", index: 0 },
    {
      type: "done",
      finishReason: "tool_calls",
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    },
  ]);
});

test("OpenAIStreamAccumulator: multiple tool calls (interleaved indices) get independent start/delta/end", () => {
  const acc = new OpenAIStreamAccumulator();
  const out: StreamChunk[] = [];
  out.push(
    ...acc.process({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: "c0", function: { name: "a", arguments: "" } },
            ],
          },
        },
      ],
    })
  );
  out.push(
    ...acc.process({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 1, id: "c1", function: { name: "b", arguments: "" } },
            ],
          },
        },
      ],
    })
  );
  out.push(
    ...acc.process({
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, function: { arguments: "{}" } }],
          },
        },
      ],
    })
  );
  out.push(
    ...acc.process({
      choices: [
        {
          delta: {
            tool_calls: [{ index: 1, function: { arguments: "{}" } }],
          },
        },
      ],
    })
  );
  out.push(
    ...acc.process({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })
  );
  out.push(...acc.finish());

  assert.deepEqual(out, [
    { type: "tool-call-start", index: 0, id: "c0", name: "a" },
    { type: "tool-call-start", index: 1, id: "c1", name: "b" },
    { type: "tool-call-delta", index: 0, argumentsDelta: "{}" },
    { type: "tool-call-delta", index: 1, argumentsDelta: "{}" },
    { type: "tool-call-end", index: 0 },
    { type: "tool-call-end", index: 1 },
    {
      type: "done",
      finishReason: "tool_calls",
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    },
  ]);
});

test("OpenAIStreamAccumulator: trailing chunk with usage populates done.usage", () => {
  const acc = new OpenAIStreamAccumulator();
  acc.process({ choices: [{ delta: { content: "ok" } }] });
  acc.process({ choices: [{ delta: {}, finish_reason: "stop" }] });
  // openai sometimes sends a final chunk with empty choices and a usage object
  acc.process({
    choices: [],
    usage: {
      prompt_tokens: 7,
      completion_tokens: 2,
      total_tokens: 9,
      prompt_tokens_details: { cached_tokens: 4 },
    },
  });
  const tail = acc.finish();
  assert.deepEqual(tail, [
    {
      type: "done",
      finishReason: "stop",
      usage: {
        promptTokens: 7,
        completionTokens: 2,
        totalTokens: 9,
        cachedTokens: 4,
      },
    },
  ]);
});

test("OpenAIStreamAccumulator: finish_reason from mid-chunk reflected in done", () => {
  const acc = new OpenAIStreamAccumulator();
  acc.process({ choices: [{ delta: { content: "x" }, finish_reason: "length" }] });
  const tail = acc.finish();
  assert.equal(tail.length, 1);
  assert.equal(tail[0]!.type, "done");
  if (tail[0]!.type === "done") {
    assert.equal(tail[0]!.finishReason, "length");
  }
});

test("OpenAIStreamAccumulator: empty content '' and arguments '' do NOT yield phantom deltas", () => {
  const acc = new OpenAIStreamAccumulator();
  // empty content
  const out1 = acc.process({ choices: [{ delta: { content: "" } }] });
  assert.deepEqual(out1, []);
  // tool-call chunk with empty arguments string -> still emits start (first sight) but NO delta
  const out2 = acc.process({
    choices: [
      {
        delta: {
          tool_calls: [
            { index: 0, id: "c1", function: { name: "n", arguments: "" } },
          ],
        },
      },
    ],
  });
  assert.deepEqual(out2, [
    { type: "tool-call-start", index: 0, id: "c1", name: "n" },
  ]);
  // subsequent chunk for same index with empty args -> nothing
  const out3 = acc.process({
    choices: [
      { delta: { tool_calls: [{ index: 0, function: { arguments: "" } }] } },
    ],
  });
  assert.deepEqual(out3, []);
});

test("OpenAIStreamAccumulator: text-only stream finish() yields just done (no tool-call-end)", () => {
  const acc = new OpenAIStreamAccumulator();
  acc.process({ choices: [{ delta: { content: "hi" } }] });
  acc.process({ choices: [{ delta: {}, finish_reason: "stop" }] });
  const tail = acc.finish();
  assert.equal(tail.length, 1);
  assert.equal(tail[0]!.type, "done");
});

test("OpenAIStreamAccumulator: chunk with no choices is a no-op", () => {
  const acc = new OpenAIStreamAccumulator();
  const out = acc.process({});
  assert.deepEqual(out, []);
});

// ---------------------------------------------------------------------------
// extractOpenAIError
// ---------------------------------------------------------------------------

test("extractOpenAIError: standard body with code+message", () => {
  const r = extractOpenAIError(
    {
      error: {
        code: "invalid_api_key",
        type: "authentication_error",
        message: "bad key",
      },
    },
    "fallback"
  );
  assert.equal(r.code, "invalid_api_key");
  assert.equal(r.message, "bad key");
});

test("extractOpenAIError: only type, no code -> uses type", () => {
  const r = extractOpenAIError(
    { error: { type: "rate_limit_error", message: "slow down" } },
    "fallback"
  );
  assert.equal(r.code, "rate_limit_error");
  assert.equal(r.message, "slow down");
});

test("extractOpenAIError: missing error field -> code undefined, message=fallback", () => {
  const r = extractOpenAIError({}, "HTTP 500");
  assert.equal(r.code, undefined);
  assert.equal(r.message, "HTTP 500");
});

test("extractOpenAIError: undefined body -> fallback", () => {
  const r = extractOpenAIError(undefined, "boom");
  assert.equal(r.code, undefined);
  assert.equal(r.message, "boom");
});

// ---------------------------------------------------------------------------
// Reviewer-found gaps (round 2)
// ---------------------------------------------------------------------------

test("messagesToOpenAI: assistant with empty toolCalls array -> tool_calls field omitted", () => {
  const out = messagesToOpenAI([
    { role: "assistant", content: "hi", toolCalls: [] },
  ]);
  assert.deepEqual(out, [{ role: "assistant", content: "hi" }]);
  assert.equal(
    Object.prototype.hasOwnProperty.call(out[0], "tool_calls"),
    false
  );
});

test("messagesToOpenAI: assistant with both content and toolCalls", () => {
  const out = messagesToOpenAI([
    {
      role: "assistant",
      content: "let me search",
      toolCalls: [
        {
          id: "call_1",
          name: "search",
          arguments: { q: "x" },
          argumentsRaw: '{"q":"x"}',
        },
      ],
    },
  ]);
  assert.deepEqual(out, [
    {
      role: "assistant",
      content: "let me search",
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "search", arguments: '{"q":"x"}' },
        },
      ],
    },
  ]);
});

test("OpenAIStreamAccumulator: usage in chunk with empty choices is recorded into done", () => {
  const acc = new OpenAIStreamAccumulator();
  // Real-world final OpenAI frame: choices=[] but usage filled.
  const out = acc.process({
    choices: [],
    usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
  });
  assert.deepEqual(out, []); // doesn't emit on the spot
  const tail = acc.finish();
  assert.equal(tail.length, 1);
  assert.equal(tail[0]!.type, "done");
  if (tail[0]!.type === "done") {
    assert.deepEqual(tail[0]!.usage, {
      promptTokens: 7,
      completionTokens: 3,
      totalTokens: 10,
    });
  }
});

test("OpenAIStreamAccumulator: delta.content === null is ignored (not emitted)", () => {
  const acc = new OpenAIStreamAccumulator();
  // Real OpenAI streams send `content: null` on the role-only opening frame.
  const out = acc.process({ choices: [{ delta: { content: null } }] });
  assert.deepEqual(out, []);
});

test("OpenAIStreamAccumulator: raw.usage === null is ignored", () => {
  const acc = new OpenAIStreamAccumulator();
  // Mid-stream OpenAI frames have `usage: null` until the final frame.
  acc.process({
    choices: [{ delta: { content: "x" } }],
    usage: null,
  });
  const tail = acc.finish();
  assert.equal(tail[0]!.type, "done");
  if (tail[0]!.type === "done") {
    // Untouched defaults — usage was never set.
    assert.deepEqual(tail[0]!.usage, {
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    });
  }
});

test("OpenAIStreamAccumulator: tool_calls item missing function field still emits start", () => {
  const acc = new OpenAIStreamAccumulator();
  // Defensive: malformed chunk has no `function` field.
  const out = acc.process({
    choices: [{ delta: { tool_calls: [{ index: 0, id: "call_x" }] } }],
  });
  assert.deepEqual(out, [
    { type: "tool-call-start", index: 0, id: "call_x", name: "" },
  ]);
});

test("parseOpenAIResponse: tool_calls item with missing function fields", () => {
  const r = parseOpenAIResponse({
    choices: [
      {
        message: {
          content: null,
          tool_calls: [{ id: "call_x" /* no function */ }],
        },
        finish_reason: "tool_calls",
      },
    ],
  });
  assert.equal(r.toolCalls.length, 1);
  assert.equal(r.toolCalls[0]!.id, "call_x");
  assert.equal(r.toolCalls[0]!.name, "");
  assert.deepEqual(r.toolCalls[0]!.arguments, {});
  assert.equal(r.toolCalls[0]!.argumentsRaw, "");
});

test("parseOpenAIResponse: empty tool_calls array -> toolCalls is empty array", () => {
  const r = parseOpenAIResponse({
    choices: [
      { message: { content: "ok", tool_calls: [] }, finish_reason: "stop" },
    ],
  });
  assert.deepEqual(r.toolCalls, []);
});

test("parseUsage: reasoning_tokens=0 boundary is preserved", () => {
  const u = parseUsage({
    prompt_tokens: 1,
    completion_tokens: 1,
    total_tokens: 2,
    completion_tokens_details: { reasoning_tokens: 0 },
  });
  assert.equal(u.reasoningTokens, 0);
});

test("mapFinishReason: empty string -> 'unknown'", () => {
  assert.equal(mapFinishReason(""), "unknown");
});

test("extractOpenAIError: empty string code falls back to type", () => {
  // Bug fix: `??` would have kept "", `||` lets type take over.
  const r = extractOpenAIError(
    { error: { code: "", type: "rate_limit_error", message: "..." } },
    "fallback"
  );
  assert.equal(r.code, "rate_limit_error");
});
