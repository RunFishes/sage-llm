/**
 * Tests for OpenAIProvider.
 *
 * Each test stubs `fetch` via the `fetch` config option. We assert on the
 * outgoing request (URL/method/headers/body) and on how the provider classifies
 * the response into `InvokeResult` / `StreamChunk[]` / `LLMError`.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod/v4";

import { OpenAIProvider } from "../src/providers/openai.ts";
import { LLMError, LLMErrorType } from "../src/errors.ts";
import type { InvokeOptions, StreamChunk } from "../src/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface MockCall {
  url: string;
  init: RequestInit;
}

function makeMockFetch(
  handler: (url: string, init: RequestInit) => Response | Promise<Response>
) {
  const calls: MockCall[] = [];
  const fn = async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return handler(url, init);
  };
  return { fn: fn as unknown as typeof globalThis.fetch, calls };
}

/** Build a Response whose body yields the given byte chunks, in order. */
function responseFromChunks(
  chunks: (string | Uint8Array)[],
  init?: ResponseInit
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) {
        controller.enqueue(typeof c === "string" ? encoder.encode(c) : c);
      }
      controller.close();
    },
  });
  return new Response(stream, init);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function getRequestBody(call: MockCall): Record<string, unknown> {
  return JSON.parse(call.init.body as string) as Record<string, unknown>;
}

const baseMessages: InvokeOptions["messages"] = [
  { role: "user", content: "hi" },
];

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

test("constructor: throws when baseURL missing", () => {
  assert.throws(
    () =>
      new OpenAIProvider({
        baseURL: "",
        model: "gpt-4o",
      }),
    /baseURL is required/
  );
});

test("constructor: throws when model missing", () => {
  assert.throws(
    () =>
      new OpenAIProvider({
        baseURL: "https://api.openai.com/v1",
        model: "",
      }),
    /model is required/
  );
});

test("constructor: strips trailing slash from baseURL", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://api.example.com/v1/",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await p.invoke({ messages: baseMessages });
  assert.equal(mock.calls[0].url, "https://api.example.com/v1/chat/completions");
});

test("constructor: custom chatPath used instead of default", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://api.example.com",
    model: "gpt-4o",
    chatPath: "/custom/path",
    fetch: mock.fn,
  });
  await p.invoke({ messages: baseMessages });
  assert.equal(mock.calls[0].url, "https://api.example.com/custom/path");
});

test("constructor: name is openai and model is passed through", () => {
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "my-model",
  });
  assert.equal(p.name, "openai");
  assert.equal(p.model, "my-model");
});

// ---------------------------------------------------------------------------
// invoke — success path
// ---------------------------------------------------------------------------

test("invoke: sends POST to baseURL+chatPath with correct headers", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "hello" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    apiKey: "sk-abc",
    fetch: mock.fn,
  });
  await p.invoke({ messages: baseMessages });

  const call = mock.calls[0];
  assert.equal(call.url, "https://x/chat/completions");
  assert.equal(call.init.method, "POST");
  const headers = call.init.headers as Record<string, string>;
  assert.equal(headers["Content-Type"], "application/json");
  assert.equal(headers["Authorization"], "Bearer sk-abc");
});

test("invoke: custom headers merged in; organization adds OpenAI-Organization", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    apiKey: "sk",
    organization: "org-1",
    headers: { "X-Custom": "yes" },
    fetch: mock.fn,
  });
  await p.invoke({ messages: baseMessages });
  const headers = mock.calls[0].init.headers as Record<string, string>;
  assert.equal(headers["X-Custom"], "yes");
  assert.equal(headers["OpenAI-Organization"], "org-1");
});

test("invoke: body has model + messages, stream is false (non-streaming)", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await p.invoke({ messages: baseMessages });
  const body = getRequestBody(mock.calls[0]);
  assert.equal(body.model, "gpt-4o");
  assert.deepEqual(body.messages, [{ role: "user", content: "hi" }]);
  assert.equal(body.stream, false);
  assert.equal("stream_options" in body, false);
});

test("invoke: tools + parallel_tool_calls:false + tool_choice in body", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await p.invoke({
    messages: baseMessages,
    tools: [
      {
        name: "sum",
        description: "add two numbers",
        inputSchema: z.object({ a: z.number(), b: z.number() }),
      },
    ],
    toolChoice: "auto",
  });
  const body = getRequestBody(mock.calls[0]);
  assert.ok(Array.isArray(body.tools));
  assert.equal((body.tools as unknown[]).length, 1);
  assert.equal(body.parallel_tool_calls, false);
  assert.equal(body.tool_choice, "auto");
});

test("invoke: temperature/maxTokens — options override config; otherwise omitted", async () => {
  // Case 1: options override config.
  {
    const mock = makeMockFetch(() =>
      jsonResponse({
        choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
      })
    );
    const p = new OpenAIProvider({
      baseURL: "https://x",
      model: "gpt-4o",
      temperature: 0.1,
      maxTokens: 100,
      fetch: mock.fn,
    });
    await p.invoke({
      messages: baseMessages,
      temperature: 0.9,
      maxTokens: 500,
    });
    const body = getRequestBody(mock.calls[0]);
    assert.equal(body.temperature, 0.9);
    assert.equal(body.max_tokens, 500);
  }
  // Case 2: only config provided.
  {
    const mock = makeMockFetch(() =>
      jsonResponse({
        choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
      })
    );
    const p = new OpenAIProvider({
      baseURL: "https://x",
      model: "gpt-4o",
      temperature: 0.2,
      maxTokens: 10,
      fetch: mock.fn,
    });
    await p.invoke({ messages: baseMessages });
    const body = getRequestBody(mock.calls[0]);
    assert.equal(body.temperature, 0.2);
    assert.equal(body.max_tokens, 10);
  }
  // Case 3: neither — keys absent.
  {
    const mock = makeMockFetch(() =>
      jsonResponse({
        choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
      })
    );
    const p = new OpenAIProvider({
      baseURL: "https://x",
      model: "gpt-4o",
      fetch: mock.fn,
    });
    await p.invoke({ messages: baseMessages });
    const body = getRequestBody(mock.calls[0]);
    assert.equal("temperature" in body, false);
    assert.equal("max_tokens" in body, false);
  }
});

test("invoke: parses successful JSON response into InvokeResult", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [
        {
          message: {
            content: "hello world",
            tool_calls: [
              {
                id: "call_1",
                function: { name: "sum", arguments: '{"a":1,"b":2}' },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  const res = await p.invoke({ messages: baseMessages });
  assert.equal(res.content, "hello world");
  assert.equal(res.finishReason, "tool_calls");
  assert.equal(res.toolCalls.length, 1);
  assert.equal(res.toolCalls[0].id, "call_1");
  assert.equal(res.toolCalls[0].name, "sum");
  assert.deepEqual(res.toolCalls[0].arguments, { a: 1, b: 2 });
  assert.equal(res.toolCalls[0].argumentsRaw, '{"a":1,"b":2}');
  assert.equal(res.usage.promptTokens, 10);
  assert.equal(res.usage.completionTokens, 5);
  assert.equal(res.usage.totalTokens, 15);
});

// ---------------------------------------------------------------------------
// invoke — error path
// ---------------------------------------------------------------------------

test("invoke: 401 → AUTH, not retryable", async () => {
  const mock = makeMockFetch(
    () =>
      new Response(JSON.stringify({ error: { message: "bad key" } }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.AUTH);
    assert.equal(err.retryable, false);
    assert.equal(err.statusCode, 401);
    assert.match(err.message, /bad key/);
    return true;
  });
});

test("invoke: 429 → RATE_LIMIT, retryable", async () => {
  const mock = makeMockFetch(
    () =>
      new Response(JSON.stringify({ error: { message: "slow down" } }), {
        status: 429,
        headers: { "Content-Type": "application/json" },
      })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.RATE_LIMIT);
    assert.equal(err.retryable, true);
    return true;
  });
});

test("invoke: 500 → SERVER, retryable", async () => {
  const mock = makeMockFetch(
    () =>
      new Response(JSON.stringify({ error: { message: "boom" } }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.SERVER);
    assert.equal(err.retryable, true);
    return true;
  });
});

test("invoke: 400 + code=context_length_exceeded → CONTEXT_LENGTH", async () => {
  const mock = makeMockFetch(
    () =>
      new Response(
        JSON.stringify({
          error: {
            code: "context_length_exceeded",
            message: "too many tokens",
          },
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      )
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.CONTEXT_LENGTH);
    assert.equal(err.retryable, false);
    return true;
  });
});

test("invoke: 400 + no code but message matches → CONTEXT_LENGTH (regex fallback)", async () => {
  const mock = makeMockFetch(
    () =>
      new Response(
        JSON.stringify({
          error: { message: "Your prompt exceeds the maximum context length." },
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      )
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.CONTEXT_LENGTH);
    return true;
  });
});

test("invoke: 400 + code=content_filter → CONTENT_FILTER", async () => {
  const mock = makeMockFetch(
    () =>
      new Response(
        JSON.stringify({
          error: { code: "content_filter", message: "blocked" },
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      )
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.CONTENT_FILTER);
    assert.equal(err.retryable, false);
    return true;
  });
});

test("invoke: response is not JSON → INVALID_RESPONSE", async () => {
  const mock = makeMockFetch(
    () =>
      new Response("plain text not json", {
        status: 200,
        headers: { "Content-Type": "text/plain" },
      })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.INVALID_RESPONSE);
    return true;
  });
});

test("invoke: JSON missing choices[0] → INVALID_RESPONSE", async () => {
  const mock = makeMockFetch(() => jsonResponse({ choices: [] }));
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.INVALID_RESPONSE);
    return true;
  });
});

test("invoke: fetch throws TypeError → NETWORK", async () => {
  const mock = makeMockFetch(() => {
    throw new TypeError("Failed to fetch");
  });
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(p.invoke({ messages: baseMessages }), (err: unknown) => {
    assert.ok(err instanceof LLMError);
    assert.equal(err.type, LLMErrorType.NETWORK);
    assert.equal(err.retryable, true);
    return true;
  });
});

test("invoke: pre-aborted signal → ABORTED", async () => {
  const controller = new AbortController();
  controller.abort();
  // When fetch is called with an already-aborted signal, fetch throws AbortError.
  const mock = makeMockFetch((_url, init) => {
    if (init.signal?.aborted) {
      const err = new Error("The operation was aborted");
      err.name = "AbortError";
      throw err;
    }
    return jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    });
  });
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await assert.rejects(
    p.invoke({ messages: baseMessages, signal: controller.signal }),
    (err: unknown) => {
      assert.ok(err instanceof LLMError);
      assert.equal(err.type, LLMErrorType.ABORTED);
      assert.equal(err.retryable, false);
      return true;
    }
  );
});

// ---------------------------------------------------------------------------
// stream — success path
// ---------------------------------------------------------------------------

test("stream: yields text-delta, tool-call sequence, then done", async () => {
  const sse =
    `data: ${JSON.stringify({
      choices: [{ delta: { content: "Hello " } }],
    })}\n\n` +
    `data: ${JSON.stringify({
      choices: [{ delta: { content: "world" } }],
    })}\n\n` +
    `data: ${JSON.stringify({
      choices: [
        {
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_a",
                function: { name: "sum", arguments: '{"a":' },
              },
            ],
          },
        },
      ],
    })}\n\n` +
    `data: ${JSON.stringify({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, function: { arguments: "1," } },
            ],
          },
        },
      ],
    })}\n\n` +
    `data: ${JSON.stringify({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, function: { arguments: '"b":2}' } },
            ],
          },
        },
      ],
    })}\n\n` +
    `data: ${JSON.stringify({
      choices: [{ delta: {}, finish_reason: "tool_calls" }],
      usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
    })}\n\n` +
    `data: [DONE]\n\n`;

  const mock = makeMockFetch(() =>
    responseFromChunks([sse], {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });

  const chunks: StreamChunk[] = [];
  for await (const c of p.stream({ messages: baseMessages })) {
    chunks.push(c);
  }

  // Verify request body had stream:true + stream_options.
  const reqBody = getRequestBody(mock.calls[0]);
  assert.equal(reqBody.stream, true);
  assert.deepEqual(reqBody.stream_options, { include_usage: true });

  // Text deltas.
  const texts = chunks
    .filter((c) => c.type === "text-delta")
    .map((c) => (c as { type: "text-delta"; text: string }).text);
  assert.deepEqual(texts, ["Hello ", "world"]);

  // Tool-call-start before any deltas, only once.
  const startIdx = chunks.findIndex((c) => c.type === "tool-call-start");
  const firstDeltaIdx = chunks.findIndex((c) => c.type === "tool-call-delta");
  assert.ok(startIdx >= 0 && startIdx < firstDeltaIdx);
  const start = chunks[startIdx] as Extract<
    StreamChunk,
    { type: "tool-call-start" }
  >;
  assert.equal(start.id, "call_a");
  assert.equal(start.name, "sum");
  assert.equal(start.index, 0);

  // Argument deltas concat to full JSON.
  const argDeltas = chunks
    .filter((c) => c.type === "tool-call-delta")
    .map(
      (c) =>
        (c as Extract<StreamChunk, { type: "tool-call-delta" }>).argumentsDelta
    );
  assert.equal(argDeltas.join(""), '{"a":1,"b":2}');

  // tool-call-end appears after deltas, before done.
  const endIdx = chunks.findIndex((c) => c.type === "tool-call-end");
  const doneIdx = chunks.findIndex((c) => c.type === "done");
  assert.ok(endIdx > 0 && endIdx < doneIdx);

  // Done is last; carries finishReason + usage.
  assert.equal(doneIdx, chunks.length - 1);
  const done = chunks[doneIdx] as Extract<StreamChunk, { type: "done" }>;
  assert.equal(done.finishReason, "tool_calls");
  assert.equal(done.usage.totalTokens, 10);
  assert.equal(done.usage.promptTokens, 7);
  assert.equal(done.usage.completionTokens, 3);
});

// ---------------------------------------------------------------------------
// stream — error path
// ---------------------------------------------------------------------------

test("stream: HTTP 401 throws AUTH before any chunk", async () => {
  const mock = makeMockFetch(
    () =>
      new Response(JSON.stringify({ error: { message: "bad key" } }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  let caught: unknown;
  try {
    for await (const _c of p.stream({ messages: baseMessages })) {
      // unreachable
    }
  } catch (err) {
    caught = err;
  }
  assert.ok(caught instanceof LLMError);
  assert.equal((caught as LLMError).type, LLMErrorType.AUTH);
});

test("stream: unparseable data line is skipped, stream continues", async () => {
  const sse =
    `: heartbeat\ndata: not json at all\n\n` +
    `data: ${JSON.stringify({
      choices: [{ delta: { content: "after" } }],
    })}\n\n` +
    `data: ${JSON.stringify({
      choices: [{ delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    })}\n\n` +
    `data: [DONE]\n\n`;

  const mock = makeMockFetch(() =>
    responseFromChunks([sse], {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });

  const chunks: StreamChunk[] = [];
  for await (const c of p.stream({ messages: baseMessages })) {
    chunks.push(c);
  }
  const texts = chunks
    .filter((c) => c.type === "text-delta")
    .map((c) => (c as { type: "text-delta"; text: string }).text);
  assert.deepEqual(texts, ["after"]);
  const done = chunks[chunks.length - 1];
  assert.equal(done.type, "done");
});

test("stream: abort mid-stream → ABORTED", async () => {
  const controller = new AbortController();
  // Stream that yields one chunk then never completes.
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      const enc = new TextEncoder();
      c.enqueue(
        enc.encode(
          `data: ${JSON.stringify({
            choices: [{ delta: { content: "first " } }],
          })}\n\n`
        )
      );
      // Never close.
    },
  });
  const mock = makeMockFetch(
    () =>
      new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });

  let caught: unknown;
  const collected: StreamChunk[] = [];
  try {
    for await (const c of p.stream({
      messages: baseMessages,
      signal: controller.signal,
    })) {
      collected.push(c);
      controller.abort();
    }
  } catch (err) {
    caught = err;
  }
  assert.ok(caught instanceof LLMError);
  assert.equal((caught as LLMError).type, LLMErrorType.ABORTED);
  assert.equal(collected.length, 1);
});

// ---------------------------------------------------------------------------
// Request body details
// ---------------------------------------------------------------------------

test("body: streaming request sets stream:true + stream_options.include_usage", async () => {
  const sse = `data: [DONE]\n\n`;
  const mock = makeMockFetch(() =>
    responseFromChunks([sse], {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  for await (const _c of p.stream({ messages: baseMessages })) {
    // drain
  }
  const body = getRequestBody(mock.calls[0]);
  assert.equal(body.stream, true);
  assert.deepEqual(body.stream_options, { include_usage: true });
});

test("body: omits tools/tool_choice/parallel_tool_calls when no tools", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await p.invoke({ messages: baseMessages });
  const body = getRequestBody(mock.calls[0]);
  assert.equal("tools" in body, false);
  assert.equal("tool_choice" in body, false);
  assert.equal("parallel_tool_calls" in body, false);
});

test('body: toolChoice "auto" sent verbatim', async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await p.invoke({
    messages: baseMessages,
    tools: [
      {
        name: "t",
        inputSchema: z.object({}),
      },
    ],
    toolChoice: "auto",
  });
  const body = getRequestBody(mock.calls[0]);
  assert.equal(body.tool_choice, "auto");
});

test("body: toolChoice {name} translates to function-typed object", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await p.invoke({
    messages: baseMessages,
    tools: [
      {
        name: "myfn",
        inputSchema: z.object({}),
      },
    ],
    toolChoice: { name: "myfn" },
  });
  const body = getRequestBody(mock.calls[0]);
  assert.deepEqual(body.tool_choice, {
    type: "function",
    function: { name: "myfn" },
  });
});

// ---------------------------------------------------------------------------
// Reviewer-found gaps (round 2)
// ---------------------------------------------------------------------------

test("invoke: omits Authorization header when apiKey not configured", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    fetch: mock.fn,
  });
  await p.invoke({ messages: baseMessages });
  const headers = mock.calls[0]!.init.headers as Record<string, string>;
  assert.equal("Authorization" in headers, false);
});

test("invoke: HTTP error with non-JSON body falls back to status text", async () => {
  const mock = makeMockFetch(
    () =>
      new Response("Internal Server Error", {
        status: 503,
        statusText: "Service Unavailable",
      })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    apiKey: "k",
    fetch: mock.fn,
  });
  await assert.rejects(
    p.invoke({ messages: baseMessages }),
    (err: unknown) => {
      assert.ok(err instanceof LLMError);
      assert.equal((err as LLMError).type, LLMErrorType.SERVER);
      assert.match((err as LLMError).message, /503/);
      return true;
    }
  );
});

test("stream: integration with multi tool-call (index 0 + index 1)", async () => {
  // Two tool calls, alternating index 0 and 1, args split across chunks.
  const sse =
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"search","arguments":""}}]}}]}\n\n' +
    'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"fetch","arguments":""}}]}}]}\n\n' +
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\""}}]}}]}\n\n' +
    'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"u"}}]}}]}\n\n' +
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"hi\\"}"}}]}}]}\n\n' +
    'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"rl\\":\\"x\\"}"}}]}}]}\n\n' +
    'data: {"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n' +
    "data: [DONE]\n\n";
  const mock = makeMockFetch(() =>
    responseFromChunks([sse], {
      headers: { "content-type": "text/event-stream" },
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    apiKey: "k",
    fetch: mock.fn,
  });

  const out: StreamChunk[] = [];
  for await (const c of p.stream({ messages: baseMessages })) out.push(c);

  // Two starts (one per index), six deltas (3 per index counting empty-string
  // chunk filtered), two ends (sorted by index), one done.
  const starts = out.filter((c) => c.type === "tool-call-start");
  const ends = out.filter((c) => c.type === "tool-call-end");
  const done = out[out.length - 1]!;

  assert.equal(starts.length, 2);
  assert.deepEqual(
    starts.map((c) => (c as Extract<StreamChunk, { type: "tool-call-start" }>).index),
    [0, 1]
  );
  assert.equal(ends.length, 2);
  assert.deepEqual(
    ends.map((c) => (c as Extract<StreamChunk, { type: "tool-call-end" }>).index),
    [0, 1]
  );
  assert.equal(done.type, "done");

  // Reassemble each tool's args from deltas, ensure they form valid JSON.
  const args0 = out
    .filter(
      (c): c is Extract<StreamChunk, { type: "tool-call-delta" }> =>
        c.type === "tool-call-delta" && c.index === 0
    )
    .map((c) => c.argumentsDelta)
    .join("");
  const args1 = out
    .filter(
      (c): c is Extract<StreamChunk, { type: "tool-call-delta" }> =>
        c.type === "tool-call-delta" && c.index === 1
    )
    .map((c) => c.argumentsDelta)
    .join("");
  assert.deepEqual(JSON.parse(args0), { q: "hi" });
  assert.deepEqual(JSON.parse(args1), { url: "x" });
});

test("invoke: tool_choice='none' passes through to body", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    apiKey: "k",
    fetch: mock.fn,
  });
  await p.invoke({
    messages: baseMessages,
    tools: [{ name: "t", inputSchema: z.object({}) }],
    toolChoice: "none",
  });
  const body = JSON.parse(mock.calls[0]!.init.body as string) as {
    tool_choice: unknown;
  };
  assert.equal(body.tool_choice, "none");
});

test("invoke: tool_choice='required' passes through to body", async () => {
  const mock = makeMockFetch(() =>
    jsonResponse({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    })
  );
  const p = new OpenAIProvider({
    baseURL: "https://x",
    model: "gpt-4o",
    apiKey: "k",
    fetch: mock.fn,
  });
  await p.invoke({
    messages: baseMessages,
    tools: [{ name: "t", inputSchema: z.object({}) }],
    toolChoice: "required",
  });
  const body = JSON.parse(mock.calls[0]!.init.body as string) as {
    tool_choice: unknown;
  };
  assert.equal(body.tool_choice, "required");
});
