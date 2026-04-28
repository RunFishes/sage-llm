/**
 * OpenAI provider — implements `LLMProvider` against OpenAI's chat.completions
 * endpoint. Also works for any OpenAI-compatible API (DeepSeek, Qwen, vllm, etc.).
 *
 * This file does flow control: build request, fetch, classify errors, drive
 * streaming. All format conversion lives in `./openai-translate.ts`.
 */

import { LLMError, LLMErrorType } from "../errors.ts";
import { parseSSE } from "../sse.ts";
import type {
  InvokeOptions,
  InvokeResult,
  StreamChunk,
} from "../types.ts";
import {
  extractOpenAIError,
  messagesToOpenAI,
  OpenAIStreamAccumulator,
  parseOpenAIResponse,
  toolChoiceToOpenAI,
  toolsToOpenAI,
} from "./openai-translate.ts";
import type { LLMProvider, ProviderConfig } from "./types.ts";

export interface OpenAIConfig extends ProviderConfig {
  /** Optional `OpenAI-Organization` header. */
  organization?: string;
  /**
   * Path appended to baseURL. Defaults to `/chat/completions`.
   * Override for vendors that don't follow OpenAI's URL convention.
   */
  chatPath?: string;
}

const DEFAULT_CHAT_PATH = "/chat/completions";

export class OpenAIProvider implements LLMProvider {
  readonly name = "openai";
  readonly model: string;

  private readonly config: Required<
    Omit<OpenAIConfig, "apiKey" | "headers" | "organization" | "temperature" | "maxTokens">
  > &
    Pick<OpenAIConfig, "apiKey" | "headers" | "organization" | "temperature" | "maxTokens">;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(config: OpenAIConfig) {
    if (!config.baseURL) throw new Error("OpenAIProvider: baseURL is required");
    if (!config.model) throw new Error("OpenAIProvider: model is required");

    this.model = config.model;
    this.config = {
      baseURL: stripTrailingSlash(config.baseURL),
      model: config.model,
      chatPath: config.chatPath ?? DEFAULT_CHAT_PATH,
      fetch: config.fetch ?? globalThis.fetch,
      apiKey: config.apiKey,
      organization: config.organization,
      headers: config.headers,
      temperature: config.temperature,
      maxTokens: config.maxTokens,
    };
    // Bind fetch to globalThis to avoid `Illegal invocation` when default is used.
    this.fetchImpl = (config.fetch ?? globalThis.fetch).bind(globalThis);
  }

  async invoke(options: InvokeOptions): Promise<InvokeResult> {
    const body = this.buildRequestBody(options, /* stream */ false);
    const response = await this.send(body, options.signal);
    if (!response.ok) await this.throwHTTPError(response);

    let data: unknown;
    try {
      data = await response.json();
    } catch (err) {
      throw new LLMError(
        LLMErrorType.INVALID_RESPONSE,
        "Failed to parse OpenAI response as JSON",
        { cause: err }
      );
    }

    try {
      return parseOpenAIResponse(data);
    } catch (err) {
      throw new LLMError(
        LLMErrorType.INVALID_RESPONSE,
        (err as Error).message,
        { cause: err, raw: data }
      );
    }
  }

  async *stream(options: InvokeOptions): AsyncGenerator<StreamChunk, void, void> {
    const body = this.buildRequestBody(options, /* stream */ true);
    const response = await this.send(body, options.signal);
    if (!response.ok) await this.throwHTTPError(response);

    const accumulator = new OpenAIStreamAccumulator();

    try {
      for await (const event of parseSSE(response, options.signal)) {
        if (event.data === "[DONE]") break;
        let chunk: unknown;
        try {
          chunk = JSON.parse(event.data);
        } catch {
          // Some compatible providers send keep-alive comments or partial junk.
          // Skip lines we can't parse instead of aborting the stream.
          continue;
        }
        for (const out of accumulator.process(chunk as never)) {
          yield out;
        }
      }
      for (const out of accumulator.finish()) yield out;
    } catch (err) {
      if (err instanceof LLMError) throw err;
      throw LLMError.fromFetchError(err);
    }
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private buildRequestBody(options: InvokeOptions, stream: boolean) {
    const body: Record<string, unknown> = {
      model: this.config.model,
      messages: messagesToOpenAI(options.messages),
      stream,
    };

    if (options.tools && options.tools.length > 0) {
      body.tools = toolsToOpenAI(options.tools);
      body.parallel_tool_calls = false;
      const tc = toolChoiceToOpenAI(options.toolChoice);
      if (tc !== undefined) body.tool_choice = tc;
    }

    const temperature = options.temperature ?? this.config.temperature;
    if (temperature !== undefined) body.temperature = temperature;

    const maxTokens = options.maxTokens ?? this.config.maxTokens;
    if (maxTokens !== undefined) body.max_tokens = maxTokens;

    if (stream) {
      // Ask OpenAI to include a final usage frame in the stream.
      body.stream_options = { include_usage: true };
    }

    return body;
  }

  private async send(
    body: unknown,
    signal: AbortSignal | undefined
  ): Promise<Response> {
    const url = `${this.config.baseURL}${this.config.chatPath}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(this.config.headers ?? {}),
    };
    if (this.config.apiKey) {
      headers["Authorization"] = `Bearer ${this.config.apiKey}`;
    }
    if (this.config.organization) {
      headers["OpenAI-Organization"] = this.config.organization;
    }

    try {
      return await this.fetchImpl(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal,
      });
    } catch (err) {
      throw LLMError.fromFetchError(err);
    }
  }

  /** Reads the error body and throws a typed `LLMError`. Always throws. */
  private async throwHTTPError(response: Response): Promise<never> {
    let body: unknown = undefined;
    try {
      body = await response.json();
    } catch {
      // Body was not JSON — that's fine, fall back to status text.
    }
    const { code, message } = extractOpenAIError(
      body,
      `HTTP ${response.status} ${response.statusText}`
    );
    throw LLMError.fromHTTPError(response.status, code, message, body);
  }
}

function stripTrailingSlash(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}
