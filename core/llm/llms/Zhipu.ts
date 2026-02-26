import { streamSse } from "@continuedev/fetch";
import { ChatMessage, CompletionOptions, LLMOptions } from "../../index.js";
import { BaseLLM } from "../index.js";

class Zhipu extends BaseLLM {
  static providerName = "zhipu";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://open.bigmodel.cn/api/paas/v4/",
    model: "glm-4.7",
  };

  constructor(options: LLMOptions) {
    super(options);
    this.templateMessages = undefined;
  }

  private _convertArgs(
    options: CompletionOptions,
    prompt: string | ChatMessage[],
  ) {
    const finalOptions: any = {
      model: options.model,
      stream: options.stream ?? true,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      thinking: { type: "enabled" }, // Force enabled for 4.7 as per requirements
    };

    if (Array.isArray(prompt)) {
      finalOptions.messages = prompt;
    } else {
      finalOptions.messages = [{ role: "user", content: prompt }];
    }

    return finalOptions;
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const response = await this.fetch(this.apiBase + "chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(this._convertArgs(options, messages)),
      signal,
    });

    for await (const chunk of streamSse(response)) {
      if (chunk.choices && chunk.choices.length > 0) {
        const delta = chunk.choices[0].delta;
        if (delta.content) {
          yield { role: "assistant", content: delta.content };
        }
      }
    }
  }
}

export default Zhipu;
