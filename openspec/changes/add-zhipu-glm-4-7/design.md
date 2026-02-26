# Design: Zhipu GLM-4.7 Integration

## Summary

This design details the technical implementation of the Zhipu GLM-4.7 provider. The core challenge is injecting the non-standard `thinking` parameter into the request body while maintaining compatibility with the OpenAI-style chat completion endpoint that Zhipu uses.

## Architecture Changes

### New Provider Class: `Zhipu.ts`

We will create a new class `Zhipu` that extends `BaseLLM`.

```typescript
class Zhipu extends BaseLLM {
  static providerName = "zhipu";
  static defaultOptions = {
    apiBase: "https://open.bigmodel.cn/api/paas/v4/",
    model: "glm-4.7",
  };

  // ... implementation details
}
```

### Request Construction

The `_getGenerateOptions` or equivalent method in `BaseLLM` (often `_convertArgs` or similar depending on base class specifics, checking `OpenAI` implementation as reference) needs to be overridden or the `fetch` call adjusted.

Since `BaseLLM` often delegates to `openaiAdapter` for OpenAI-compatible providers, we have two options:

1.  **Use `openai` provider with configuration**: This is hardest because `thinking` object is not standard.
2.  **Custom Provider (Selected Approach)**: Implement `_streamChat` (and `streamComplete`) to construct the specific body required by Zhipu.

**Request Body Structure:**

```json
{
  "model": "glm-4.7",
  "messages": [...],
  "thinking": { "type": "enabled" }, // <--- The key difference
  "stream": true,
  "max_tokens": 65536,
  "temperature": 1.0
}
```

### Template Detection

In `core/llm/autodetect.ts`, we will map `glm-4.7` to `chatml` template, which is the standard for GLM-4 models.

## Implementation Details

### `core/llm/llms/Zhipu.ts`

- **Extends**: `BaseLLM`
- **Implements**: `_streamChat` to handle the specific request body.
- **Handling Streams**: Zhipu's stream format is standard SSE (Server-Sent Events) with `data: [DONE]` termination, similar to OpenAI. We can likely reuse `streamResponse` utility.

### Code Snippets

#### `Zhipu.ts`

```typescript
import { BaseLLM } from "../index.js";
import { ChatMessage, CompletionOptions, LLMOptions } from "../../index.js";

class Zhipu extends BaseLLM {
  static providerName = "zhipu";

  // Maps Continue's internal options to Zhipu's API body
  private _convertArgs(
    options: CompletionOptions,
    prompt: string | ChatMessage[],
  ) {
    const finalOptions: any = {
      model: this.model,
      stream: options.stream,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      thinking: { type: "enabled" }, // Force enabled for 4.7
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
    // Implementation using fetch and streamResponse
    // ...
  }
}
```

## Alternatives Considered

- **Generic OpenAI Provider**: Attempted to use `openai` provider with `completionOptions` hack.
  - _Why Rejected_: The `thinking` parameter is a nested object `{"type": "enabled"}`, and standard OpenAI adapters might strip unknown fields or not support passing arbitrary nested objects easily without dirty hacks. A dedicated provider is cleaner and more maintainable for future Zhipu-specific features (like `web_search`).

## Verification Plan

- **Unit Tests**: Add a test in `core/llm/llms/Zhipu.test.ts` (if testing infrastructure allows) to verify request body construction.
- **Manual Verification**: Run `glm-4.7` from GUI and check `Output` channel or proxy logs to ensure `thinking` param was sent.
