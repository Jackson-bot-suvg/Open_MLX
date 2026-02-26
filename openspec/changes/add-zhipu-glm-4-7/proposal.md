# Proposal: Add Zhipu GLM-4.7 Model

## Summary

Add support for **Zhipu GLM-4.7** to Continue, including its unique "thinking" capability.

## Problem

Users currently cannot easily select Zhipu AI's latest GLM-4.7 models from the Continue GUI. They have to manually configure it as a custom OpenAI-compatible provider, which is less user-friendly than having a preset.

## Goals

- [ ] Add `glm-4.7` preset to `gui/src/pages/AddNewModel/configs/models.ts`.
- [ ] Implement `thinking` parameter support for `glm-4.7`.
- [ ] Verify `stream: true` behavior with Zhipu API.

## What Changes

### GUI Configuration

- **`gui/src/pages/AddNewModel/configs/models.ts`**: Add `glm-4.7` definition with `contextLength: 65536`.

### Core Logic

- **`core/llm/llms/Zhipu.ts`**: (New File) Create a dedicated provider for Zhipu because it requires a custom `thinking` object in the request body (`"thinking": { "type": "enabled" }`) which is non-standard for OpenAI-compatible providers.
- **`core/llm/autodetect.ts`**: Map `glm-4.7` to `chatml` template (or appropriate template).
- **`core/llm/index.ts`**: Register the new Zhipu provider.

## Capabilities

### New Capabilities

- `model-zhipu-glm4-7`: Support for Zhipu GLM-4.7 model with thinking capabilities.

## Impact

- **Users**: Can easily select Zhipu models.
- **Codebase**: Minimal impact, mostly configuration and potentially a new provider file.
