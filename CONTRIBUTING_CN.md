# 为 Continue 做贡献 (Contributing to Continue)

### 本地运行文档服务器

你可以使用以下任一方法在本地运行文档服务器：

#### 方法 1: NPM 脚本

1. 打开终端并导航到项目的 `docs` 子目录。看到 `docusaurus.config.js` 文件说明你在正确的位置。

2. 运行以下命令安装文档服务器所需的依赖：

   ```bash
   npm install
   ```

3. 运行以下命令启动文档服务器：

   ```bash
   npm run dev
   ```

#### 方法 2: VS Code 任务

1. 在项目根目录下打开 VS Code。

2. 打开 VS Code 命令面板 (`cmd/ctrl+shift+p`) 并选择 `Tasks: Run Task`。

3. 寻找并选择 `docs:start` 任务。

这将启动一个本地服务器，你可以在默认浏览器中看到渲染后的文档，通常可以通过 `http://localhost:3000` 访问。

## 🧑‍💻 贡献代码

我们欢迎各种经验水平的开发者贡献代码——从首次贡献者到经验丰富的开源维护者。虽然我们的目标是保持高标准的可靠性和可维护性，但我们也努力让流程尽可能友好和简单。

### 环境搭建与启动 (Environment Setup)

#### 推荐实测环境配置 (Verified Recommended Setup)

为了确保开发环境配置正确（特别是 Python 和 Node.js 版本），**我们强烈建议使用 Conda 环境**。以下是针对本项目的启动步骤：

1.  **创建并激活环境**：
    需要 Python 3.11（用于编译 SQLite 等组件）和 Node.js 20。

    ```bash
    # 创建名为 openmlx 的环境
    conda create -n openmlx python=3.11 nodejs=20 -y

    # 激活环境
    conda activate openmlx
    ```

    > **重要提示**：在执行所有开发命令或启动 VS Code 之前，**必须**先激活此环境。

2.  **安装项目依赖**：
    在项目根目录下运行安装脚本：

    ```bash
    bash scripts/install-dependencies.sh
    ```

#### 依赖要求 (Pre-requisites)

如果你不想使用 Conda，你需要手动满足以下要求：

- Node.js 版本 20.19.0 (LTS) 或更高。你可以从 [nodejs.org](https://nodejs.org/en/download) 获取，或者如果你使用 NVM (Node Version Manager)，可以通过运行以下命令设置正确版本：
  ```bash
  nvm use
  ```
- 全局安装 Vite：
  ```bash
  npm i -g vite
  ```

#### Fork Continue 仓库

1. 前往 [Continue GitHub 仓库](https://github.com/continuedev/continue) 并将其 Fork 到你的 GitHub 账户。

2. 将你的 Fork 仓库 Clone 到本地机器。使用：`git clone https://github.com/YOUR_USERNAME/continue.git`

3. 导航到克隆的目录并确保你在 `main` 分支上。从那里创建你的功能/修复分支，以此类推：`git checkout -b 123-my-feature-branch`

4. 向 `main` 分支发送 Pull Request。

#### VS Code (调试与启动)

**关键步骤**：为了让 VS Code 继承正确的环境变量，**请务必在已激活 `openmlx` 环境的终端中启动 VS Code**：

```bash
conda activate openmlx
code .
```

1. 打开 VS Code 命令面板 (`cmd/ctrl+shift+p`) 并选择 `Tasks: Run Task`，然后选择 `install-all-dependencies`（如果之前没有运行过安装脚本）。

2. **开始调试 (Start Debugging)**：

   1. 切换到 "运行和调试" (Run and Debug) 视图 (`Cmd+Shift+D`)。
   2. 在下拉菜单中选择 **`Launch extension`**。
   3. 点击播放按钮 (F5)。
   4. 这将以调试模式启动扩展，并打开一个新的 VS Code 窗口（即 _Host VS Code_），其中已安装该扩充。
      1. 这个带扩展的新窗口称为 _Host VS Code_。
      2. 你开始调试的原始窗口称为 _Main VS Code_。

3. 如果要打包扩展，请在 `extensions/vscode` 目录下运行 `npm run package`，或者选择 `Tasks: Run Task` 然后选择 `vscode-extension:package`。这将生成 `extensions/vscode/build/continue-{VERSION}.vsix`，你可以通过右键点击并选择 "Install Extension VSIX" 来安装。

##### 调试 (Debugging)

**断点**：可以在调试 `core` 和 `extensions/vscode` 文件夹时使用断点，但目前 **不支持** 在 `gui` 代码中使用断点。

**热重载 (Hot-reloading)**：Vite 启用了热重载，所以如果你对 `gui` 做了任何修改，它们应该会在无需重建的情况下自动生效。在某些情况下，你可能需要刷新 _Host VS Code_ 窗口才能看到更改。

同样，对 `core` 或 `extensions/vscode` 的修改只需通过 `cmd/ctrl+shift+p` "Reload Window" 重新加载 _Host VS Code_ 窗口即可生效。

### 格式化 (Formatting)

Continue 使用 [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) 来格式化 JavaScript/TypeScript。请在 VS Code 中安装 Prettier 扩展并在设置中启用 "Format on Save"（保存时自动格式化）。

### 主题颜色 (Theme Colors)

Continue 有一套命名的主题颜色，我们将它们映射到扩展颜色和 Tailwind 类，可以在 [gui/src/styles/theme.ts](gui/src/styles/theme.ts) 中找到。

使用主题颜色的指南：

- 尽可能使用 Tailwind 颜色。如果在 VS Code 中开发，下载 [Tailwind CSS Intellisense extension](https://marketplace.visualstudio.com/items?itemName=bradlc.vscode-tailwindcss) 以获得极佳的建议。
- 避免在主题之外使用任何显式的类和 CSS 变量（例如 `text-yellow-400`）。

添加/更新主题颜色的指南：

- 选择合理的 VS Code 变量添加/更新到 [gui/src/styles/theme.ts](gui/src/styles/theme.ts)（参见 [这里](https://code.visualstudio.com/api/references/theme-color) 和 [这里](https://www.notion.so/1fa1d55165f78097b551e3bc296fcf76?pvs=25) 获取灵感）。
- 选择合理的 JetBrains 命名颜色添加/更新到 `GetTheme.kt`（旗舰 LLM 可以给你很好的建议）。
- 如果需要，更新 `tailwind.config.js`。
- 使用主题测试页面检查颜色。可以在 dev/debug 模式下通过 `Settings` -> `Help` -> `Theme Test Page` 访问。

### 测试 (Testing)

我们需要混合使用单元测试、功能测试和端到端 (e2e) 测试套件，主要关注功能测试。这些测试会在每个 pull request 上运行。如果你的 PR 导致其中任何一个测试失败，我们会要求你在合并之前解决这个问题。

在做贡献时，请更新或创建适当的测试以帮助验证你实现的正确性。

### 审查流程 (Review Process)

- **初步审查** - 一名维护者将被指派为主要审查者。
- **反馈循环** - 审查者可能会要求更改。我们重视你的工作，但也希望确保代码可维护并遵循我们的模式。
- **批准与合并** - 一旦 PR 获得批准，它将被合并到 `main` 分支。

### 获取帮助 (Getting Help)

加入 [GitHub Discussions](https://github.com/continuedev/continue/discussions) 与维护者和其他贡献者互动。

## 贡献新的 LLM 提供商/模型 (Contributing New LLM Providers/Models)

### 添加 LLM 提供商

Continue 支持十几种不同的 LLM "提供商"，使得在 OpenAI, Ollama, Together, LM Studio, Msty 等平台上运行模型变得容易。你可以在 [这里](https://github.com/continuedev/continue/tree/main/core/llm/llms) 找到所有现有的提供商。如果你发现缺少某一个，可以按以下步骤添加：

1. 在 `core/llm/llms` 目录中创建一个新文件。文件名应该是提供商的名称，并且它应该导出一个扩展了 `BaseLLM` 的类。这个类应该包含以下最小实现。我们建议查看现有的提供商以获取更多详细信息。[LlamaCpp Provider](./core/llm/llms/LlamaCpp.ts) 是一个很好的简单示例。
2. 将你的提供商添加到 [core/llm/llms/index.ts](./core/llm/llms/index.ts) 中的 `LLMs` 数组。
3. 如果你的提供商支持图像，将其添加到 [core/llm/autodetect.ts](./core/llm/autodetect.ts) 中的 `PROVIDER_SUPPORTS_IMAGES` 数组。
4. 在 [`docs/customize/model-providers/more`](./docs/customize/model-providers/more) 中添加你的提供商的文档页面。这应该展示如何在 `config.yaml` 中配置你的提供商的示例，并解释有哪些可用选项。

### 添加模型

虽然任何与受支持提供商配合使用的模型都可以与 Continue 一起使用，但我们保留了一份推荐模型列表，可以从 UI 或 `config.json` 自动配置。添加新模型时应更新以下文件：

- [AddNewModel 页面](./gui/src/pages/AddNewModel/configs/) - 此目录定义了侧边栏模型选择 UI 中显示哪些模型选项。要添加新模型：
  1. 在 [configs/models.ts](./gui/src/pages/AddNewModel/configs/models.ts) 中为模型添加一个 `ModelPackage` 条目，参考文件顶部附近的许多示例。
  2. 将模型添加到 [configs/providers.ts](./gui/src/pages/AddNewModel/configs/providers.ts) 中其提供商的数组内（如果需要，添加提供商）。
- LLM 提供商：由于许多提供商使用自定义字符串来标识模型，你必须添加从 Continue 模型名称（你添加到 `index.d.ts` 的名称）到这些提供商各自模型字符串的转换：[Ollama](./core/llm/llms/Ollama.ts), [Together](./core/llm/llms/Together.ts), 和 [Replicate](./core/llm/llms/Replicate.ts)。你可以在这里找到它们完整的模型列表：[Ollama](https://ollama.ai/library), [Together](https://docs.together.ai/docs/inference-models), [Replicate](https://replicate.com/collections/streaming-language-models)。
- [Prompt Templates](./core/llm/autodetect.ts) - 在此文件中你会找到 `autodetectTemplateType` 函数。确保对于你刚刚添加的模型名称，此函数返回正确的模板类型。这假设该模型的聊天模板已经在 Continue 中内置。如果没有，你将不得不添加模板类型以及相应的编辑和聊天模板。

## 贡献者许可协议 (CLA)

我们要求所有贡献者接受 CLA，并使其像在你的 PR 上评论一样简单：

1. 打开你的 pull request。
2. 粘贴以下评论（如果你以前签署过，则回复 `recheck`）：

   ```text
   I have read the CLA Document and I hereby sign the CLA
   ```

3. CLA-Assistant 机器人会在仓库中记录你的签名并将状态检查标记为通过。
