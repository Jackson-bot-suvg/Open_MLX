# VS Code Extension 架构详解 (VSCODE_AGENT_CN.md)

本文档详细解析 `extensions/vscode` 模块的架构，帮助你理解并修改 VS Code 插件部分。

## 1. 入口与生命周期

插件的入口点在 `package.json` 中定义为 `out/extension.js`，对应的源码是 `src/extension.ts`。

- **`src/extension.ts`**: 这是一个轻量级的入口，它通过动态导入 `src/activation/activate.ts` 来启动插件。这主要是为了性能优化（Lazy Loading）。
- **`src/activation/activate.ts`**:
  - **核心初始化**: 创建 `GlobalContext`，进行环境检查。
  - **`VsCodeExtension` 初始化**: 实例化核心控制类 `VsCodeExtension`。
  - **API 注册**: 创建 `VsCodeContinueApi` 并返回，供其他插件调用。

## 2. 核心控制类 (`VsCodeExtension`)

`src/extension/VsCodeExtension.ts` 是插件的“大脑”。它负责协调各个组件：

- **初始化组件**:
  - `VsCodeIde`: 实现 IDE 接口，封装 VS Code API。
  - `Core`: Continue 的核心逻辑（在 `core/` 目录）。
  - `ContinueGUIWebviewViewProvider`: 侧边栏 GUI。
  - `ContinueConsoleWebviewViewProvider`: 底部面板 Console。
- **事件监听**:
  - `vscode.workspace.onDidChangeTextDocument`: 监听文件变化，触发自动补全或索引更新。
  - `vscode.window.onDidChangeTextEditorSelection`: 监听光标移动。
  - `fs.watch`: 监听配置文件变化 (`config.json`, `config.ts`) 并自动重载。
- **依赖注入**: 将 `ide`, `configHandler` 等注入到 `Core` 和 Messenger 中。

## 3. IDE 接口实现 (`VsCodeIde`)

`src/VsCodeIde.ts` 实现了 core 定义的 `IDE` 接口。它是 Core 逻辑与 VS Code 编辑器之间的桥梁。

- **主要功能**:
  - `readFile`, `writeFile`: 文件系统操作。
  - `getOpenFiles`, `getVisibleFiles`: 获取编辑器状态。
  - `showLines`, `openFile`: 控制编辑器跳转。
  - `runCommand`: 在终端运行命令。
  - `getDiff`: 获取 git diff。
- **修改建议**: 如果你需要让 core 获取更多 VS Code 特有的信息（如调试状态、特定插件信息），请在这里添加方法。

## 4. UI 与 Webview (`gui`)

VS Code 插件的 UI (侧边栏聊天窗口) 是一个嵌入的 React 应用，源码位于根目录的 `gui/`。

- **`src/ContinueGUIWebviewViewProvider.ts`**:
  - 负责创建 Webview。
  - 加载 `gui/dist/index.js` (生产环境) 或连接本地开发服务器 (开发环境)。
  - **消息通信**: 使用 `webviewProtocol` 处理 Extension (后端) 与 Webview (前端) 之间的通信。
- **`src/webviewProtocol.ts`**: 定义了通信协议。

**如果你要“魔改”界面**:

1.  修改根目录下的 `gui/` 代码（React）。
2.  在 `extensions/vscode/src/` 中处理新的消息类型（如果涉及到与后端交互）。

## 5. 消息通信 (`Messenger`)

Continue 使用一种基于请求/响应的消息机制：

- **`core/protocol/messenger.ts`**: 定义了消息传递的基础架构。
- **`src/VsCodeMessenger.ts`**: 专门处理 VS Code 环境下的消息路由。
  - 它将来自 Webview 的请求（如 "llm/streamChat"）转发给 `Core` 处理。
  - 它将 `Core` 的事件（如 "configUpdate"）转发给 Webview。

## 6. 修改指南

### 场景 A：添加一个新的命令

1.  在 `package.json` 的 `contributes.commands` 中注册命令。
2.  在 `src/commands.ts` 中实现命令逻辑。
3.  在 `src/extension/VsCodeExtension.ts` 中注册该命令。

### 场景 B：修改侧边栏 UI

1.  进入 `gui/` 目录进行 React 开发。
2.  如果需要新的数据，在 `Core` 或 `IDE` 接口中添加获取数据的方法。
3.  通过 `webviewProtocol` 添加新的消息类型来传递数据。

### 场景 C：修改自动补全逻辑

1.  主要逻辑在 `core/autocomplete/`。
2.  VS Code 特有的触发逻辑在 `src/autocomplete/completionProvider.ts`。

## 目录结构速查

```
extensions/vscode/
├── package.json          # 插件配置、命令注册、视图容器
├── src/
│   ├── extension.ts      # 入口
│   ├── VsCodeIde.ts      # IDE 接口实现
│   ├── commands.ts       # 命令实现
│   ├── activation/       # 激活逻辑
│   ├── autocomplete/     # 补全提供者 (VS Code 侧)
│   ├── extension/
│   │   └── VsCodeExtension.ts # 核心控制器
│   └── ContinueGUIWebviewViewProvider.ts # 侧边栏 WebView
└── gui/                  # (根目录) React 前端代码
```
