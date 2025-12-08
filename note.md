# Project Request: Desktop AI Screen Logger (LumosLog)

## 1. 项目概述 (Project Overview)
我要开发一个 Python 桌面应用程序，名为 "LumosLog"。
它的核心功能是：后台自动截图，使用本地算法判断画面是否变化。如果变化，调用 Vision AI 分析当前屏幕活动；如果未变，则复用上一条记录。最终目标是生成一份“今日活动日报”。

**核心原则：** 1. **架构简单**：代码结构清晰，易于维护。
2. **极致省钱**：通过本地图像对比算法，最大程度减少 API 调用。
3. **低资源占用**：优化内存管理，支持长期后台运行。
4. **美观现代**：使用 Material Design 风格。

## 2. 技术栈 (Tech Stack)
* **语言**: Python 3.10+
* **GUI 框架**: PySide6 (Qt for Python)
* **UI 美化**: `qt_material` (使用 'dark_teal.xml' 主题)
* **截图工具**: `mss` (高性能跨平台截图)
* **图像处理**: `Pillow` (PIL) - 用于压缩和计算差异
* **AI 接口**: `openai` (调用 gpt-4o-mini)
* **环境管理**: `python-dotenv` (管理 API Key)

## 3. 功能需求 (Functional Requirements)

### 3.1 主界面 (Main UI)
* **状态指示**: 显示当前运行状态（运行中/待机），使用颜色区分（绿色/灰色）。
* **控制按钮**: 
    * "开始监控 / 停止监控" (Toggle Button)
    * "生成今日日报" (Action Button)
* **日志显示区**: 一个只读的文本框，实时滚动显示带时间戳的活动日志。
* **系统托盘 (System Tray)**:
    * 点击窗口关闭按钮时，**不退出程序**，而是最小化到托盘。
    * 托盘右键菜单包含：显示主界面、退出程序。

### 3.2 监控逻辑 (Worker Thread)
这是一个独立的 `QThread`，逻辑如下：
1.  **截图**: 每隔 `INTERVAL` 秒（默认 60s）截取主屏幕。
2.  **预处理**: 将图片 Resize 到 `512x512` (保持比例)，并转换为 RGB。
3.  **本地去重 (关键)**:
    * 计算 `当前图片` 与 `上一张图片` 的 RMS (Root Mean Square) 差异值。
    * 如果差异值 < `THRESHOLD` (例如 10)，判定为“画面静止”。
    * **静止处理**: 不调用 API，直接记录日志 "同上 (静止)"。
    * **变化处理**: 调用 OpenAI Vision API。
4.  **AI 分析**:
    * 模型: (用户自定义)
    * Prompt: "用5-10个字简短描述屏幕上正在进行的主要任务（例如：编写Python代码、浏览B站视频、空闲待机）。"
    * Token限制: `max_tokens=30`
5.  **内存优化**: 每次循环结束显式调用 `gc.collect()`，并删除不再需要的图片对象。

### 3.3 数据存储 (Storage)
* 格式: `JSONL` (JSON Lines)。
* 文件名: `daily_log_{YYYY-MM-DD}.jsonl`。
* 字段: `{"timestamp": "HH:MM:SS", "activity": "写代码", "is_api_call": true/false}`

### 3.4 日报生成 (Report Generation)
* 读取当天的 JSONL 文件。
* 将日志合并为纯文本 Context。
* 调用 LLM 生成总结。
* **输出内容**: 时间轴摘要、主要成就、时间分布统计。
* **展示方式**: 弹出一个新的子窗口显示 Markdown 渲染后的日报。

## 4. 非功能需求 (Non-Functional Requirements)
* **异常处理**: 网络断开或 API 报错时，程序不应崩溃，而是记录错误并稍后重试。
* **配置化**: API Key 必须从 `.env` 文件或环境变量读取，不要硬编码。

## 5. 代码结构建议
请生成单一文件 `app.py` 或最简模块化结构，包含以下类：
1.  `MonitorWorker(QThread)`: 处理截图、Diff 和 API 逻辑。
2.  `AppWindow(QWidget)`: 主界面 UI。
3.  `ReportWindow(QWidget)`: 显示日报结果。
4.  `main`: 程序入口，负责应用 qt_material 主题。

---

我希望选择模型和填api是一个可以交互的用户界面。默认模型是硅基流动。Qwen/Qwen3-VL-8B-Instruct
我希望有一个地方可以输出完整详细的调试信息，每一步程序干了什么，都说出来。
然后我们先进行最小单元的尝试，先试一下api调用，返回日志这一步能否完成
import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
        }
    ],
    "stream": False,
    "max_tokens": 4096,
    "enable_thinking": False,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "stop": [],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": { "type": "text" },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "<string>",
                "description": "<string>",
                "parameters": {},
                "strict": False
            }
        }
    ]
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)

---

分析画面是否变化的功能是怎么实现的，我应该怎么测试它的效果。

现在模型识别到的动作太笼统了，比如编写代码，我要知道编写的是哪个项目的文档，比如查看api文档，是在哪个网页查看的api文档？要实现这个功能，应该要调一下提示词吧，然后还要获取前台的窗口信息之类的？请你思考一下，我不清楚。
然后，日报的格式，要是格式化的，比如几点到几点：做了某某事，不要是段落性的陈述。

我觉得生成日报的模型可以用贵一点的，请你分开写填api的地方。日报模型我用gemini 2.5 flash。
你也要提供一个按钮，让我可以直接复制今天（也可能是过去的一天）的结构化的日志信息。拼接上提示词，让我可以直接发送给ai。

我还是觉得，这个视觉处理识别之后，生成的信息和单独发送一个窗口句柄，获取的信息有啥不一样呢？

请你把测试中的测试api连通的提示词改成立即截图，然后让ai尽可能多地返回所获得的信息。因为我想知道截图给ai最多能获得多少信息。

单独的日报生成模型，也要填写baseurl吧？也要有单独的测试连通？

