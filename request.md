# vLLM 请求全生命周期：一个 Request 的完整旅程

> 从 HTTP 请求到达 API 服务器，到最终输出返回给用户的完整链路。
>
> 核对基准：`vllm` 仓库 `releases/v0.20.0`。该分支默认 `VLLM_USE_V2_MODEL_RUNNER=0`，
> 因此 GPU 执行主链路仍走 `vllm/v1/worker/gpu_model_runner.py`；设置
> `VLLM_USE_V2_MODEL_RUNNER=1` 时会切到拆分后的
> `vllm/v1/worker/gpu/model_runner.py` 等文件，本文在 GPU 阶段单独标注。
>
> 参考对照：vLLM 官方 Architecture Overview（latest developer preview）：
> <https://docs.vllm.ai/en/latest/design/arch_overview/>。官方文档更偏部署级组件概览；
> 本文以 request 视角补全调用栈、队列、循环和数据对象流转。

## 目录

1. [全局概览](#1-全局概览)
2. [Stage 1：API 接收与解析](#2-stage-1api-接收与解析)
3. [Stage 2：渲染与分词](#3-stage-2渲染与分词)
4. [Stage 3：多模态处理](#4-stage-3多模态处理)
5. [Stage 4：提交到引擎](#5-stage-4提交到引擎)
6. [Stage 5：调度排队](#6-stage-5调度排队)
7. [Stage 6：Worker 输入准备](#7-stage-6worker-输入准备)
8. [Stage 7：模型前向推理](#8-stage-7模型前向推理)
9. [Stage 8：采样](#9-stage-8采样)
10. [Stage 9：调度器更新状态](#10-stage-9调度器更新状态)
11. [Stage 10：输出处理与反分词](#11-stage-10输出处理与反分词)
12. [Stage 11：流式返回](#12-stage-11流式返回)
13. [横切关注点](#13-横切关注点)
14. [代码阅读顺序](#14-代码阅读顺序)

---

## 1. 全局概览

一个请求从 HTTP 抵达到最终返回，经过 **API Server、EngineCore、GPU Worker**
三个主要运行上下文（另有 API 侧后台 output handler）和 **11 个阶段**：

```
┌──────────────────────────────────────────────────────────────────────────┐
│  API Server 进程 (FastAPI + Uvicorn)                                      │
│                                                                          │
│  HTTP Request                                                            │
│     │                                                                    │
│     ├─ Stage 1: API 接收与解析 ─── 路由、参数校验、构建请求对象                │
│     ├─ Stage 2: 渲染与分词 ────── Chat Template + Tokenizer             │
│     ├─ Stage 3: 多模态处理 ────── 图片/音频/视频编码                    │
│     ├─ Stage 4: 提交到引擎 ────── InputProcessor + ZMQ 发送             │
│     │                                                                   │
│     │  ═══════════ ZMQ IPC ═══════════                                  │
│     │                                                                   │
│  Engine Core 进程                                                       │
│     │                                                                   │
│     ├─ Stage 5: 调度排队 ──────── Scheduler + KV Cache 分配            │
│     │                                                                   │
│     │  ═══════════ 内部调用 ═══════════                                  │
│     │                                                                   │
│  Worker 进程 (GPU)                                                      │
│     │                                                                   │
│     ├─ Stage 6: Worker 输入准备 ── GPU tensor 构建、block table 更新    │
│     ├─ Stage 7: 模型前向推理 ──── Transformer forward pass              │
│     ├─ Stage 8: 采样 ──────────── Logits → Sampler → Token IDs          │
│     │                                                                   │
│     │  ═══════════ 返回 Engine Core ═══════════                          │
│     │                                                                   │
│     ├─ Stage 9: 调度器更新状态 ── 追加 token、检查停止、释放资源        │
│     │                                                                   │
│     │  ═══════════ ZMQ IPC ═══════════                                  │
│     │                                                                   │
│  API Server 进程                                                        │
│     │                                                                   │
│     ├─ Stage 10: 输出处理 ──────── 反分词、stop string 检查、logprobs   │
│     ├─ Stage 11: 流式返回 ──────── SSE chunks / JSON response           │
│     │                                                                   │
│     ▼                                                                   │
│  HTTP Response                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**注意**：Stage 5-9 在每个 iteration 中重复执行（每步生成一个或多个 token），直到请求完成。

### 阅读路径

建议按三层来读：

1. 先看本节总图，建立进程边界：API Server → EngineCore → GPU Worker → API Server。
2. 再看 Stage 1-4，理解一个 HTTP request 如何被解析、渲染、分词、多模态处理，并提交成
   `EngineCoreRequest`。
3. 最后看 Stage 5-11，理解 request 如何在每个 engine iteration 中被调度、执行、采样、更新状态，
   再被反分词并返回给客户端。

文中有三类图：

- **时序图**：说明调用顺序和跨进程边界，放在对应 stage 附近。
- **调用栈图**：说明某个复杂模块内部如何逐层调用，例如 renderer / multimodal processor。
- **数据对象流转图**：放在附录，用于回顾各阶段核心对象如何变形。


---

## 2. Stage 1：API 接收与解析

### 入口

**文件**: `vllm/entrypoints/openai/api_server.py`

```
run_server() (行 671)
  └── run_server_worker() (行 681)
        ├── build_async_engine_client() → 创建 AsyncLLM 实例
        ├── build_app() → 注册 FastAPI 路由
        └── uvicorn.run()
```

### 路由

**文件**: `vllm/entrypoints/openai/chat_completion/api_router.py`

`POST /v1/chat/completions` → `create_chat_completion()` (行 53)

### 请求对象

**文件**: `vllm/entrypoints/openai/chat_completion/protocol.py`

`ChatCompletionRequest` (行 150) — Pydantic 模型，包含：
- `model`, `messages`, `temperature`, `max_tokens`, `stream`
- `stop`, `stop_token_ids`, `logprobs`
- `response_format`（结构化输出）
- 多模态数据（`image_url`, `audio_url` 等）

### 服务层编排

**文件**: `vllm/entrypoints/openai/chat_completion/serving.py`

`OpenAIServingChat.create_chat_completion()` (行 229)：

1. 获取 tokenizer，创建 reasoning parser（如需要）
2. 调用 `render_chat_request()`，再委托 OpenAIServingRender 做渲染和分词（→ Stage 2-3）
3. 构建 `SamplingParams`（通过 `request.to_sampling_params()`）
4. 调用 `engine_client.generate()`（→ Stage 4）
5. 路由到 `chat_completion_stream_generator()` 或 `chat_completion_full_generator()`（→ Stage 11）

### Client → API Server → AsyncLLM 时序图

这一段关注 HTTP request 如何走到 `AsyncLLM.generate()`。此时还没有进入 EngineCore 的调度循环，
主要发生在 API Server 进程内。图里把 renderer 压缩成一列；Stage 2-3 会继续展开 renderer 和
多模态处理的内部调用栈。

```
Client              API Server / Router             OpenAIServingChat             Renderer / Tokenizer          AsyncLLM
  │                         │                                │                              │                       │
  │ POST /v1/chat/completions                               │                              │                       │
  │────────────────────────>│                                │                              │                       │
  │                         │ create_chat_completion()       │                              │                       │
  │                         │───────────────────────────────>│                              │                       │
  │                         │                                │                              │                       │
  │                         │                                │ 校验 model / request params   │                       │
  │                         │                                │ 创建 chat template context    │                       │
  │                         │                                │─────────────────────────────>│                       │
  │                         │                                │                              │ render_chat_request() │
  │                         │                                │                              │ ├ render_messages()   │
  │                         │                                │                              │ ├ tokenize_prompts()  │
  │                         │                                │                              │ └ process MM inputs   │
  │                         │                                │<─────────────────────────────│                       │
  │                         │                                │ prompt_token_ids / mm data    │                       │
  │                         │                                │                              │                       │
  │                         │                                │ request.to_sampling_params()  │                       │
  │                         │                                │ result_generator =            │                       │
  │                         │                                │   engine_client.generate(...) │                       │
  │                         │                                │──────────────────────────────────────────────────────>│
  │                         │                                │                              │                       │
  │                         │                                │ 返回 async generator          │                       │
  │                         │<───────────────────────────────│                              │                       │
  │                         │ StreamingResponse / JSON path  │                              │                       │
  │                         │                                │                              │                       │
  │<────────────────────────│ 后续由 Stage 11 持续写 SSE/JSON │                              │                       │
```

几个容易混淆的点：

- `engine_client` 是 serving 层看到的接口名；V1 异步服务里通常就是 `AsyncLLM` 实例。
- `engine_client.generate(...)` 返回的是 async generator。调用它后，真正的 request 提交发生在
  generator 被迭代、进入 `AsyncLLM.generate()` 时。
- 流式响应时，API Server 不会等请求完整生成完才返回；它持有这个 generator，后续每拿到一个
  `RequestOutput` 就转成一段 SSE。

---

## 3. Stage 2：渲染与分词

### 先区分 Serving Render 和 Renderer

这层名字容易混：

- `OpenAIServingChat`：Chat Completions 的业务 serving 层。它知道 OpenAI API 语义、model 校验、
  tool/reasoning parser、SamplingParams、SSE/JSON 响应，但不直接实现 chat template 和多模态处理。
- `OpenAIServingRender`：可复用的“请求预处理服务层”。GPU-less render endpoint 会直接用它；
  `OpenAIServingChat.render_chat_request()` 也会在完成 engine-aware checks 后委托给它。
- `BaseRenderer`：模型无关的 renderer 骨架，定义四步主流程：
  `render_messages` → `tokenize_prompts` → `_apply_prompt_extras` → `process_for_engine`。
- `HfRenderer`：`BaseRenderer` 的 HF tokenizer 实现，主要负责解析 OpenAI chat messages、
  选择 content format、调用 Jinja chat template，并把多模态原始数据挂到 prompt dict 上。
- `BaseMultiModalProcessor`：vLLM 的模型级多模态处理器，不要和 HF 的
  `transformers.ProcessorMixin` 混淆。它内部通常会调用 HF processor 生成 image/audio/video tensor，
  再计算 placeholder、hash、prompt update 等 vLLM 需要的元数据。

### Chat request 渲染调用栈

下面这条栈发生在 `engine_client.generate()` 之前，产物是 `EngineInput`，后续才会交给
`AsyncLLM.add_request()` 转成 `EngineCoreRequest`。

```
OpenAIServingChat.create_chat_completion()
  │
  ├─ tokenizer / chat_template_kwargs / reasoning_parser 初始化
  │
  ├─ render_chat_request(request)
  │    │
  │    ├─ OpenAIServingChat.render_chat_request()
  │    │    ├─ _check_model(request)
  │    │    ├─ if engine_client.errored: raise dead_error
  │    │    └─ openai_serving_render.render_chat(request)
  │    │
  │    └─ OpenAIServingRender.render_chat(request)
  │         ├─ Mistral / harmony / tool_choice 等 OpenAI 兼容校验
  │         ├─ validate_chat_template(...)
  │         └─ preprocess_chat(...)
  │              ├─ 构造 ChatParams
  │              │    ├─ chat_template
  │              │    ├─ chat_template_content_format
  │              │    ├─ tools / documents / reasoning parser kwargs
  │              │    ├─ mm_processor_kwargs
  │              │    └─ media_io_kwargs
  │              │
  │              └─ renderer.render_chat_async([messages], chat_params, ...)
  │                   │
  │                   ├─ HfRenderer.render_messages_async(messages, chat_params)
  │                   │    ├─ resolve_chat_template_content_format(...)
  │                   │    ├─ parse_chat_messages_async(...)
  │                   │    │    ├─ 规范化 OpenAI message content
  │                   │    │    ├─ 下载/读取/解析 image/audio/video parts
  │                   │    │    ├─ 生成 conversation
  │                   │    │    ├─ 收集 multi_modal_data
  │                   │    │    └─ 收集 multi_modal_uuids
  │                   │    ├─ safe_apply_chat_template(...)
  │                   │    │    └─ HF tokenizer.apply_chat_template / Jinja
  │                   │    ├─ parse_dec_only_prompt(prompt_raw)
  │                   │    └─ prompt["multi_modal_data"] = mm_data
  │                   │
  │                   ├─ BaseRenderer.tokenize_prompts(...)
  │                   │    └─ _tokenize_singleton_prompt(...)
  │                   │         └─ tokenizer.encode(...)，除非已有 prompt_token_ids
  │                   │
  │                   ├─ BaseRenderer._apply_prompt_extras(...)
  │                   │    └─ 写入 cache_salt 等附加字段
  │                   │
  │                   └─ BaseRenderer.process_for_engine_async(...)
  │                        ├─ _process_singleton_async(...)
  │                        └─ _process_tokens_async(...)
  │                             ├─ 无 multi_modal_data:
  │                             │    └─ tokens_input(prompt_token_ids)
  │                             └─ 有 multi_modal_data:
  │                                  └─ _process_multimodal_async(...)
  │
  ├─ 得到 conversation, engine_inputs
  ├─ request.to_sampling_params(...)
  └─ engine_client.generate(engine_input, sampling_params, request_id, ...)
```

### Renderer

**文件**: `vllm/renderers/base.py`

`render_chat()` (行 969) 执行四步：

```
render_chat()
  ├── render_messages()        → 应用 Jinja2 chat template，生成文本
  ├── tokenize_prompts()       → 文本 → token IDs
  ├── _apply_prompt_extras()   → 写入 cache_salt 等附加字段
  └── process_for_engine()     → 处理多模态数据（如果有）并生成 EngineInput
```

### 分词

**文件**: `vllm/renderers/base.py`

`BaseRenderer._tokenize_prompt()` (行 410) 使用 HuggingFace tokenizer 将文本转为 token ID 列表。
`HfRenderer.render_messages()` (行 622) 负责解析 chat messages、应用 `safe_apply_chat_template()`，
再把结果交给 base renderer 的分词流程。

支持异步并发分词：`AsyncMicrobatchTokenizer`（`vllm/utils/async_utils.py`）用于大批量场景。

### 输出

渲染中间态和最终态要分开看：

- `HfRenderer.render_messages_async()` 返回的是 prompt dict：包含 `prompt` / `prompt_token_ids`
  以及原始 `multi_modal_data`、`multi_modal_uuids`。
- `BaseRenderer.process_for_engine_async()` 返回的是最终 `EngineInput`：纯文本请求会是
  `tokens_input(prompt_token_ids)`；多模态请求会是 `MultiModalInput`。

最终 `EngineInput` 常见字段包括：
- `prompt_token_ids`: `[int]` — 分词后的 token ID 列表
- `type`: `token` / `multimodal` / `embeds` 等 engine input 类型
- `mm_kwargs` / `mm_hashes` / `mm_placeholders`（多模态请求）
- `arrival_time`、`cache_salt` 等附加字段

---

## 4. Stage 3：多模态处理

仅在请求包含图片/音频/视频时执行。

**文件**: `vllm/renderers/base.py`

多模态有两段：先在 `HfRenderer.render_messages_async()` 里从 OpenAI messages 中解析出原始
`multi_modal_data`；然后在 `BaseRenderer._process_multimodal()` 里调用模型级
`BaseMultiModalProcessor.apply()`，产出真正喂给引擎的 `MultiModalInput`。

### 多模态调用栈

```
OpenAIServingRender.preprocess_chat()
  │
  └─ renderer.render_chat_async(...)
       │
       ├─ HfRenderer.render_messages_async(...)
       │    │
       │    ├─ parse_chat_messages_async(...)
       │    │    ├─ 遍历 request.messages
       │    │    ├─ 按 content_format 解析 text / image_url / audio / video
       │    │    ├─ media_io_kwargs 控制媒体读取/下载参数
       │    │    ├─ mm_processor_kwargs 收集传给 processor 的模型相关参数
       │    │    ├─ 输出 conversation
       │    │    ├─ 输出 mm_data: MultiModalDataDict
       │    │    └─ 输出 mm_uuids: MultiModalUUIDDict | None
       │    │
       │    ├─ safe_apply_chat_template(...)
       │    │    └─ 把 conversation 渲染成 prompt_raw
       │    │
       │    └─ prompt dict
       │         ├─ prompt / prompt_token_ids
       │         ├─ multi_modal_data = mm_data
       │         └─ multi_modal_uuids = mm_uuids
       │
       ├─ BaseRenderer.tokenize_prompts(...)
       │    └─ 如果 prompt 仍是文本，用 tokenizer.encode 得到 prompt_token_ids
       │
       └─ BaseRenderer.process_for_engine_async(...)
            │
            └─ _process_tokens_async(prompt)
                 │
                 └─ _process_multimodal_async(
                       prompt_token_ids,
                       multi_modal_data,
                       mm_processor_kwargs,
                       multi_modal_uuids,
                       skip_mm_cache,
                    )
                      │
                      ├─ 选择 mm_processor
                      │    ├─ 正常 generate: self.mm_processor
                      │    └─ render endpoint / skip_mm_cache: readonly processor
                      │
                      ├─ mm_processor.info.parse_mm_data(mm_data)
                      │    └─ 原始 image/audio/video → MultiModalDataItems
                      │
                      ├─ parse_mm_uuids(mm_uuids)
                      ├─ _process_mm_uuids(...)
                      │    └─ 校验或生成每个多模态 item 的 uuid
                      │
                      ├─ ProcessorInputs(...)
                      │    ├─ prompt: token ids 或文本
                      │    ├─ mm_data_items
                      │    ├─ mm_uuid_items
                      │    ├─ hf_processor_mm_kwargs
                      │    └─ tokenization_kwargs
                      │
                      └─ BaseMultiModalProcessor.apply(processor_inputs)
                           │
                           ├─ _cached_apply_hf_processor(...)
                           │    ├─ cache 命中: 复用 mm_kwargs / prompt updates
                           │    └─ cache miss:
                           │         └─ _apply_hf_processor(...)
                           │              ├─ _apply_hf_processor_main(...)
                           │              │    ├─ 如果 prompt 是文本:
                           │              │    │    └─ _apply_hf_processor_text_mm(...)
                           │              │    │         ├─ _get_hf_mm_data(...)
                           │              │    │         └─ _call_hf_processor(...)
                           │              │    │              └─ HF Processor(...)
                           │              │    │                   # transformers.ProcessorMixin
                           │              │    │                   # text + media → input_ids + media tensors
                           │              │    │
                           │              │    └─ 如果 prompt 已经是 token ids:
                           │              │         ├─ _apply_hf_processor_tokens_only(...)
                           │              │         └─ _apply_hf_processor_mm_only(...)
                           │              │              └─ HF Processor(...)
                           │              │                   # media → pixel_values/input_features/...
                           │              │
                           │              ├─ 返回 prompt_ids + BatchFeature
                           │              ├─ MultiModalKwargsItems.from_hf_inputs(...)
                           │              ├─ inputs.get_mm_hashes(...)
                           │              └─ _get_mm_prompt_updates(...)
                           │
                           ├─ _maybe_apply_prompt_updates(...)
                           │    ├─ HF processor 已更新 prompt:
                           │    │    └─ _find_mm_placeholders(prompt_ids, updates)
                           │    └─ HF processor 未更新 prompt:
                           │         └─ _apply_prompt_updates(...)
                           │              ├─ 查找/替换 image/audio/video placeholder
                           │              └─ 重新定位 placeholder ranges
                           │
                           └─ mm_input(...)
                                ├─ prompt_token_ids
                                ├─ mm_kwargs
                                ├─ mm_hashes
                                └─ mm_placeholders
```

### _process_multimodal()

**文件**: `vllm/renderers/base.py`

`_process_multimodal()` (行 675) 的核心可以简化为：

```
_process_multimodal()
  ├── 选择 mm_processor / readonly_mm_processor
  ├── mm_processor.info.parse_mm_data(mm_data)
  ├── parse_mm_uuids(mm_uuids)
  ├── _process_mm_uuids()
  ├── 构造 ProcessorInputs
  ├── mm_processor.apply()
  │     ├── 调 HF processor 处理图片/音频/视频
  │     ├── 生成或校验 placeholder token IDs
  │     ├── 计算 mm_hashes，用于缓存和跨进程复用
  │     └── 产生 mm_kwargs + mm_placeholders
  └── 返回 MultiModalInput
```

`_process_multimodal()` 返回的是 engine input 里的多模态字段；之后
`InputProcessor.process_inputs()` 会在行 322-358 将占位符、hash 和 processor 输出
整理成 `MultiModalFeatureSpec` 列表。`MultiModalFeatureSpec` 包含：
- 每个多模态项的 embedding 张量
- 位置信息（offset + length）
- 对应的 token 位置范围

这些 features 会在 Stage 7 被注入到模型中（通过 vision encoder → embedding 替换）。

### vLLM MultiModalProcessor 和 HF Processor 的分工

```
BaseMultiModalProcessor.apply()
  │
  ├─ vLLM 负责：
  │    ├─ 解析 MultiModalDataItems
  │    ├─ 处理 processor cache
  │    ├─ 计算 mm_hashes
  │    ├─ 定义每种模型的 prompt updates / placeholder 规则
  │    ├─ 把 HF BatchFeature 转成 MultiModalKwargsItems
  │    └─ 返回 vLLM EngineInput 需要的 mm_kwargs / mm_placeholders
  │
  └─ HF Processor 负责：
       ├─ 图像 resize / normalize / patch 化
       ├─ 音频 feature extraction
       ├─ 视频 frame / vision feature 前处理
       ├─ 某些模型的 text + media 联合 tokenization
       └─ 返回 BatchFeature，例如 pixel_values / image_grid_thw / input_features
```

所以可以把 HF processor 理解为“模型家族的媒体前处理实现”，而 vLLM 的
`BaseMultiModalProcessor` 是“把 HF 结果接入 vLLM 调度和 KV/placeholder 体系的适配层”。

---

## 5. Stage 4：提交到引擎

### 先区分三个名字

OpenAI serving 层里变量名通常叫 `engine_client`，类型是
`vllm/engine/protocol.py` 里的 `EngineClient` 协议。V1 异步服务时，这个对象通常就是
`vllm/v1/engine/async_llm.py` 里的 `AsyncLLM`。

`AsyncLLM` 不是直接跑 scheduler / GPU 的对象。它在 API Server 进程里做三件事：

1. 把渲染后的输入转成 `EngineCoreRequest`。
2. 为每个 request 注册一个前端状态和一个 per-request 输出收集器。
3. 通过内部的 `EngineCoreClient` 把请求送到 EngineCore，并在后台把 EngineCore 的输出搬回 per-request 队列。

`EngineCoreClient` 是 `AsyncLLM` 内部持有的传输层对象。异步多进程路径下实际类是
`AsyncMPClient`，它负责 ZMQ、msgpack、utility RPC、DP 选路、engine 死亡检测等；它本身不做
token 级调度，也不执行模型。

`EngineCore` / `EngineCoreProc` 才是 scheduler 和 model executor 所在的核心。多进程服务时，
`EngineCoreProc` 在独立进程中运行 busy loop，并额外启动输入/输出 socket 线程，把 ZMQ IO
和主调度循环解耦。

### 提交与回收的边界图

Stage 4 只负责把 API 侧的 `EngineInput` 变成 EngineCore 可接收的请求，并建立返回通道。
EngineCore 内部如何调度和执行，放到 Stage 5-9 展开；输出如何回到 HTTP client，放到 Stage 10-11 展开。

```
OpenAI serving task
  │
  │ engine_client.generate(...)
  ▼
AsyncLLM.generate()
  │
  ├─ add_request()
  │    ├─ InputProcessor.process_inputs()
  │    │    └─ EngineInput / PromptType → EngineCoreRequest
  │    ├─ OutputProcessor.add_request()
  │    │    └─ request_id → RequestState + RequestOutputCollector
  │    └─ EngineCoreClient.add_request_async()
  │         └─ ZMQ ROUTER send: [engine_identity, request_type=ADD, msgpack(request)]
  │
  ├─ generate() 自己不读 ZMQ
  │    └─ 等待这个 request 的 RequestOutputCollector
  │
  └─ yield RequestOutput 给 serving 层

AsyncLLM output_handler task      EngineCoreClient / AsyncMPClient
  │                                      │
  ├─ await engine_core.get_output_async()│
  │<─────────────────────────────────────│ outputs_queue.get()
  │                                      │   ↑
  ├─ OutputProcessor.process_outputs()   │   │ ZMQ PULL recv EngineCoreOutputs
  │    └─ collector.put(RequestOutput)   │   │
  │                                      │   │
  └─ 必要时 abort stop-string 命中请求     │   │
```

注意这里有两条并行的前端路径：

- `generate()` 所在的请求 task：负责提交请求，然后只消费这个 request 自己的
  `RequestOutputCollector`。
- `_run_output_handler()` 创建的后台 task：负责消费所有 EngineCore 返回的 batch 输出，
  再按 `request_id` 分发到对应 collector。

所以 `generate()` 不会直接从 ZMQ socket 读 EngineCore 输出；它读的是已经经过
`OutputProcessor` 处理后的 per-request collector。

### AsyncLLM.generate()

**文件**: `vllm/v1/engine/async_llm.py:521`

这是 API Server 和引擎之间的主入口：

```python
async def generate(self, prompt, sampling_params, request_id, ...) -> AsyncGenerator[RequestOutput]:
    # 1. 处理输入、注册请求、发送到 EngineCore
    q = await self.add_request(request_id, prompt, sampling_params, ...)

    # 2. 持续从 per-request 队列拉取输出
    while not finished:
        out = q.get_nowait() or await q.get()
        finished = out.finished
        yield out    # ← 返回给 API Server
```

### add_request()

**文件**: `async_llm.py:280`

```
add_request()
  ├── InputProcessor.process_inputs()     → 验证参数，创建 EngineCoreRequest
  │     (input_processor.py:234)
  │
  ├── RequestOutputCollector(...)         → 当前 generate() 的 per-request 输出收集器
  │     (output_processor.py:45)
  │
  ├── OutputProcessor.add_request()       → 注册 RequestState + Detokenizer + collector
  │     (output_processor.py:508)
  │
  └── _add_request()
        ├── 注册到 OutputProcessor
        └── engine_core.add_request_async()  → ZMQ 发送
              (core_client.py:1058)
```

### IPC

**文件**: `vllm/v1/engine/core_client.py`

`AsyncMPClient.add_request_async()` (行 1058) 会先写入 `request.client_index`，再把
`EngineCoreRequest` 通过 msgpack 序列化，经 ZMQ ROUTER socket 发送到 EngineCore 进程。
消息帧形态可以理解为：

```
[engine_identity, EngineCoreRequestType.ADD, msgpack_frame_0, msgpack_frame_1, ...]
```

`engine_identity` 用于把请求送到具体的 EngineCore。单 DP 时通常就是第一个 core；
内部 DP load balancing 时，`DPLBAsyncMPClient.get_core_engine_for_request()` 会根据
各 EngineCore 的 waiting/running 计数选择较空的 core，并记录 `req_id → engine`，方便
后续 abort 精确路由。

如果请求里带有多模态 tensor 或其他可零拷贝 buffer，`AsyncMPClient` 会把 Python 对象暂存在
`pending_messages`，直到 ZMQ 的 `MessageTracker` 表示底层 buffer 已经发送完成，避免 tensor
被过早释放。

**关键点**：`add_request_async()` 只负责把请求发给 EngineCore 并确保 output queue task 已启动；
真正的调度、KV cache 分配、模型执行都在 EngineCore / Worker 侧。`generate()` 本身是 async
generator，后续从 per-request collector 拉取输出并逐步 `yield`。

### 这条链路里有哪些队列

| 位置 | 队列 / 缓冲 | 生产者 | 消费者 | 作用 |
|------|-------------|--------|--------|------|
| API 进程 / `AsyncLLM` | `RequestOutputCollector` | `OutputProcessor.process_outputs()` | 当前 request 的 `generate()` 循环 | 单个 request 的输出交接；DELTA 模式下如果生产快于消费，会聚合输出 |
| API 进程 / `AsyncMPClient` | `outputs_queue: asyncio.Queue` | `process_outputs_socket()` ZMQ 后台 task | `AsyncLLM.output_handler` | 所有 EngineCore batch 输出的进程内异步缓冲 |
| API → EngineCore | ZMQ `ROUTER -> DEALER` | `AsyncMPClient._send_input_message()` | `EngineCoreProc.process_input_sockets()` | 发送 ADD / ABORT / UTILITY 等请求 |
| EngineCore 进程 | `input_queue: queue.Queue` | `process_input_sockets()` 输入线程 | `run_busy_loop()` 主线程 | 把 socket IO 和调度循环解耦；主线程从这里取 ADD/ABORT/UTILITY |
| EngineCore 进程 | `aborts_queue: queue.Queue` | 输入线程收到 ABORT 时额外写入 | `step()` 后的 `_process_aborts_queue()` | 模型执行期间也能尽早记录 abort，执行结束后批量处理 |
| EngineCore 进程 | `output_queue: queue.Queue` | `run_busy_loop()` 主线程 | `process_output_sockets()` 输出线程 | 把 scheduler 产生的 `EngineCoreOutputs` 交给 ZMQ 输出线程 |
| EngineCore → API | ZMQ `PUSH -> PULL` | `process_output_sockets()` | `AsyncMPClient.process_outputs_socket()` | 返回 `EngineCoreOutputs`，包括 token 输出、scheduler stats、utility result |

这意味着一个 request 的输出实际经过两次“批到单”的拆分：

1. EngineCore 的 `Scheduler.update_from_output()` 按 `client_index` 组装
   `dict[int, EngineCoreOutputs]`，保证多 API frontend 时输出回到提交它的 client。
2. API 进程的 `OutputProcessor.process_outputs()` 再按 `request_id` 找到
   `RequestState.queue`，把 batch 中每个 `EngineCoreOutput` 投递给对应的
   `RequestOutputCollector`。

### 三个循环

1. `AsyncLLM.generate()` 的 per-request 循环：提交请求后执行
   `q.get_nowait() or await q.get()`，每拿到一个 `RequestOutput` 就 `yield` 给 serving 层；
   直到 `out.finished=True`。
2. `AsyncLLM._run_output_handler()` 的全局后台循环：不断
   `await engine_core.get_output_async()`，处理来自 EngineCore 的 batch 输出，并投递到各个
   per-request collector。
3. `EngineCoreProc.run_busy_loop()` 的核心循环：先在空闲时阻塞等 `input_queue`，有请求后 drain
   当前输入，再执行一次 `step()`，把这次 iteration 的输出放入 `output_queue`；只要 scheduler
   还有 unfinished requests，就会继续 step。

这三个循环不是嵌套调用关系，而是通过队列和 ZMQ 解耦后并发运行。一个 HTTP request task 可以在
等待自己的 collector；与此同时 output handler 正在处理其他 request 的输出，EngineCore 主循环
也在持续调度 batch。

### Utility / Abort 不是普通生成请求

`EngineCoreRequestType` 除了 `ADD`，还有：

- `ABORT`：客户端断连、超时或前端 stop string 命中时发送。`AsyncLLM.abort()` 会先清理
  `OutputProcessor` 里的前端状态，再把内部 request id 发送给 EngineCore。
- `UTILITY`：profile、reset cache、sleep/wake、LoRA 管理、collective RPC 等控制面调用。
  这类请求通过同一条输入 ZMQ 到 EngineCore，但返回的是 `EngineCoreOutputs.utility_output`，
  `AsyncMPClient.process_outputs_socket()` 会用 `call_id` 找到等待中的 future，而不是放进普通
  request 输出队列。

---

## 6. Stage 5：调度排队

### EngineCore 接收

**文件**: `vllm/v1/engine/core.py`

```
EngineCoreProc.__init__()
  ├── input_queue = queue.Queue()
  ├── output_queue = queue.Queue()
  ├── 启动 process_input_sockets() 线程   → ZMQ recv / decode / preprocess
  └── 启动 process_output_sockets() 线程  → output_queue / encode / ZMQ send

process_input_sockets()
  ├── DEALER socket 从前端 ROUTER 收到 ADD / ABORT / UTILITY
  ├── ADD:
  │     ├── msgpack decode EngineCoreRequest
  │     ├── preprocess_add_request()
  │     │     ├── mm tensor IPC 还原多模态 features
  │     │     ├── Request.from_engine_core_request()
  │     │     └── structured output grammar_init()
  │     └── input_queue.put((ADD, (Request, request_wave)))
  ├── ABORT:
  │     ├── generic decode request_ids
  │     ├── aborts_queue.put(request_ids)   # 给 step 后的 eager abort 处理
  │     └── input_queue.put((ABORT, request_ids))
  └── UTILITY:
        └── input_queue.put((UTILITY, (...)))

EngineCoreProc.run_busy_loop() (行 1164)
  ├── _process_input_queue()
  │     ├── 空闲时阻塞等待 input_queue
  │     ├── 有 work 后 drain 当前 input_queue
  │     └── _handle_client_request()
  │           ├── ADD   → Scheduler.add_request() → waiting 队列
  │           ├── ABORT → Scheduler.finish_requests(...)
  │           └── UTILITY → 调 EngineCore 方法并把 UtilityOutput 放入 output_queue
  │
  └── _process_engine_step()
        └── EngineCore.step() (行 402)  → 见下方
```

`process_input_sockets()` 和 `run_busy_loop()` 分开是为了让 ZMQ recv、msgpack decode、
多模态 tensor IPC、grammar 初始化等输入侧工作能和 GPU 前向有重叠。主循环不直接读 socket，
它只读进程内 `input_queue`。

`_process_input_queue()` 的阻塞策略也很关键：如果当前没有任何 running/waiting/batch-queue work，
它会阻塞等待新输入；一旦 scheduler 已经有未完成请求，它就不长期阻塞，而是在每个 step 前尽量
drain 已到达的输入，然后继续执行下一轮模型 iteration。

### Scheduler 调度

**文件**: `vllm/v1/core/sched/scheduler.py`

`Scheduler.schedule()` (行 352) — 每个 iteration 执行：

```
schedule()
  │
  ├── Phase 1: 调度 RUNNING 请求 (行 387)
  │     ├── 计算 num_new_tokens = num_tokens_with_spec - num_computed_tokens
  │     ├── 长预填充分块（long_prefill_token_threshold）
  │     ├── 分配 KV cache blocks
  │     └── 内存不足时 → 抢占低优先级请求
  │
  ├── Phase 2: 调度 WAITING 请求 (行 567)
  │     ├── 从 waiting 队列取出请求
  │     ├── 检查 prefix cache 命中
  │     ├── 准入控制（can_fit_full_sequence）
  │     ├── 分配 KV cache blocks
  │     └── 移入 running 列表
  │
  └── 返回 SchedulerOutput
        ├── num_scheduled_tokens: {req_id: int}
        ├── scheduled_new_reqs / scheduled_cached_reqs
        ├── block_ids 映射
        └── finished_req_ids
```

**新的请求就在 Phase 2 中被从 waiting 队列取出，分配 KV cache，进入 running 列表。**

### AsyncLLM → EngineCore → GPUWorker 单轮执行时序图

这一段关注 request 已经提交到 EngineCore 之后，每个 engine iteration 如何推进。`AsyncLLM`
不直接调用 GPUWorker；它只通过 `EngineCoreClient` 发送请求、接收输出。真正的 worker 调用发生在
EngineCore 进程里的 `model_executor.execute_model()`。

```
AsyncLLM              AsyncMPClient             EngineCoreProc            Scheduler              ModelExecutor / GPUWorker
  │                         │                         │                         │                            │
  │ add_request()           │                         │                         │                            │
  │ ├ OutputProcessor.add   │                         │                         │                            │
  │ └ add_request_async()   │                         │                         │                            │
  │────────────────────────>│                         │                         │                            │
  │                         │ ZMQ ROUTER send ADD     │                         │                            │
  │                         │────────────────────────>│ input socket thread     │                            │
  │                         │                         │ ├ decode request        │                            │
  │                         │                         │ ├ preprocess_add_request│                            │
  │                         │                         │ └ input_queue.put(ADD)  │                            │
  │                         │                         │                         │                            │
  │                         │                         │ run_busy_loop()         │                            │
  │                         │                         │ ├ input_queue.get()     │                            │
  │                         │                         │ └ _handle_client_request│                            │
  │                         │                         │────────────────────────>│ add_request()              │
  │                         │                         │                         │ └ waiting.push(request)    │
  │                         │                         │                         │                            │
  │                         │                         │ step()                  │                            │
  │                         │                         │────────────────────────>│ schedule()                 │
  │                         │                         │                         │ ├ running decode requests  │
  │                         │                         │                         │ ├ waiting prefill requests │
  │                         │                         │                         │ ├ allocate KV blocks       │
  │                         │                         │                         │ └ SchedulerOutput          │
  │                         │                         │<────────────────────────│                            │
  │                         │                         │                         │                            │
  │                         │                         │ execute_model(output)   │                            │
  │                         │                         │───────────────────────────────────────────────────────>│
  │                         │                         │                         │                            │ GPUWorker.execute_model()
  │                         │                         │                         │                            │ ├ _update_states()
  │                         │                         │                         │                            │ ├ prepare input tensors
  │                         │                         │                         │                            │ ├ model forward
  │                         │                         │                         │                            │ └ stash logits/hidden states
  │                         │                         │<───────────────────────────────────────────────────────│
  │                         │                         │ future.result()         │                            │
  │                         │                         │                         │                            │
  │                         │                         │ get_grammar_bitmask()   │                            │
  │                         │                         │────────────────────────>│ structured output mask      │
  │                         │                         │<────────────────────────│                            │
  │                         │                         │                         │                            │
  │                         │                         │ sample_tokens(mask)     │                            │
  │                         │                         │───────────────────────────────────────────────────────>│
  │                         │                         │                         │                            │ ├ apply grammar mask
  │                         │                         │                         │                            │ ├ logits processors
  │                         │                         │                         │                            │ └ sample token ids
  │                         │                         │<───────────────────────────────────────────────────────│
  │                         │                         │ ModelRunnerOutput       │                            │
  │                         │                         │                         │                            │
  │                         │                         │ update_from_output()    │                            │
  │                         │                         │────────────────────────>│ append tokens / stop check  │
  │                         │                         │                         │ free finished KV blocks     │
  │                         │                         │                         │ build EngineCoreOutputs     │
  │                         │                         │<────────────────────────│                            │
  │                         │                         │ output_queue.put(...)   │                            │
  │                         │                         │ output socket thread    │                            │
  │                         │<────────────────────────│ ZMQ PUSH EngineCoreOutputs                            │
  │ outputs_queue.put(...)  │                         │                         │                            │
  │<────────────────────────│                         │                         │                            │
  │ output_handler          │                         │                         │                            │
  │ ├ process_outputs()     │                         │                         │                            │
  │ └ collector.put(output) │                         │                         │                            │
```

这一轮结束后，如果 request 没有 finished，它仍在 Scheduler 的 running 集合里。下一轮
`schedule()` 会在 Phase 1 继续调度它，通常 decode 阶段每轮只追加 1 个新 token；如果启用了
speculative decoding，则一轮可能同时验证多个 draft token。

这里的关键边界是：

- `AsyncLLM ↔ EngineCoreProc`：跨进程 ZMQ + msgpack，传的是 `EngineCoreRequest` /
  `EngineCoreOutputs`。
- `EngineCoreProc ↔ Scheduler`：同进程 Python 调用，传的是内部 `Request`、`SchedulerOutput`。
- `EngineCoreProc ↔ GPUWorker`：通过 `model_executor` 调用 executor/worker，传的是
  `SchedulerOutput`，worker 侧再构建 GPU tensor、执行 forward 和采样。

### EngineCore 输出返回

`EngineCore.step()` 返回的是 `dict[int, EngineCoreOutputs]`：key 是 `client_index`，
value 是要发回该 frontend 的一批输出。`EngineCoreProc._process_engine_step()` 会把每个
`(client_index, EngineCoreOutputs)` 放入 `output_queue`；输出线程
`process_output_sockets()` 再把它编码成 msgpack，通过对应的 ZMQ PUSH socket 发回 API 进程。

`EngineCoreOutputs` 里可能包含三类信息：

- `outputs`: token / pooling 维度的 `EngineCoreOutput` 列表，后续由前端 `OutputProcessor`
  转成 `RequestOutput`。
- `scheduler_stats`: scheduler 统计信息，前端 output handler 用于指标记录。
- `utility_output`: 控制面 RPC 的结果，`AsyncMPClient` 会用 `call_id` 唤醒对应 future，不进入
  普通 request 输出路径。

---

## 7. Stage 6：Worker 输入准备

### Worker / ModelRunner / Model 的边界

官方架构页把 GPU 执行侧拆成三层：

```
GPU Worker process
  │
  ├─ Worker
  │    ├─ 绑定 rank / local_rank
  │    ├─ 管理一个 accelerator device
  │    ├─ 初始化分布式通信和 GPU 内存
  │    └─ 持有一个 model runner
  │
  ├─ ModelRunner
  │    ├─ 加载模型
  │    ├─ 根据 SchedulerOutput 准备 GPU tensors
  │    ├─ 管理 CUDA Graph / attention metadata / InputBatch
  │    ├─ 调用 model forward
  │    └─ 执行 logits 处理与采样
  │
  └─ Model
       └─ 真正的 torch.nn.Module：embedding、attention、MLP/MoE、LM head 等
```

`rank` 用于全局分布式编排；`local_rank` 主要用于选择本机 accelerator 和访问本地资源。
从 request 角度看，EngineCore 不直接调用 `torch.nn.Module`，而是把 `SchedulerOutput`
交给 executor/worker；worker 再委托 model runner 生成 GPU 输入并调用 model。

### execute_model()

默认路径（`VLLM_USE_V2_MODEL_RUNNER=0`）：

**文件**: `vllm/v1/worker/gpu_model_runner.py`

可选 V2 路径（`VLLM_USE_V2_MODEL_RUNNER=1`）：

**文件**: `vllm/v1/worker/gpu/model_runner.py`

选择逻辑在 `vllm/v1/worker/gpu_worker.py:153` 和 `gpu_worker.py:295-310`。

`GPUModelRunner.execute_model()` 首先更新持久化批次状态：

```
_update_states(scheduler_output)        # 默认路径：gpu_model_runner.py:1061
  ├── 移除 finished_req_ids 对应的 cached state / InputBatch 条目
  ├── 释放 encoder cache、处理需要清零的新 KV cache blocks
  ├── 写入 scheduled_new_reqs / resumed / running 的请求状态
  └── 更新 persistent InputBatch 的 block table、采样元数据和 LoRA 状态
```

### _prepare_inputs()

默认路径中是 `_prepare_inputs()` (行 1776) 构建 GPU 输入张量：

```
_prepare_inputs()
  ├── 排序请求（decode 在前，prefill 在后）
  ├── 构建 idx_mapping（request → batch index）
  ├── 计算 query_start_loc（attention 的 cumsum）
  ├── [Prefill] Triton kernel 填充 input_ids
  ├── [Decode] 合并上次采样 token + draft tokens
  ├── Triton kernel 计算 positions 和 seq_lens
  └── 返回 InputBatch（所有 GPU tensor 就绪）
```

V2 runner 中对应方法是 `vllm/v1/worker/gpu/model_runner.py:697` 的
`prepare_inputs()`，结构更模块化，`InputBatch` 也位于
`vllm/v1/worker/gpu/input_batch.py:36`。

### Attention Metadata

```
默认 runner:
  ├── 从 InputBatch / block_table 读取 block tables
  ├── 计算 slot mappings、query_start_loc、seq_lens 等
  └── 构建 attention metadata（FlashAttention/FlashInfer 等后端格式）

V2 runner:
  └── model_state.prepare_attn() 负责同类信息的模块化准备
```

---

## 8. Stage 7：模型前向推理

**文件**: `vllm/v1/worker/gpu_model_runner.py`

```
execute_model() 继续
  │
  ├── [FULL CUDA Graph] → cudagraph_manager.run_fullgraph() → graph.replay()
  │
  ├── [Eager / PIECEWISE] → model(**model_inputs)
  │     │
  │     ├── Embedding lookup（input_ids → hidden_states）
  │     ├── [多模态] 注入 vision/audio embeddings 到对应位置
  │     │
  │     ├── N × TransformerLayer:
  │     │     ├── RMSNorm
  │     │     ├── Self-Attention（PagedAttention，读写 KV cache）
  │     │     ├── RMSNorm
  │     │     └── FFN / MoE
  │     │
  │     └── Final RMSNorm → Linear → logits（或 hidden_states）
  │
  └── 存储 hidden_states 和 logits 到 execute_model_state
```

**关键**：`execute_model()` 执行模型前向并准备 hidden_states / logits 等中间状态，但不做最终采样；
grammar bitmask 需要在 logits 产生之后、采样之前应用，因此采样在后续 `sample_tokens()` 中进行。

---

## 9. Stage 8：采样

**默认文件**: `vllm/v1/worker/gpu_model_runner.py:4140`

```
EngineCore.step() 中:
  execute_model() → 前向推理并暂存 hidden_states / logits（不采样）
  get_grammar_bitmask() → 生成结构化输出 bitmask
  sample_tokens(grammar_output) → 实际采样
```

### sample_tokens()

```
sample_tokens(grammar_output) (行 4140)
  │
  ├── 从 execute_model_state 取出 logits
  │
  ├── [结构化输出] apply_grammar_bitmask() → 将不允许的 token 设为 -inf
  │     (v1/structured_output/utils.py:44)
  │
  ├── _sample(logits, ...) → 调用 Sampler
  │     (gpu_model_runner.py:3329 → v1/sample/sampler.py:68)
  │     │
  │     ├── 默认 Sampler.forward() / apply_logits_processors():
  │     │     ├── logit_bias / allowed_token_ids
  │     │     ├── frequency / presence / repetition penalty
  │     │     ├── bad words masking
  │     │     ├── temperature 缩放
  │     │     ├── min-p / top-k / top-p 过滤
  │     │     └── 返回处理后的 logits
  │     │
  │     └── top-k/top-p sampler 或 greedy path
  │           （V2 runner 中为 v1/worker/gpu/sample/sampler.py:57，
  │            参数处理入口为 apply_sampling_params()，并使用 gumbel_sample）
  │
  ├── _update_states_after_model_execute() → 更新 GPU 侧请求状态
  │
  └── [Speculative Decoding] propose_draft_token_ids()
        → 用 draft model 生成候选 token（如果启用）
```

---

## 10. Stage 9：调度器更新状态

### update_from_output()

**文件**: `vllm/v1/core/sched/scheduler.py:1303`

`EngineCore.step()` 拿到 model output 后调用：

```
update_from_output(scheduler_output, model_output) (行 1303)
  │
  ├── 遍历所有被调度的请求：
  │     │
  │     ├── [Spec Decoding] 计算被拒绝的 draft tokens，回退 num_computed_tokens
  │     │
  │     ├── _update_request_with_output() (行 1635)
  │     │     ├── request.append_output_token_ids(token)
  │     │     └── check_stop() → 检查 EOS / max_tokens / stop tokens
  │     │           (sched/utils.py:94)
  │     │
  │     ├── [结构化输出] grammar.accept_tokens() 推进状态机
  │     │
  │     └── 如果 stopped:
  │           ├── _handle_stopped_request() (行 1592)
  │           │     └── 释放 KV cache、移出 running 列表
  │           └── 创建 EngineCoreOutput
  │                 ├── new_token_ids: [int]
  │                 ├── finish_reason
  │                 ├── logprobs
  │                 └── prefill_stats
  │
  └── 返回 EngineCoreOutputs → 放入 output queue → ZMQ 发回 API 进程
```

**如果请求未完成**，下一轮 `schedule()` 的 Phase 1 会继续处理它（此时 num_computed_tokens 已增加，只需计算 1 个新 decode token）。

---

## 11. Stage 10：输出处理与反分词

### OutputProcessor

**文件**: `vllm/v1/engine/output_processor.py`

API Server 进程这里也分两层：

1. `AsyncMPClient.process_outputs_socket()` 是 ZMQ 层后台 task。它从 PULL socket 收到
   `EngineCoreOutputs`，decode 后放进 `AsyncMPClient.outputs_queue`。
2. `AsyncLLM._run_output_handler()` 是语义处理层后台 task。它调用
   `engine_core.get_output_async()`，实际就是从 `outputs_queue` 取出一个 batch，然后交给
   `OutputProcessor.process_outputs()`。

`OutputProcessor.process_outputs()` 的职责是把 EngineCore 的 token 级输出转换成 API 可消费的
`RequestOutput`：

```
process_outputs(engine_core_outputs) (行 572)
  │
  ├── 遍历每个 EngineCoreOutput：
  │     │
  │     ├── Detokenizer.update() (detokenizer.py:95)
  │     │     ├── 增量解码每个新 token ID → 文本
  │     │     │     （FastIncrementalDetokenizer 使用 Rust 实现的 DecodeStream）
  │     │     └── 检查 stop strings
  │     │
  │     ├── LogprobsProcessor.update_from_output()
  │     │     → 处理 sample logprobs 和 prompt logprobs
  │     │
  │     ├── RequestState.make_request_output() (行 269)
  │     │     ├── stream_interval 控制（每 N 个 token 发射一次输出）
  │     │     └── 创建 RequestOutput（text, token_ids, logprobs, finish_reason）
  │     │
  │     └── 放入 per-request 的 RequestOutputCollector 队列
  │
  └── 清理已完成的请求
```

`RequestOutputCollector` 不是标准 `asyncio.Queue`，而是一个“单槽 + asyncio.Event”的轻量收集器。
如果 producer 已经放了一个 `RequestOutput`，consumer 还没取走，DELTA 输出会通过
`RequestOutput.add(..., aggregate=True)` 合并，减少高负载下的 task 切换和队列堆积。`generate()`
侧每次 `get_nowait()` / `await get()` 拿到的是已经 detokenize、已经按 stream interval 处理好的
`RequestOutput`。

如果前端 detokenizer 发现 stop string，但 EngineCore 侧只是生成了普通 token、尚未知道这个
字符串级停止条件，`process_outputs()` 会把 request id 放入 `reqs_to_abort`。随后
`AsyncLLM.output_handler` 调用 `engine_core.abort_requests_async()`，通知 EngineCore 清理这个请求的
调度状态和 KV cache。

### EngineCore 输出 → Client 时序图

这张图对应上面 `EngineCoreProc.output_queue.put(...)` 之后的返回路径。它解释为什么
`AsyncLLM.generate()` 只需要等自己的 collector，而不用理解 EngineCore 的 batch 输出。

```
GPUWorker            Scheduler              EngineCoreProc             AsyncMPClient              AsyncLLM               API Server              Client
  │                       │                        │                         │                       │                       │                    │
  │ sampled token ids     │                        │                         │                       │                       │                    │
  │──────────────────────>│ update_from_output()   │                         │                       │                       │                    │
  │                       │ ├ append output token  │                         │                       │                       │                    │
  │                       │ ├ check EOS/max_tokens │                         │                       │                       │                    │
  │                       │ └ EngineCoreOutput     │                         │                       │                       │                    │
  │                       │───────────────────────>│                         │                       │                       │                    │
  │                       │                        │ output_queue.put()      │                       │                       │                    │
  │                       │                        │ output socket thread    │                       │                       │                    │
  │                       │                        │ ZMQ PUSH msgpack        │                       │                       │                    │
  │                       │                        │────────────────────────>│ process_outputs_socket│                       │                    │
  │                       │                        │                         │ ├ decode              │                       │                    │
  │                       │                        │                         │ └ outputs_queue.put   │                       │                    │
  │                       │                        │                         │──────────────────────>│ output_handler       │                    │
  │                       │                        │                         │                       │ ├ get_output_async() │                    │
  │                       │                        │                         │                       │ ├ process_outputs()  │                    │
  │                       │                        │                         │                       │ │ ├ detokenize       │                    │
  │                       │                        │                         │                       │ │ ├ logprobs         │                    │
  │                       │                        │                         │                       │ │ └ stop strings     │                    │
  │                       │                        │                         │                       │ └ collector.put()   │                    │
  │                       │                        │                         │                       │                    generate() loop       │
  │                       │                        │                         │                       │<─────────────────── q.get()              │
  │                       │                        │                         │                       │ yield RequestOutput │                    │
  │                       │                        │                         │                       │────────────────────>│ build SSE chunk     │
  │                       │                        │                         │                       │                     │───────────────────>│
  │                       │                        │                         │                       │                     │ data: {...}         │
```

如果 `RequestOutput.finished=False`，这条返回链路会在下一轮 token 生成时再次发生；如果
`finished=True`，`generate()` 循环结束，serving 层发送最终 usage / `[DONE]` 或完整 JSON。

### 反分词

**文件**: `vllm/v1/engine/detokenizer.py`

`IncrementalDetokenizer` 支持两种实现：
- `FastIncrementalDetokenizer` (行 167)：使用 `tokenizers` 库的 `DecodeStream`（Rust 实现），每次只解码一个 token
- `SlowIncrementalDetokenizer` (行 245)：Python 回退实现

增量解码避免了每步对全部历史 token 重新解码。

---

## 12. Stage 11：流式返回

### AsyncLLM.generate() 的输出循环

**文件**: `async_llm.py:570-583`

```python
# generate() 是一个 async generator
while not finished:
    out = q.get_nowait() or await q.get()   # 从 per-request 队列拉取
    finished = out.finished
    yield out   # → 返回给 API Server 的 serving 层
```

### SSE 流式响应

**文件**: `serving.py`

`chat_completion_stream_generator()` (行 525)：

```
for request_output in result_generator:    # result_generator = engine_client.generate()
    │
    ├── 构建 ChatCompletionStreamResponse
    │     ├── delta.content（增量文本）
    │     ├── delta.reasoning_content（推理内容，如 thinking mode）
    │     └── usage 信息
    │
    └── yield SSE chunk: "data: {json}\n\n"

最后 yield "data: [DONE]\n\n"
```

### 非流式响应

`chat_completion_full_generator()` (行 1273)：
- 收集所有 `RequestOutput`
- 构建一个完整的 `ChatCompletionResponse`
- 返回 JSON response

---

## 13. 横切关注点

### 13.1 结构化输出（Structured Output）

贯穿多个阶段：

```
Stage 1: ChatCompletionRequest.response_format → SamplingParams.structured_outputs
Stage 4: InputProcessor 创建 StructuredOutputRequest
Stage 5: Scheduler.add_request() → grammar_init() 异步编译 grammar
Stage 8: get_grammar_bitmask() → 填充 allowed/disallowed token bitmask
         apply_grammar_bitmask() → logits 中不允许的 token 设为 -inf
Stage 9: grammar.accept_tokens() → 推进 grammar 状态机
```

**文件**: `vllm/v1/structured_output/__init__.py`

支持的后端：xgrammar, guidance, outlines, lm-format-enforcer。

### 13.2 Speculative Decoding

```
Stage 8: sample_tokens() 结束后 → propose_draft_token_ids()
         draft model 生成 N 个候选 token
Stage 5: 下一步 schedule() 时，draft tokens 被计入 num_tokens_with_spec
Stage 7: 模型一次前向计算验证所有 draft tokens
Stage 9: update_from_output() 计算接受/拒绝的 draft tokens
         被拒绝的 tokens → 回退 num_computed_tokens
```

### 13.3 请求 Abort

随时可能发生（客户端断连、超时等）：
- API 进程：`generate()` 被 CancelledError 中断 → 调用 `abort()`
- EngineCore：`_process_aborts_queue()` 在 step 之间处理 abort
- Worker：完成的请求在下一个 `_update_states()` 中被移除

### 13.4 LoRA

```
Stage 1: ChatCompletionRequest 中指定 LoRA adapter
Stage 4: InputProcessor 验证 LoRA 配置
Stage 5: Scheduler 跟踪 active LoRAs（受 max_loras 约束）
Stage 6: Worker 加载 LoRA weights 到 GPU
Stage 8: CUDA Graph 可能需要不同的 graph（按 num_active_loras 特化）
```

### 13.5 DP / TP / PP 下的 request 路由

本文主线默认画的是“一个 request 进入一个 EngineCore，再调度一组 GPU workers”。分布式部署时，
这条链路会被并行复制：

- **TP/PP**：一个 EngineCore 调度一组 worker；这些 worker 共同完成同一个模型 forward。
- **DP**：有多个 EngineCore，每个 DP rank 管一组 worker；前端 `EngineCoreClient` 或外部负载均衡
  选择把 request 发到哪个 DP rank。
- **DP Coordinator**：在内部 DP 负载均衡或 MoE 场景中，协调各 EngineCore 的运行状态、队列计数和
  wave 进度。普通单 DP 读代码时可以先不进入这一层。

这也是 `EngineCoreRequest.client_index` 和 `request_id → engine` 路由表存在的原因：多 API server
和多 EngineCore 时，输出必须回到发起该 request 的 frontend，并且 abort 必须发到持有该 request 的
EngineCore。

---

## 14. 代码阅读顺序

### 第一阶段：API 入口（一个 HTTP 请求怎么进来的）

| 顺序 | 文件 | 关键方法 | 说明 |
|------|------|----------|------|
| 1 | `entrypoints/openai/api_server.py` | `run_server_worker()` (行 681) | 服务器启动 |
| 2 | `entrypoints/openai/chat_completion/api_router.py` | `create_chat_completion()` (行 53) | POST handler |
| 3 | `entrypoints/openai/chat_completion/protocol.py` | `ChatCompletionRequest` (行 150) | 请求结构 |
| 4 | `entrypoints/openai/chat_completion/serving.py` | `create_chat_completion()` (行 229) | 编排逻辑 |

### 第二阶段：渲染与分词（prompt 怎么变成 token IDs）

| 顺序 | 文件 | 关键方法 | 说明 |
|------|------|----------|------|
| 5 | `entrypoints/serve/render/serving.py` | `OpenAIServingRender.render_chat()` | 请求预处理服务层 |
| 6 | `entrypoints/serve/render/serving.py` | `preprocess_chat()` | 构造 ChatParams 并调用 renderer |
| 7 | `renderers/hf.py` | `render_messages_async()` | HF chat template + OpenAI message 解析 |
| 8 | `renderers/base.py` | `render_chat_async()` | renderer 主流程 |
| 9 | `renderers/base.py` | `_tokenize_prompt()` | 分词 |
| 10 | `renderers/base.py` | `_process_multimodal()` | 多模态处理入口 |
| 11 | `multimodal/processing/processor.py` | `BaseMultiModalProcessor.apply()` | vLLM 多模态 processor |

### 第三阶段：引擎提交（请求怎么到 EngineCore）

| 顺序 | 文件 | 关键方法 | 说明 |
|------|------|----------|------|
| 12 | `v1/engine/async_llm.py` | `generate()` (行 521) | 主入口 |
| 13 | `v1/engine/async_llm.py` | `add_request()` (行 280) | 处理 + 注册 + 发送 |
| 14 | `v1/engine/input_processor.py` | `process_inputs()` (行 234) | 输入处理 |
| 15 | `v1/engine/output_processor.py` | `add_request()` (行 508) | 注册 RequestState |
| 16 | `v1/engine/core_client.py` | `add_request_async()` (行 1058) | ZMQ 发送 |

### 第四阶段：调度（请求何时被执行）

| 顺序 | 文件 | 关键方法 | 说明 |
|------|------|----------|------|
| 17 | `v1/engine/core.py` | `run_busy_loop()` (行 1164) | 核心循环 |
| 18 | `v1/engine/core.py` | `step()` (行 402) | 单步执行 |
| 19 | `v1/core/sched/scheduler.py` | `schedule()` (行 352) | 调度算法 |
| 20 | `v1/core/sched/scheduler.py` | `add_request()` (行 1741) | 加入 waiting |
| 21 | `v1/request.py` | `Request` (行 55) | 请求对象 |

### 第五阶段：GPU 执行（模型如何处理请求）

| 顺序 | 文件 | 关键方法 | 说明 |
|------|------|----------|------|
| 22 | `v1/worker/gpu_model_runner.py` | `execute_model()` (行 3787) | 默认 GPU runner |
| 23 | `v1/worker/gpu_model_runner.py` | `_prepare_inputs()` (行 1776) | GPU 输入构建 |
| 24 | `v1/worker/gpu_model_runner.py` | `sample_tokens()` (行 4140) | 采样入口 |
| 25 | `v1/sample/sampler.py` | `forward()` (行 68) | 默认采样逻辑 |
| 25.1 | `v1/worker/gpu/model_runner.py` | `execute_model()` (行 955) | V2 runner 可选路径 |
| 25.2 | `v1/worker/gpu/sample/sampler.py` | `__call__()` (行 57) | V2 采样逻辑 |

### 第六阶段：输出处理（token IDs 怎么变成文本）

| 顺序 | 文件 | 关键方法 | 说明 |
|------|------|----------|------|
| 26 | `v1/core/sched/scheduler.py` | `update_from_output()` (行 1303) | 追加 token、检查停止 |
| 27 | `v1/core/sched/utils.py` | `check_stop()` (行 94) | 停止条件判断 |
| 28 | `v1/engine/output_processor.py` | `process_outputs()` (行 572) | 反分词 + 构建输出 |
| 29 | `v1/engine/detokenizer.py` | `update()` (行 95) | 增量反分词 |

### 第七阶段：响应返回（怎么发回给用户）

| 顺序 | 文件 | 关键方法 | 说明 |
|------|------|----------|------|
| 30 | `v1/engine/async_llm.py` | `_run_output_handler()` (行 634) | 后台拉取输出 |
| 31 | `v1/engine/async_llm.py` | `generate()` (行 570) | yield 给上层 |
| 32 | `entrypoints/openai/chat_completion/serving.py` | `chat_completion_stream_generator()` (行 525) | SSE 流式 |
| 33 | `entrypoints/openai/chat_completion/serving.py` | `chat_completion_full_generator()` (行 1273) | 完整 JSON |

---

## 附录：官方架构页补充的部署视角

官方 Architecture Overview 强调的是进程数量和部署拓扑，可以和本文 request 链路这样对应：

| 组件 | 官方架构视角 | 本文 request 视角 |
|------|--------------|-------------------|
| API Server Process | 处理 HTTP、输入处理、流式返回；通过 ZMQ 连接 EngineCore | Stage 1-4 和 Stage 10-11，包含 OpenAI serving、renderer、AsyncLLM output handler |
| Engine Core Process | 每个 DP rank 一个；运行 scheduler、KV cache 管理、调度 GPU worker | Stage 5 和 Stage 9，重点看 `run_busy_loop()`、`schedule()`、`update_from_output()` |
| GPU Worker Process | 通常一个 worker 控制一个 GPU；执行模型 forward 并管理 GPU 内存 | Stage 6-8，重点看 `GPUModelRunner.execute_model()` 和 `sample_tokens()` |
| DP Coordinator | `--data-parallel-size > 1` 时出现；负责 DP rank 间负载均衡，MoE 场景还要协调同步 forward | 本文只在 EngineCoreClient/DP load balancing 和横切关注点中点到，普通单 DP 请求可先忽略 |

进程数量可以用一个粗略公式记住：如果有 `N` 张 GPU、`TP` tensor parallel size、`PP`
pipeline parallel size、`DP` data parallel size、`A` 个 API server，那么通常有：

```
API Server:     A，默认常随 DP 扩展
EngineCore:     DP
GPU Worker:     N = DP × PP × TP
DP Coordinator: DP > 1 时通常为 1
总数:           A + DP + N (+ DP Coordinator)
```

例如单机 4 GPU、`tp=4` 通常是 `1 API server + 1 EngineCore + 4 GPU workers`；
`tp=2, dp=4` 的 8 GPU 部署会变成多组 API server / EngineCore / worker，并额外有 DP coordinator。
这解释了为什么本文的 request 只画“一条 EngineCore 链路”，但实际部署中可能有多个并行的同构链路。

## 附录：关键数据对象流转图

```
ChatCompletionRequest                   [protocol.py:150]
    │  (Pydantic model: messages, temperature, max_tokens, ...)
    │
    ▼
render_chat() → prompt_token_ids        [renderers/base.py:969]
    │  (list[int]: 分词后的 token IDs)
    │  (+ mm_kwargs / mm_hashes / mm_placeholders 如果有多模态)
    │
    ▼
EngineInput                            [inputs]
    │  (token / multimodal / embeds 等 typed dict)
    │
    ▼
SamplingParams                          [sampling_params.py]
    │  (temperature, top_p, max_tokens, stop, logprobs, structured_outputs, ...)
    │
    ▼
EngineCoreRequest                       [v1/engine/__init__.py]
    │  (prompt_token_ids, sampling_params, mm_features, lora_request, ...)
    │  (mm_features 由 InputProcessor 从 EngineInput 的多模态字段整理而来)
    │  ═══ ZMQ IPC ═══
    ▼
Request                                 [v1/request.py:55]
    │  (num_computed_tokens, output_token_ids, status, kv_cache_blocks, ...)
    │
    ▼
SchedulerOutput                         [v1/core/sched/output.py:179]
    │  (scheduled_new/cached_reqs, num_scheduled_tokens, block_ids)
    │
    ▼
InputBatch (GPU tensors)                [v1/worker/gpu_model_runner.py]
    │  (input_ids, positions, query_start_loc, seq_lens)
    │  (V2 runner: v1/worker/gpu/input_batch.py:36)
    │
    ▼
Hidden States (GPU tensor)              [模型输出]
    │
    ▼
Logits → Sampled Token IDs (GPU)        [采样输出]
    │
    ▼
EngineCoreOutput                        [v1/engine/__init__.py]
    │  (new_token_ids, finish_reason, logprobs)
    │  ═══ ZMQ IPC ═══
    ▼
RequestOutput                           [outputs.py]
    │  (outputs=[CompletionOutput(text, token_ids, logprobs)], finished, ...)
    │
    ▼
ChatCompletionStreamResponse / ChatCompletionResponse
    │  [OpenAI 格式]
    ▼
HTTP Response (SSE / JSON)
```

---

## 附录：压缩完整时序图

这张图用于最后回顾全链路。细节以正文各 stage 附近的局部时序图和调用栈为准。

```
Client          API Server           AsyncLLM          EngineCore         Scheduler         GPUWorker
  │                 │                    │                  │                  │                 │
  │ POST /v1/chat   │                    │                  │                  │                 │
  │────────────────>│                    │                  │                  │                 │
  │                 │                    │                  │                  │                 │
  │                 │ render_chat()      │                  │                  │                 │
  │                 │   ├ chat template  │                  │                  │                 │
  │                 │   ├ tokenize       │                  │                  │                 │
  │                 │   └ process MM     │                  │                  │                 │
  │                 │                    │                  │                  │                 │
  │                 │ generate()         │                  │                  │                 │
  │                 │───────────────────>│                  │                  │                 │
  │                 │                    │                  │                  │                 │
  │                 │                    │ add_request()    │                  │                 │
  │                 │                    │ ├ process_inputs │                  │                 │
  │                 │                    │ ├ register state │                  │                 │
  │                 │                    │ └ ZMQ send ─────>│                  │                 │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ add_request()    │                 │
  │                 │                    │                  │─────────────────>│                 │
  │                 │                    │                  │   加入 waiting    │                 │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ step()           │                 │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ schedule() ─────>│                 │
  │                 │                    │                  │   ← SchedulerOutput                │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ execute_model() ──────────────────>│
  │                 │                    │                  │                  │ _update_states  │
  │                 │                    │                  │                  │ prepare_inputs  │
  │                 │                    │                  │                  │ model forward   │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │                  │<──── hidden_states
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ get_grammar_bitmask()              │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ sample_tokens() ──────────────────>│
  │                 │                    │                  │                  │ apply grammar   │
  │                 │                    │                  │                  │ sample          │
  │                 │                    │                  │                  │<── sampled_ids  │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ update_from_output()               │
  │                 │                    │                  │   ├ append tokens│                 │
  │                 │                    │                  │   ├ check_stop   │                 │
  │                 │                    │                  │   └ if done: free KV cache         │
  │                 │                    │                  │                  │                 │
  │                 │                    │                  │ ── ZMQ ─────────>│                 │
  │                 │                    │                  │ EngineCoreOutputs│                 │
  │                 │                    │                  │                  │                 │
  │                 │                    │ process_outputs()│                  │                 │
  │                 │                    │ ├ detokenize     │                  │                 │
  │                 │                    │ ├ check stops    │                  │                 │
  │                 │                    │ ├ make_request_output               │                 │
  │                 │                    │ └ put in queue   │                  │                 │
  │                 │                    │                  │                  │                 │
  │                 │<───────────────────│                  │                  │                 │
  │                 │ yield RequestOutput│                  │                  │                 │
  │                 │                    │                  │                  │                 │
  │<────────────────│                    │                  │                  │                 │
  │ SSE: data: {...} │                   │                  │                  │                 │
  │                 │                    │                  │                  │                 │
  │           ─ ─ ─ ─ ─ ─ 重复 step() 直到 finished ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │                 │
  │                 │                    │                  │                  │                 │
  │<────────────────│                    │                  │                  │                 │
  │ SSE: data: [DONE]                    │                  │                  │                 │
```
