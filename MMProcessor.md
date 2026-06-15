# vLLM 多模态处理全流程：性能分析与 Profiling 工具

> 标注每个阶段的硬件、计算/通信特性、潜在瓶颈，以及现有 benchmark/profiling 工具。

---

## 目录

1. [全局概览：多模态处理的三阶段模型](#1-全局概览多模态处理的三阶段模型)
2. [Phase 0：前端预处理（CPU，API 进程）](#2-phase-0前端预处理cpuapi-进程)
3. [Phase 1：IPC 与序列化（CPU，跨进程）](#3-phase-1ipc-与序列化cpu跨进程)
4. [Phase 2：GPU Encoder 执行（GPU，Worker 进程）](#4-phase-2gpu-encoder-执行gpuworker-进程)
5. [潜在性能瓶颈汇总](#5-潜在性能瓶颈汇总)
6. [现有 Benchmark 与 Profiling 工具](#6-现有-benchmark-与-profiling-工具)
7. [工具使用指南](#7-工具使用指南)

---

## 1. 全局概览：多模态处理的三阶段模型

多模态请求的处理分布在三个不同的进程/硬件环境中：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 0: API Server 进程 (CPU)                                              │
│                                                                             │
│  HTTP Request (含 image_url / audio / video)                                │
│     │                                                                       │
│     ├─ 0a. 解析 MM 数据 ─── parse_chat_messages_async()                   │
│     │       下载/读取/解码 image/audio/video 原始数据                       │
│     │                                                                       │
│     ├─ 0b. Hash 计算 ──── MultiModalHasher (blake3/sha256)                 │
│     │       将每个 MM item 序列化为 bytes → hash                           │
│     │                                                                       │
│     ├─ 0c. Cache 查询 ──── _get_cache_missing_items()                      │
│     │       LRU cache 查找已处理的 mm_kwargs                               │
│     │                                                                       │
│     ├─ 0d. HF Processor ── _apply_hf_processor()           ★ CPU 最重     │
│     │       image resize/normalize, audio feature extraction              │
│     │       仅处理 cache miss 的 item                                      │
│     │                                                                       │
│     ├─ 0e. Cache 合并 ──── _merge_mm_kwargs()                              │
│     │       将 cached + 新计算的 mm_kwargs 合并                            │
│     │                                                                       │
│     ├─ 0f. Prompt 更新 ──── _apply_prompt_updates()                        │
│     │       替换 placeholder token，计算 mm_placeholders                   │
│     │                                                                       │
│     └─ 输出: MultiModalInput { prompt_token_ids, mm_kwargs, mm_hashes,     │
│                              mm_placeholders }                              │
│                                                                             │
│  ══════════════ ZMQ IPC + Shared Memory ══════════════                      │
│                                                                             │
│ Phase 1: IPC 传输 (CPU)                                                     │
│     │                                                                       │
│     ├─ 1a. 序列化 ──── msgpack + shared memory ring buffer                │
│     ├─ 1b. ZMQ ROUTER → DEALER                                            │
│     └─ 1c. 反序列化 ──── EngineCore 侧还原 tensor                         │
│                                                                             │
│  ══════════════ model_executor 分发到 Worker ══════════════                 │
│                                                                             │
│ Phase 2: GPU Worker 进程 (GPU + CPU)                                        │
│     │                                                                       │
│     ├─ 2a. 批量收集 ──── _batch_mm_inputs_from_scheduler()                 │
│     │       遍历所有调度请求的 encoder inputs                              │
│     │                                                                       │
│     ├─ 2b. Encoder 前向 ──── _execute_mm_encoder()         ★ GPU 最重     │
│     │       Vision/Audio encoder forward → MultiModalEmbeddings            │
│     │       支持同 modality 批量处理 + CUDA Graph                          │
│     │                                                                       │
│     ├─ 2c. Embedding 收集 ──── _gather_mm_embeddings()                     │
│     │       从 encoder_cache 取出 → 构建 is_mm_embed mask                 │
│     │                                                                       │
│     ├─ 2d. Embedding 合并 ──── embed_input_ids()                           │
│     │       text embedding + MM embedding scatter merge                    │
│     │                                                                       │
│     └─ 输出: inputs_embeds (GPU tensor) → 进入 LLM forward                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键设计**：Phase 0 在 API 进程的 ThreadPoolExecutor 中运行（与主事件循环解耦）；
Phase 2 在 GPU Worker 的关键前向路径上，encoder 执行和 LLM 前向是**严格串行**的。

---

## 2. Phase 0：前端预处理（CPU，API 进程）

Phase 0 在 API 进程的 `ThreadPoolExecutor` 中运行（与 async event loop 解耦）。
核心路径是 `BaseRenderer.process_for_engine_async()` → `_process_multimodal_async()` →
`BaseMultiModalProcessor.apply()`。

下面按步骤展开，并在每个步骤中区分 Image / Video / Audio 三种模态的差异。

### 2a. 数据加载与解码（I/O + CPU）

**入口**: `chat_utils.py` → `AsyncMultiModalContentParser.parse_{image,video,audio}()`
**URL 分发**: `MediaConnector.load_from_url()` (`connector.py:285`)

三种 URL 均支持：
- `data:` base64 URL → 内存解码
- `http(s)://` → HTTP 下载（可选磁盘缓存 `VLLM_MEDIA_CACHE`）
- `file://` → 本地文件（需 `--allowed-local-media-path`）

#### Image 加载

**文件**: `vllm/multimodal/media/image.py:20-99` (`ImageMediaIO`)

```
HTTP URL → bytes → PIL.Image.open(BytesIO(data)).convert("RGB")
```

| 属性 | Image |
|------|-------|
| 加载器 | `ImageMediaIO` |
| 输入格式 | URL / base64 / file |
| 解码库 | PIL (`Image.open`) |
| 输出类型 | `PIL.Image.Image`（RGB），包装在 `MediaWithBytes` 中 |
| 特殊处理 | RGBA → RGB 用白色/自定义背景色填充 |

**耗时特征**：单张图片解码通常 < 5ms（本地）或 50-500ms（网络下载）。

#### Video 加载与帧采样

**文件**: `vllm/multimodal/media/video.py:19-166` (`VideoMediaIO`)
**帧采样后端**: `vllm/multimodal/video.py:1-1157` (`VIDEO_LOADER_REGISTRY`)

```
HTTP URL → bytes → VideoBackend.load_bytes()
                     ├─ 打开视频容器 (cv2.VideoCapture / PyAV)
                     ├─ compute_frames_index_to_sample()  ← 帧采样算法
                     ├─ 仅解码被选中的帧
                     └─ 返回 (np.ndarray[N,H,W,3], metadata_dict)
```

**帧采样是 vLLM 完成的，不在 HF Processor 中。** HF processor 收到的是已经采样好的 numpy frames。

| 属性 | Video |
|------|-------|
| 加载器 | `VideoMediaIO` |
| 输入格式 | URL / base64 / file；base64 也支持 `video/jpeg`（逗号分隔 JPEG 帧） |
| 解码库 | OpenCV (`cv2.VideoCapture`) 或 PyAV (`av.open`) |
| 输出类型 | `np.ndarray` (shape `(N, H, W, 3)`, `uint8`) + `metadata` dict |
| 默认帧数 | 32（可通过 `--media-io-kwargs` 或 per-request 覆盖） |
| 环境变量 | `VLLM_VIDEO_LOADER_BACKEND`（默认 `"opencv"`）、`VLLM_VIDEO_FETCH_TIMEOUT`（默认 30s） |

**帧采样算法**（`VideoBackend.compute_frames_index_to_sample`, `video.py:447-470`）：

```python
num_frames_to_sample = min(num_frames, total_frames)
num_frames_to_sample = min(num_frames_to_sample, floor(duration * fps))
indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
```

**已注册的视频后端**：

| 后端名 | 类 | 文件:行号 | 帧采样策略 |
|--------|-----|-----------|------------|
| `opencv` | `VideoBackend` | `video.py:433` | 均匀采样（`np.linspace`） |
| `opencv_dynamic` | `DynamicVideoBackend` | `video.py:553` | 前 `max_duration` 秒按 fps 采样，超出部分均匀采样 |
| `glm4_6v` | `GLM4_6VVideoBackend` | `video.py:642` | GLM4-6V 专用：偶数帧填充 |
| `molmo2` | `Molmo2VideoBackend` | `video.py:743` | Molmo2 专用：支持 `fps` 和 `uniform_last_frame` 两种模式 |
| `nemotron_vl` | `NemotronVLVideoBackend` | `video.py:1029` | 类似 `opencv` 但附加 `original_video_bytes` |
| `openpangu` | `OpenCVDynamicOpenPanguVideoBackend` | `video.py:1059` | OpenPangu 专用：基于时间戳的均匀采样 |

**OpenCV vs PyAV 解码差异**：

| 特性 | OpenCV (`OpenCVVideoBackendMixin`) | PyAV (`PyAVVideoBackendMixin`) |
|------|-------------------------------------|--------------------------------|
| 文件:行号 | `video.py:119-361` | `video.py:363-431` |
| 容器打开 | `cv2.VideoCapture(BytesIO(data))` | `av.open(BytesIO(data))` |
| 帧寻址 | `cap.grab()` + `cap.retrieve()` | `container.seek(target_pts)` |
| 帧恢复 | 有（forward scanning recovery，`video.py:169-272`） | 无（seek 更精确） |
| 稀疏采样效率 | 低（需逐帧 skip） | 高（直接 seek 到目标 PTS） |
| 颜色转换 | BGR → RGB (`cv2.cvtColor`) | 解码到 `rgb24` |

**耗时特征**：视频加载是 Phase 0 中 I/O 最重的操作。本地文件 30fps 10s 视频采样 32 帧约 50-200ms
（取决于解码库和帧分布）；网络下载另加。

#### Audio 加载

**文件**: `vllm/multimodal/media/audio.py:163-206` (`AudioMediaIO`)

```
HTTP URL → bytes → load_audio(BytesIO(data))
                     ├─ load_audio_soundfile()  ← 优先尝试
                     │   └─ soundfile.read(dtype="float32")
                     └─ [fallback] load_audio_pyav()
                          └─ PyAV decode + AudioResampler
```

| 属性 | Audio |
|------|-------|
| 加载器 | `AudioMediaIO` |
| 输入格式 | URL / base64 / file |
| 解码库 | soundfile（优先）/ PyAV（fallback） |
| 输出类型 | `(np.ndarray, float)` — (waveform, sample_rate) |
| 重采样 | **加载时不重采样**（`sr=None`，保持原始采样率） |
| 环境变量 | `VLLM_AUDIO_FETCH_TIMEOUT`（默认 10s） |

**soundfile fallback 触发条件**：当 soundfile 抛出 `LibsndfileError` 且 error code ∈ {0,1,3,4}
（格式检测失败），回退到 PyAV。MP3 等格式通常走 PyAV。

**耗时特征**：短音频（< 30s）解码通常 < 10ms。长音频和 MP3 格式可能稍慢。

### 2a+. 数据解析与格式化（vLLM 侧预处理）

**文件**: `vllm/multimodal/parse.py` — `MultiModalDataParser._parse_{image,video,audio}_data()`

三种模态在此阶段的处理差异显著：

#### Image 数据解析 (`parse.py:592-611`)

```
PIL.Image / np.ndarray / torch.Tensor
  → ImageProcessorItems(data_items)
  → get_processor_data() → {"images": [PIL.Image, ...]}
```

vLLM **不做任何预处理**（不 resize、不 normalize），直接将 PIL Image 传给 HF Processor。

#### Video 数据解析 (`parse.py:613-653`)

```
(np.ndarray[N,H,W,3], metadata_dict) / list[PIL.Image] / torch.Tensor
  → VideoProcessorItems(data_items)
  → get_processor_data() → {"videos": [np.ndarray, ...]}
```

关键点：
- 如果模型的 `video_needs_metadata=True`，metadata dict（含 `fps`, `total_num_frames`,
  `duration`, `frames_indices`）会被保留并传递给 HF processor
- 需要 metadata 的模型：Qwen3-VL、Gemma4-MM、GLM4-1V、NanoNemotronVL、Ernie4.5-VL、Molmo2 等
- metadata 用于 HF processor 计算 temporal positioning（如 `video_grid_thw`）

#### Audio 数据解析 (`parse.py:492-590`)

```
(np.ndarray, float) / list[float] / torch.Tensor
  → _get_audio_with_sr()          # 统一为 (ndarray, sample_rate)
  → AudioResampler.resample()     # 重采样到模型目标采样率（如 16kHz）
  → normalize_audio()             # 通道归一化（如转 mono）
  → AudioProcessorItems(data_items)
  → get_processor_data() → {"audios": [np.ndarray, ...]}
```

**Audio 是唯一在 vLLM 侧做实质预处理的模态**：
- **重采样**：`AudioResampler` 支持 `pyav`（默认）和 `scipy` 两种后端
- **通道归一化**：`normalize_audio()` 支持多种降混策略（mean / first / max / sum）
- 重采样目标由模型的 `get_data_parser()` 指定（如 Whisper → 16kHz、Ultravox → 16kHz）

### 2b. Hash 计算

**文件**: `vllm/multimodal/processing/inputs.py:25-71`, `vllm/multimodal/hasher.py:50-162`

```python
for item in mm_data_items:
    serialized = serialize_item(item)  # 任何类型 → bytes
    hash = blake3/sha256/sha512(serialized)
```

| 属性 | 值 |
|------|-----|
| 硬件 | CPU |
| 类型 | **计算密集**（序列化 + hash） |
| 同步/异步 | 同步，Python 循环逐 item 处理 |
| 缓存 | hash 本身不缓存；hash 用于后续 cache 查找 |

**三种模态的 hash 耗时差异**：

| 模态 | 数据量 | 序列化方式 | 耗时量级 |
|------|--------|-----------|---------|
| Image | ~单张 RGB PIL (H×W×3) | PIL → bytes → hash | ~0.1-1ms |
| Video | ~(N×H×W×3) numpy array | `ndarray.tobytes()` → hash | **~1-50ms**（32帧 1080p 约 30MB） |
| Audio | ~(samples,) float32 array | `ndarray.tobytes()` → hash | ~0.1-5ms（30s@16kHz 约 1.9MB） |

**潜在瓶颈**：视频场景下多帧逐个序列化+hash，没有并行化。对于多个视频 item，串行处理更慢。

### 2c. Cache 查询

**文件**: `vllm/multimodal/processing/processor.py:1299-1334`

| 属性 | 值 |
|------|-----|
| 硬件 | CPU（内存 dict 查找） |
| 类型 | 通信（dict lookup，O(1)） |
| 同步/异步 | 同步 |
| 耗时特征 | 极快，基本可忽略 |

三级缓存架构：
1. `MultiModalProcessorOnlyCache` — 本地进程 LRU cache（`cache.py:326-377`）
2. `MultiModalProcessorSenderCache` — IPC 场景下 P0 侧仅存元数据（`cache.py:379-435`）
3. `ShmObjectStoreSenderCache` — 共享内存环形缓冲区（`cache.py:437-582`）

### 2d. HF Processor 调用（Phase 0 最重的 CPU 步骤）

**文件**: `vllm/multimodal/processing/processor.py:1097-1510`

```
_cached_apply_hf_processor()
  ├─ cache 命中 → 直接返回 mm_kwargs
  └─ cache miss → _apply_hf_processor_main()
                     └─ _call_hf_processor()
                          └─ HF Processor (transformers.ProcessorMixin)
```

| 属性 | 值 |
|------|-----|
| 硬件 | CPU |
| 类型 | **计算密集** |
| 同步/异步 | 在 ThreadPoolExecutor 中执行 |
| 缓存 | 按 mm_hash 全缓存；仅 cache miss 触发实际计算 |

**三种模态在 HF Processor 中的处理差异**：

#### Image — HF Processor 做了什么

HF Image Processor（如 `CLIPImageProcessor`、`SiglipImageProcessor`）：
1. **Resize** — 到模型目标分辨率（如 336×336 / 448×448）
2. **Center crop / Smart resize** — 模型特定
3. **Normalize** — ImageNet mean/std 归一化
4. **Tensor 转换** — PIL → `torch.Tensor` (float16/bfloat16)

输出：`BatchFeature` 含 `pixel_values` (shape `(B, C, H, W)`)

**耗时**：单张 ~2-10ms；多张可批量处理。

**模型特定 override**：

| 模型 | 文件:行号 | 特殊处理 |
|------|-----------|---------|
| Qwen2-VL | `qwen2_vl.py:848` | `smart_resize()` 根据 `min_pixels`/`max_pixels` 计算分辨率 |
| InternVL | `internvl.py:215-240` | 额外添加 `image_token_id` 到输出 |
| Llava-OneVision | `llava_onevision.py:324-393` | 先处理文本，再逐个处理 image/video（HF 不支持不同尺寸批量） |
| Phi-3-Vision | `phi3v.py:397-420` | 修正负数 placeholder token ID（HF 插入 -1, -2 等） |
| PaliGemma | `paligemma.py:155` | 自定义 `_call_hf_processor` override |

#### Video — HF Processor 做了什么

HF Video Processor 收到 `videos=[np.ndarray(N,H,W,3)]`（已经由 vLLM 采样好的帧）：

1. **逐帧处理** — 每帧等同于 image processing（resize + normalize + tensor）
2. **模型特定 temporal 处理** — 如 Qwen2-VL 的 `video_grid_thw` 计算
3. **Tensor 堆叠** — 输出 `(B, N, C, H, W)` 或展平为 `(B*T, C, H, W)`

输出：`BatchFeature` 含 `pixel_values_videos` / `video_grid_thw` 等

**耗时**：帧数 × 单帧处理时间。32 帧约为单张图片的 32 倍，~60-300ms。

**模型特定 override**：

| 模型 | 文件:行号 | 特殊处理 |
|------|-----------|---------|
| Qwen2-VL | `qwen2_vl.py:833-845` | 支持 `DictEmbeddingItems`（预计算 embedding bypass） |
| MiniCPM-V | `minicpmv.py:525-535` | 自定义 `MiniCPMVVideoEmbeddingItems` |
| Llava-OneVision | `llava_onevision.py:324-393` | 逐个 video 循环处理（不同尺寸不能批量） |
| Keye | `keye.py:958-973` | 自定义 `DictEmbeddingItems` |

#### Audio — HF Processor 做了什么

HF Audio Feature Extractor（如 `WhisperFeatureExtractor`）：
1. **Log-mel spectrogram** — 短时傅里叶变换 + mel filter bank
2. **Padding / Truncation** — 到模型目标长度
3. **Normalize** — 特征归一化
4. **Tensor 转换** — → `torch.Tensor`

输出：`BatchFeature` 含 `input_features` / `feature_attention_mask`

**耗时**：通常 ~5-20ms，比 image/video 快得多。

**模型特定 override**：

| 模型 | 文件:行号 | 特殊处理 |
|------|-----------|---------|
| Whisper | `whisper.py:738` | 重命名 `audios` → `audio`，过滤非音频 tok_kwargs |
| Qwen2.5 Omni | `qwen2_5_omni_thinker.py:464` | 后处理 `input_features` → `input_audio_features`，加 `audio_feature_lengths` |
| Ultravox | `ultravox.py:189` | 传递 `sampling_rate`，重命名 `audio_values` → `audio_features` |
| GlmASR | `glmasr.py:660` | 支持预计算 embedding bypass |
| KimiAudio | `kimi_audio.py:166` | 自定义 data parser |

#### 三种模态 HF Processor 耗时对比

```
Image:  [Resize] [Normalize] [Tensor]                    ~2-10ms/item
Video:  [Frame×N] [Resize×N] [Normalize×N] [Tensor×N]    ~60-300ms/item  ★ 最重
Audio:  [Mel Spectrogram] [Normalize] [Tensor]            ~5-20ms/item
```

### 2e. Cache 合并

**文件**: `vllm/multimodal/processing/processor.py:1347-1396`

| 属性 | 值 |
|------|-----|
| 硬件 | CPU |
| 类型 | 通信（dict 操作 + 可能的 msgpack 序列化） |
| 同步/异步 | 同步 |
| 耗时特征 | 通常很快；共享内存写满时可能触发 MemoryError |

将 cached items 和新计算的 items 按顺序合并成一个完整的 `mm_kwargs` 列表。
对于同一请求中的多个 MM item，每个 item 独立 cache：已 cache 的跳过，仅处理 miss 的。

### 2f. Prompt 更新（Placeholder 替换）

**文件**: `vllm/multimodal/processing/processor.py:1528-1661`

| 属性 | 值 |
|------|-----|
| 硬件 | CPU |
| 类型 | 计算（token 序列扫描与替换） |
| 同步/异步 | 同步 |
| 耗时特征 | 通常很快；fallback 到 string matching 时较慢 |

将 `<image>` / `<video>` / `<audio>` 等 placeholder token 替换为模型特定的
placeholder token 序列，计算每个 MM item 对应的 token 位置范围（`mm_placeholders`）。

### Phase 0 三种模态完整对比

```
                    Image                         Video                           Audio
                    ─────                         ─────                           ─────
Loader:             ImageMediaIO                  VideoMediaIO                    AudioMediaIO
                    (PIL Image.open)              (OpenCV / PyAV)                 (soundfile / PyAV)

vLLM 侧预处理:      无                            帧采样（均匀/FPS/模型特定）       重采样 + 通道归一化
                    (直接传 PIL)                   + metadata 生成                  (AudioResampler + normalize)

传给 HF 的格式:      {"images": [PIL.Image]}      {"videos": [np.ndarray]}        {"audios": [np.ndarray]}
                                                  + 可选 metadata dict

HF Processor 做什么: resize → normalize            逐帧 resize → normalize          mel spectrogram → normalize
                    → tensor                      → tensor → temporal 处理          → padding → tensor

典型耗时:            ~2-10ms/item                  ~60-300ms/item (32帧)            ~5-20ms/item
                    (轻)                          ★★★ 最重                         (中等)

Hash 序列化开销:     小 (~MB)                      大 (~30MB for 32帧1080p)         中 (~2MB for 30s@16kHz)

Cache 命中收益:      跳过 HF resize/normalize      跳过 32× HF resize/normalize     跳过 mel spectrogram
                    中等收益                       ★★★ 收益最大                      中等收益

模型特殊 override:   Qwen2/InternVL/Phi3V 等       Qwen2VL/MiniCPMV/LlavaOV 等      Whisper/Qwen2.5Omni 等
                    ~6 个模型                      ~4 个模型                         ~15 个模型
```

---

## 3. Phase 1：IPC 与序列化（CPU，跨进程）

### 序列化与共享内存传输

**文件**: `vllm/multimodal/cache.py:437-582`

| 属性 | 值 |
|------|-----|
| 硬件 | CPU + 共享内存 |
| 类型 | 通信（序列化 + 内存拷贝） |
| 同步/异步 | 同步（在 Phase 0 的 thread pool 中） |
| 耗时特征 | 大 tensor 的 msgpack 序列化较慢；受共享内存环形缓冲区大小限制 |

传输路径：
```
API 进程                    EngineCore 进程
  │                              │
  ├─ msgpack serialize           │
  ├─ ShmRingBuffer.put()         │
  │  (可能触发 MemoryError)       │
  │                              ├─ ShmRingBuffer.get()
  ├─ ZMQ ROUTER ──────────────>  ├─ msgpack deserialize
  │                              └─ 还原 mm_kwargs tensors
```

**注意**：`pending_messages` 机制确保 ZMQ buffer 发送完成前 tensor 不被释放。

---

## 4. Phase 2：GPU Encoder 执行（GPU，Worker 进程）

### 执行流程总览

```
execute_model(scheduler_output)
  │
  ├─ _update_states()                    [CPU]  释放 finished 请求的 encoder cache
  │
  ├─ _process_inputs()
  │    ├─ _execute_mm_encoder()           [GPU]  ★ encoder 前向
  │    ├─ _gather_mm_embeddings()         [CPU+GPU] 构建 mask + 取出 embedding
  │    └─ embed_input_ids()               [GPU]  text embed + MM merge
  │
  └─ model(**model_inputs)               [GPU]  LLM forward
```

**关键**：encoder → gather → merge → LLM forward 是**严格串行**的。

### 2a. 批量收集

**文件**: `vllm/v1/worker/gpu_model_runner.py:2828-2866`

| 属性 | 值 |
|------|-----|
| 硬件 | CPU（Python dict/list 操作） |
| 类型 | 通信 |
| 耗时特征 | O(num_requests × num_features) 的 Python 循环 |

收集所有被调度请求的 `mm_features`，触发 lazy deserialization（如果使用了共享内存 cache）。

### 2b. Encoder 前向（最重的 GPU 步骤）

**文件**: `vllm/v1/worker/gpu_model_runner.py:2868-3077`

```
_execute_mm_encoder()
  ├─ _batch_mm_inputs_from_scheduler()       收集所有 encoder inputs
  ├─ 按 modality 分组                         group_and_batch_mm_kwargs()
  │    └─ 仅连续同 modality 的 item 可批量    ← 局限性
  │
  ├─ [有 CUDA Graph]
  │    └─ EncoderCudaGraphManager.execute()   预算制 graph replay
  │         ├─ greedy packing（最小先装）
  │         ├─ 按 power-of-2 budget 对齐      ← 可能浪费算力
  │         ├─ [DP 模式] TP rank 间分发 + all-gather
  │         └─ graph.replay()
  │
  ├─ [无 CUDA Graph / 超出 budget]
  │    └─ model.embed_multimodal(**mm_kwargs)  eager forward
  │
  └─ 缓存结果到 encoder_cache[mm_hash]
```

| 属性 | 值 |
|------|-----|
| 硬件 | GPU（encoder forward）+ CPU（数据 staging） |
| 类型 | **计算密集**（ViT / Audio encoder 前向） |
| 耗时特征 | 取决于模型大小和输入分辨率；ViT-L 约 10-50ms，ViT-big 可能更长 |
| 缓存 | `self.encoder_cache: dict[str, Tensor]`，按 mm_hash 索引 |
| 批量 | 跨请求同 modality 可批量；但仅限连续同 modality（FIXME: hacky） |

**CUDA Graph 的预算制**：
- 预算为 power-of-2 token 数（64, 128, 256, ...）
- greedy packing 将 item 尽量塞入最小可用 budget
- 超出最大 budget 的 item fallback 到 eager
- padding 浪费比例会被记录到日志

### 2c. Embedding 收集

**文件**: `vllm/v1/worker/gpu_model_runner.py:3079-3178`

```
_gather_mm_embeddings()
  ├─ is_mm_embed = torch.zeros(total_tokens, dtype=bool, device="cpu")  ← CPU 上分配
  │
  ├─ for req in requests:
  │    for feature in req.mm_features:
  │      encoder_output = encoder_cache[mm_hash]    ← GPU tensor
  │      sliced = encoder_output[start:end]         ← GPU 切片
  │      mm_embeds_list.append(sliced)
  │      is_mm_embed[start:end] = True              ← CPU mask 设置
  │
  └─ return mm_embeds, is_mm_embed
```

| 属性 | 值 |
|------|-----|
| 硬件 | CPU（boolean mask）+ GPU（tensor 切片） |
| 类型 | 通信（mask 构建 + tensor 索引） |
| 耗时特征 | O(num_requests × num_features) 的 Python 循环 |

**潜在问题**：
- `is_mm_embed` 在 CPU 分配，后续用于 GPU scatter merge 时需要 CPU→GPU 传输
- 嵌套 Python 循环在大 batch 下可能有明显开销

### 2d. Embedding 合并

**文件**: `vllm/model_executor/models/interfaces.py:374-408`, `utils.py:456-492`

```
embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
  ├─ text_embeds = embed(input_ids.masked_fill(is_multimodal, 0))  ← OOV token mask
  ├─ mm_embeds_flat = flatten(multimodal_embeddings)
  └─ text_embeds[is_multimodal] = mm_embeds_flat                   ← in-place scatter
```

| 属性 | 值 |
|------|-----|
| 硬件 | GPU（embedding lookup + scatter） |
| 类型 | 计算（embedding lookup）+ 通信（scatter write） |
| 耗时特征 | embedding lookup 快；scatter 需要 CPU→GPU 的 mask 传输 |

**注意**：代码中有显式 GPU→GPU copy 和 TODO 注释：
```python
# gpu_model_runner.py:3412
self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
# TODO(woosuk): Avoid the copy. Optimize.
```

### Encoder Cache 管理（Scheduler 侧）

**文件**: `vllm/v1/core/encoder_cache_manager.py:17-266`

| 属性 | 值 |
|------|-----|
| 硬件 | CPU（Scheduler 进程） |
| 类型 | 通信（dict 操作，budget 管理） |
| 缓存策略 | 基于 token 数量的 budget；LRU 驱逐无引用的 entry |
| 跨请求复用 | 相同 mm_hash 的 encoder 输出可跨请求共享 |

### Encoder CUDA Graph 详细机制

**文件**: `vllm/v1/worker/encoder_cudagraph.py:1-625`

```
EncoderCudaGraphManager
  ├─ 初始化时：为每个 power-of-2 budget 预分配 input/output buffer
  │    └─ capture CUDA graph（含 warmup）
  │
  ├─ 推理时 execute():
  │    ├─ 按 modality 分组 items
  │    ├─ 对每组：
  │    │    ├─ greedy packing → 选最小可用 budget
  │    │    ├─ 填充 input buffer → graph.replay()
  │    │    └─ 从 output buffer 取出结果
  │    └─ 超出 budget 的 item → eager forward fallback
  │
  └─ [DP 模式] _dp_shard()
       ├─ 按 input size 将 item 分配到 TP rank
       ├─ 各 rank 本地执行
       └─ all-gather 合并结果
```

---

## 5. 潜在性能瓶颈汇总

### 按优先级排序

| 优先级 | 瓶颈点 | 阶段 | 类型 | 文件:行号 | 原因 |
|--------|--------|------|------|-----------|------|
| **🔴 HIGH** | HF Processor 调用 | Phase 0 | CPU 计算 | `processor.py:1097-1114` | 图像/视频预处理是最重的 CPU 操作；视频逐帧处理 |
| **🔴 HIGH** | Encoder 前向 | Phase 2 | GPU 计算 | `gpu_model_runner.py:2868-3077` | ViT/Audio encoder 是 MM 请求中最重的 GPU 操作 |
| **🔴 HIGH** | `is_mm_embed` CPU mask → GPU 传输 | Phase 2 | CPU→GPU 通信 | `gpu_model_runner.py:3087-3088` | Boolean mask 在 CPU 创建，用于 GPU scatter，隐式同步 |
| **🟡 MED** | Hash 计算（视频场景） | Phase 0 | CPU 计算 | `hasher.py:50-162` | 多帧逐个序列化 + hash，无并行化 |
| **🟡 MED** | Post-merge GPU memcpy | Phase 2 | GPU 内存带宽 | `gpu_model_runner.py:3412` | 显式 embedding copy；代码已有 TODO 标记 |
| **🟡 MED** | `_gather_mm_embeddings` Python 循环 | Phase 2 | CPU 开销 | `gpu_model_runner.py:3095-3169` | 逐请求逐 feature 的 Python 循环 + CPU tensor 操作 |
| **🟡 MED** | CUDA Graph padding 浪费 | Phase 2 | GPU 计算浪费 | `encoder_cudagraph.py:414-416` | Budget 对齐导致 padding，浪费 GPU 算力 |
| **🟡 MED** | Encoder 与 LLM 串行 | Phase 2 | 架构限制 | `gpu_model_runner.py:3393-3412` | Encoder → Merge → LLM 严格串行，无重叠 |
| **🟢 LOW** | 共享内存序列化 | Phase 1 | CPU/IPC | `cache.py:437-582` | 大 tensor 的 msgpack 序列化 |
| **🟢 LOW** | Token 替换 fallback | Phase 0 | CPU 计算 | `processor.py:1551-1560` | String matching + re-encode |
| **🟢 LOW** | Encoder cache dict lookup | Phase 2 | CPU 开销 | `gpu_model_runner.py:3134` | Python dict 在热循环中 |

### 按阶段汇总

```
                    CPU 时间                           GPU 时间
Phase 0:  [HF Processor] ████████████████ (最重)
          [Hash]         ████
          [Cache]        █
          [Prompt 更新]  ██

Phase 1:  [IPC 序列化]   ███

Phase 2:                                   [Encoder Forward] ████████████████ (最重)
                                            [Embed Merge]    ██
                                            [Memcpy]         █
          [Gather Python 循环] ███
          [CPU mask]          ██
```

### 多模态特有的性能特征（vs 纯文本请求）

1. **Prefill 阶段显著变长**：encoder 前向 + embedding merge 额外增加了 prefill 时间
2. **CPU 预处理阻塞**：HF processor 在 CPU 上处理大图/视频可能成为请求延迟的瓶颈
3. **KV Cache 压力**：每个 MM item 占用大量 KV cache slot（如 576+ tokens per image）
4. **调度复杂度**：encoder cache 的 budget 管理和驱逐增加了调度开销
5. **内存峰值**：encoder 前向的中间激活 + encoder cache + text embedding 并存

---

## 6. 现有 Benchmark 与 Profiling 工具

### 6.1 `vllm bench mm-processor`（核心 MM 专用 benchmark）

**文件**: `vllm/benchmarks/mm_processor.py`（539 行）

**用法**：
```bash
vllm bench mm-processor \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset-name random-mm \
  --num-prompts 10 \
  --num-warmups 1 \
  --output-json results.json
```

**CLI 参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-name` | `random-mm` | 数据集类型：`random-mm` 或 `hf` |
| `--dataset-path` | - | HF 数据集路径（如 `lmarena-ai/VisionArena-Chat`） |
| `--num-prompts` | 10 | 测试 prompt 数量 |
| `--num-warmups` | 1 | 预热轮数 |
| `--metric-percentiles` | `99` | 输出的百分位 |
| `--output-json` | - | 保存结果的 JSON 文件路径 |
| 所有 `EngineArgs` | - | `--model`, `--tensor-parallel-size` 等 |

**测量的阶段**：
- `apply_hf_processor_secs` — HF Processor 调用耗时
- `get_mm_hashes_secs` — Hash 计算耗时
- `get_cache_missing_items_secs` — Cache miss 检测耗时
- `merge_mm_kwargs_secs` — Cache 合并耗时
- `apply_prompt_updates_secs` — Prompt 更新耗时
- `preprocessor_total_secs` — 预处理总耗时
- `encoder_forward_secs` — GPU Encoder 前向耗时
- `num_encoder_calls` — Encoder 调用次数

**底层 Timing 基础设施**：

1. **TimingContext**（`vllm/multimodal/processing/context.py:47`）
   - 使用 `time.perf_counter()` 的高精度计时器
   - `record(stage)` 上下文管理器自动记录阶段耗时

2. **MultiModalTimingRegistry**（`vllm/multimodal/registry.py:350`）
   - 按 request_id 存储 TimingContext
   - `stat()` 返回所有统计并清空 registry
   - 仅当 `observability_config.enable_mm_processor_stats=True` 时启用

3. **EncoderTimingStats**（`vllm/v1/worker/gpu_model_runner.py:7422-7436`）
   - 记录 encoder 前向耗时和调用次数
   - 使用 `torch.accelerator.synchronize()` + `time.perf_counter()` 确保精确 GPU 计时

### 6.2 `vllm bench` 通用 benchmark

**文件**: `vllm/benchmarks/` 目录

```bash
# 通用 serving benchmark（含 TTFT/ITL 等延迟指标）
vllm bench serve \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset-name random-mm \
  --num-prompts 100

# 离线 throughput benchmark
vllm bench latency \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --batch-size 1
```

### 6.3 Kernel 级 MM Benchmark

| 文件 | 用途 |
|------|------|
| `benchmarks/kernels/benchmark_vit_bilinear_pos_embed.py` | Triton bilinear pos embed kernel vs PyTorch（Qwen3-VL） |
| `benchmarks/kernels/benchmark_vit_fp8_attn.py` | FP8 vs BF16 ViT attention（FlashInfer cuDNN backend） |

### 6.4 Profiler 基础设施

#### Torch Profiler 集成

```bash
# 通过 CLI 启用
--profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'

# 或通过 HTTP API（serve 模式）
curl http://localhost:8000/start_profile
curl http://localhost:8000/stop_profile
```

**配置项**（`vllm/config/profiler.py`）：
- `delay_iterations` — 延迟多少 iteration 开始
- `max_iterations` — 最大记录 iteration 数
- `warmup_iterations` — warmup iteration 数
- `torch_profiler_with_stack` — 是否记录 stack trace

#### Layerwise Profiler

**文件**: `vllm/profiler/layerwise_profile.py`

逐层 profiling 工具，输出 CUDA 时间树，可定位 MM encoder 内部各层的耗时分布。

**可视化工具**：
- `tools/profiler/visualize_layerwise_profile.py` — 可视化 layerwise profile
- `tools/profiler/print_layerwise_table.py` — 表格形式输出

#### Nsys Profiling

**目录**: `tools/profiler/nsys_profile_tools/`

NVIDIA Nsight Systems profiling 工具集，可分析 GPU kernel 级别的耗时和 overlap。

### 6.5 Profiling 示例脚本

| 文件 | 说明 |
|------|------|
| `examples/features/profiling/simple_profiling_offline.py` | 简单离线 profiling 示例 |
| `examples/features/profiling/run_one_batch_offline.py` | 单批次 profiling，支持 `--profile prefill\|decode\|both` |

### 6.6 MM 内存 Profiling

```bash
# 跳过 MM 内存 profiling（仅 profile 语言模型）
--skip-mm-profiling

# 或在代码中
MultiModalConfig(skip_mm_profiling=True)
```

`gpu_model_runner.py:6108` 的 `profile_run()` 在引擎初始化时执行，会：
- 创建 dummy MM 数据
- 执行 encoder forward 测量内存占用
- 计算 encoder cache size

### 6.7 Benchmark 数据集

**文件**: `vllm/benchmarks/datasets/datasets.py`

| 数据集 | 行号 | 说明 |
|--------|------|------|
| `RandomMultiModalDataset` | 881 | 合成 MM 数据，可配置 image/video bucket |
| `VisionArenaDataset` | 3258 | HuggingFace VisionArena 数据集 |
| `MultiModalConversationDataset` | 3193 | MM 对话数据集 |
| `CustomImageDataset` | 2536 | 自定义图片数据集 |
| `CustomAudioDataset` | 2768 | 自定义音频数据集 |

---

## 7. 工具使用指南

### 快速开始：发现 MM 性能瓶颈

#### Step 1：运行 MM Processor Benchmark

```bash
# 基础：测量各阶段延迟
vllm bench mm-processor \
  --model <your_model> \
  --dataset-name random-mm \
  --num-prompts 50 \
  --output-json mm_bench_baseline.json

# 使用真实数据集
vllm bench mm-processor \
  --model <your_model> \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 100
```

**关注指标**：
- `preprocessor_total_secs` — CPU 预处理总耗时
- `apply_hf_processor_secs` — HF Processor 占比
- `encoder_forward_secs` — GPU encoder 耗时
- 各阶段的 p50 / p99 分布

#### Step 2：对比纯文本 vs MM 请求延迟

```bash
# 纯文本 serving benchmark
vllm bench serve --model <your_model> --dataset-name random --num-prompts 100

# MM serving benchmark
vllm bench serve --model <your_model> --dataset-name random-mm --num-prompts 100
```

对比 TTFT（Time To First Token），差值即为 MM 处理引入的额外延迟。

#### Step 3：GPU 级 Profiling（深入 encoder 内部）

```bash
# 方法 1：PyTorch Profiler
python examples/features/profiling/run_one_batch_offline.py \
  --model <your_model> \
  --profile prefill \
  --batch-size 1

# 方法 2：nsys profiling
nsys profile -o mm_profile \
  python -c "
from vllm import LLM
llm = LLM(model='<your_model>')
# ... 发送 MM 请求 ...
"

# 方法 3：layerwise profiling（定位 encoder 内部耗时）
python -c "
from vllm.profiler.layerwise_profile import layerwise_profile
# 在 model forward 前后启用
"
```

#### Step 4：分析 encoder CUDA Graph 效率

查看日志中的 CUDA Graph waste percentage 输出（`encoder_cudagraph.py:414-416`）。

如果 padding 浪费严重，考虑调整 `--mm-encoder-max-batch-size` 或相关配置。

### 实验设计建议

| 实验场景 | 工具 | 关注点 |
|----------|------|--------|
| CPU 预处理瓶颈 | `vllm bench mm-processor` | `apply_hf_processor` vs `preprocessor_total` 占比 |
| GPU encoder 瓶颈 | `vllm bench mm-processor` + nsys | `encoder_forward_secs`，encoder kernel 时间分布 |
| 高并发下 MM 延迟 | `vllm bench serve --dataset-name random-mm` | TTFT p99，对比纯文本基线 |
| 大图片/视频性能 | 自定义 `RandomMultiModalDataset` bucket | 不同分辨率/帧数下的延迟曲线 |
| Encoder CUDA Graph 效率 | 日志 + nsys | padding 浪费比例，graph replay 时间 |
| 跨请求 MM cache 命中率 | 自定义脚本 + timing registry | cache miss → HF processor 调用次数 |
| 端到端 overlap 分析 | nsys timeline | encoder 与 LLM 的 timeline 是否可重叠 |

---

## 关键文件索引

| 组件 | 文件 | 行号 |
|------|------|------|
| MM 入口（Renderer） | `vllm/renderers/base.py` | 686-724 |
| HF Processor 调用 | `vllm/multimodal/processing/processor.py` | 1097-1510 |
| Hash 计算 | `vllm/multimodal/hasher.py` | 50-162 |
| MM Cache（三级） | `vllm/multimodal/cache.py` | 326-741 |
| Encoder 前向 | `vllm/v1/worker/gpu_model_runner.py` | 2868-3077 |
| Embedding 收集 | `vllm/v1/worker/gpu_model_runner.py` | 3079-3178 |
| Embedding 合并 | `vllm/model_executor/models/utils.py` | 456-492 |
| Encoder CUDA Graph | `vllm/v1/worker/encoder_cudagraph.py` | 1-625 |
| Encoder Cache（Scheduler） | `vllm/v1/core/encoder_cache_manager.py` | 17-266 |
| MM Benchmark | `vllm/benchmarks/mm_processor.py` | 1-539 |
| Timing 基础设施 | `vllm/multimodal/processing/context.py` | 47 |
| Timing Registry | `vllm/multimodal/registry.py` | 350 |
| Encoder Timing | `vllm/v1/worker/gpu_model_runner.py` | 7362-7436 |
| Layerwise Profiler | `vllm/profiler/layerwise_profile.py` | - |
| Profiler 配置 | `vllm/config/profiler.py` | - |
| MM 内存 Profiling | `vllm/v1/worker/gpu_model_runner.py` | 6108 |
