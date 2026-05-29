# GLM5Next PD (Prefill-Decode) 分离实现计划

## Context

GLM5Next 是 hybrid 模型（实现 `IsHybrid`），有两种层：
- **MLA attention 层**（非 KDA）：用 `DeepseekV2MLAAttention` + `MLAAttentionSpec`
- **KDA 层**（线性注意力）：用 `KimiDeltaAttention` + `MambaSpec`，有 **4 个状态张量**：`conv_state_q`, `conv_state_k`, `conv_state_v`, `recurrent_state`

vLLM 的 PD 分离已支持 hybrid 模型（Mamba2），但 NIXL connector 的 3-read transfer 只支持 `mamba_type="mamba2"`。KDA 的 `mamba_type="gdn_attention"` 会被 `NotImplementedError` 拒绝。

核心问题：**状态结构不同**
- Mamba2: 2 个状态 — conv_state(x+B+C混合), ssm_state
- KDA: 4 个状态 — conv_q, conv_k, conv_v, recurrent (各自独立)

## 方案概述

扩展现有 NIXL connector 支持 KDA 的 4 状态结构。主要修改 4 个文件：

1. `ssm_conv_transfer_utils.py` — 新增 KDA 状态分解类型和推导函数
2. `nixl/worker.py` — 新增 KDA 的 local/remote descriptor 构建方法
3. `nixl/metadata.py` — `ssm_sizes` 从 `tuple[int, int]` 泛化为 `tuple[int, ...]`
4. `nixl/scheduler.py` — 验证 N-1 token truncation 逻辑（预计无需修改）

## 详细修改

### 1. `ssm_conv_transfer_utils.py`

**路径**: `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py`

**1a. 新增 `KDAStateSplitInfo` dataclass**

```python
@dataclass(frozen=True)
class KDAStateSplitInfo:
    """KDA 的 4 个状态在 page 内的字节布局。

    DS 内存布局 (contiguous in memory):
        |--- conv_q ---|--- conv_k ---|--- conv_v ---|--- recurrent ---|
    """
    state_sizes: tuple[int, int, int, int]  # 每个 state 的字节数

    @property
    def local_conv_offsets(self) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) for conv_q, conv_k, conv_v, recurrent."""
        offsets = []
        offset = 0
        for sz in self.state_sizes:
            offsets.append((offset, sz))
            offset += sz
        return offsets

    def remote_state_offsets(
        self, local_rank_offset: int, tp_ratio: int
    ) -> list[tuple[int, int]]:
        """Hetero-TP 时每个 D rank 从 P page 读取的 (offset, size)。"""
        result = []
        offset_acc = 0
        for sz in self.state_sizes:
            local_sz = sz if tp_ratio < 0 else sz
            result.append((offset_acc + local_rank_offset * local_sz, local_sz))
            offset_acc += sz * abs(tp_ratio) if tp_ratio >= 1 else sz
        return result
```

**1b. 新增 `derive_kda_state_split()` 函数**

```python
def derive_kda_state_split(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> KDAStateSplitInfo:
    assert len(mamba_spec.shapes) == 4
    assert is_conv_state_dim_first()
    state_bytes = []
    for shape, dtype in zip(mamba_spec.shapes, mamba_spec.dtypes):
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        state_bytes.append(torch.Size(shape).numel() * dtype_size)
    return KDAStateSplitInfo(state_sizes=tuple(state_bytes))
```

**1c. 修改 `derive_mamba_conv_split()` — 按 `len(shapes)` 分发**

```python
def derive_mamba_conv_split(mamba_spec, local_tp):
    if len(mamba_spec.shapes) == 4:
        return derive_kda_state_split(mamba_spec, local_tp)
    # ... 原有 Mamba2 逻辑不变
```

返回类型变为 `Union[MambaConvSplitInfo, KDAStateSplitInfo]`。两者都提供 `local_conv_offsets` 方法。

**1d. 泛化 `compute_physical_blocks_per_logical()`**

```python
def compute_physical_blocks_per_logical(ssm_sizes, block_len):
    return math.ceil(sum(ssm_sizes) / block_len)  # 已是通用的
```

### 2. `nixl/worker.py`

**路径**: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`

**2a. `_mamba_ssm_size` 类型从 `tuple[int, int]` 改为 `tuple[int, ...]`**

```python
# Line 234: 从 (0, 0) 改为 ()
mamba_ssm_size: tuple[int, ...] = ()
```

初始化处统一用 split info 的 `state_sizes`。

**2b. `_build_mamba_local()` — 分发到 KDA 路径**

```python
def _build_mamba_local(self, base_addresses, block_size_ratio):
    if isinstance(self._conv_decomp, KDAStateSplitInfo):
        return self._build_kda_local(base_addresses, block_size_ratio)
    # ... 原有 Mamba2 逻辑
```

新增 `_build_kda_local()`：4 个 desc regions per layer（conv_q, conv_k, conv_v, recurrent），结构与 Mamba2 版本类似但 offsets 不同。

**2c. `_build_mamba_remote()` — 同样分发**

新增 `_build_kda_remote()` 处理 hetero-TP 的偏移计算。每个 state 独立 TP 分片。

**2d. `_compute_desc_ids()` — 通用化 region 数量**

```python
# 原: num_ssm_regions = len(self.block_len_per_layer) * 4
# 改: 从 split info 动态计算
num_ssm_regions = (
    len(self.block_len_per_layer) * len(self._conv_decomp.local_conv_offsets)
    if self._has_mamba else 0
)
```

### 3. `nixl/metadata.py`

**路径**: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py`

**3a. `NixlAgentMetadata.ssm_sizes` 类型变更**

```python
# Line 50: 从
ssm_sizes: tuple[int, int]
# 改为
ssm_sizes: tuple[int, ...]
```

**3b. 版本升级**

```python
NIXL_CONNECTOR_VERSION: int = 4  # 从 3 升到 4
```

### 4. `nixl/scheduler.py` — 验证，预计无需修改

- `_has_mamba` 通过检查 `MambaSpec` 设置 — KDA 层也会触发
- `_mamba_prefill_token_count()` 返回 N-1 — 对 KDA 同样适用
- `_truncate_mamba_request_for_prefill()` 截断最后一个 prompt token — 逻辑与 mamba_type 无关

## 配置与启动

### 环境要求
- NIXL connector + RDMA (NIXL backend installed)
- `VLLM_SSM_CONV_STATE_LAYOUT=DS` (必须，3-read transfer 需要)

### P (Prefill) 节点
```bash
VLLM_SSM_CONV_STATE_LAYOUT=DS \
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"NixlConnector","engine_id":"prefill"}' \
  --no-disable-hybrid-kv-cache-manager
```

### D (Decode) 节点
```bash
VLLM_SSM_CONV_STATE_LAYOUT=DS \
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"NixlConnector","engine_id":"decode"}' \
  --no-disable-hybrid-kv-cache-manager
```

## MTP + PD 交互

- MTP speculative decoding 仅在 D 端运行，不影响 P 端的 KV 传输
- **注意**: P/D 端 KDA conv_state 形状可能不同（P: `num_spec=0`, D: `num_spec>0`）
- 初期建议两端配相同 spec config 避免兼容性问题

## 风险点

1. **Hetero-TP**: KDA 的 conv_q/conv_k/conv_v 大小不同，remote offset 需逐 state 独立处理
2. **float32 recurrent state**: recurrent_state 是 float32，conv states 是 model dtype
3. **DS layout 强制**: 必须 `VLLM_SSM_CONV_STATE_LAYOUT=DS`
4. **Block 不裁剪**: KDA blocks 代表完整 state，partial prefix hit 时不能裁剪
