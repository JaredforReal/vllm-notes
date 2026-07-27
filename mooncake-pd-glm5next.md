# GLM5Next 300B — Mooncake PD 分离 setup

单机 8×B300 (sm100) 上,P (prefill, GPU 0–3) + D (decode, GPU 4–7) + router,Mooncake connector。
GLM5Next 的 hybrid 布局(MLA + KDA mamba 共享 tensor)**不需要改代码**——Mooncake 的 storage 级注册 + group 感知 传输天然兼容。

## 前置

```bash
# 已安装(本次):
uv pip install mooncake-transfer-engine-cuda13

# 模型路径
MODEL=/opensource/yuxuan/glm-5-next/300b/hf
```

环境变量说明:
- `VLLM_SSM_CONV_STATE_LAYOUT=DS` — KDA/mamba conv state 用 DS 布局(3-read 传输要求)。
- `VLLM_KV_CACHE_LAYOUT=HND` — KV cache HND 布局。
- 不需要 NIXL 的 `UCX_NET_DEVICES` / `VLLM_NIXL_SIDE_CHANNEL_PORT`(那是 NIXL 专用的)。
- Mooncake bootstrap server 由 producer 的 TP-rank0 自动起在 `0.0.0.0:8998`(默认 `VLLM_MOONCAKE_BOOTSTRAP_PORT=8998`),consumer 经 router 拿到 bootstrap addr 后连过去。

## 1. 启动 Producer (P, prefill)

后台运行,日志写 `/tmp/mc_prod.log`:

```bash
nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_SSM_CONV_STATE_LAYOUT=DS \
VLLM_KV_CACHE_LAYOUT=HND \
.venv/bin/vllm serve /opensource/yuxuan/glm-5-next/300b/hf \
  --trust-remote-code \
  --port 8100 --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\"}" \
  --served-model-name glm \
  --no-async-scheduling \
  --block-size 1024 \
  --max-num-seqs 512 \
  --attention-backend FLASHMLA_SPARSE
' > /tmp/mc_prod.log 2>&1 &
```

前台调试可直接去掉 `nohup bash -c '...'` 包裹,把环境变量和 vllm 命令平铺。

等价的前台写法:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_KV_CACHE_LAYOUT=HND \
.venv/bin/vllm serve /opensource/yuxuan/glm-5-next/300b/hf \
  --trust-remote-code --port 8100 --host 0.0.0.0 --tensor-parallel-size 4 \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' \
  --served-model-name glm --no-async-scheduling --block-size 1024 \
  --max-num-seqs 512 --attention-backend FLASHMLA_SPARSE
```

## 2. 启动 Consumer (D, decode)

```bash
nohup bash -c '
CUDA_VISIBLE_DEVICES=4,5,6,7 \
VLLM_SSM_CONV_STATE_LAYOUT=DS \
VLLM_KV_CACHE_LAYOUT=HND \
.venv/bin/vllm serve /opensource/yuxuan/glm-5-next/300b/hf \
  --trust-remote-code \
  --port 8200 --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\"}" \
  --served-model-name glm \
  --no-async-scheduling \
  --max-num-seqs 512 \
  --block-size 1024 \
  --attention-backend FLASHMLA_SPARSE
' > /tmp/mc_cons.log 2>&1 &
```

## 3. 启动 Router(mooncake 模式)

```bash
.venv/bin/vllm-router \
  --policy round_robin \
  --vllm-pd-disaggregation \
  --kv-connector mooncake \
  --prefill http://127.0.0.1:8100 \
  --decode http://127.0.0.1:8200 \
  --host 0.0.0.0 \
  --port 8000 \
  --intra-node-data-parallel-size 1
```

> **关键坑**:router **默认 `--kv-connector nixl`**。如果忘了加 `--kv-connector mooncake`,router 会发 NIXL 风格的 `kv_transfer_params`(不带 `transfer_id`),Mooncake 会打 `Missing transfer_id in kv_transfer_params from router!` 并**不做 KV 迁移**——D 侧退回本地 prefill,输出仍然连贯但不是真正的 PD。router 日志里 `kv_connector: Nixl` / `kv_connector: Mooncake` 可一眼区分。

## 4. 测试

```bash
# 经 router(8000)发请求,prefill→8100,decode→8200,KV 经 Mooncake 迁移
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  --data '{"model":"glm","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
```

## 5. 验证 KV 真的迁移了

```bash
# Producer:应看到 successful transfers>0,failed=0
grep "KV Transfer metrics" /tmp/mc_prod.log | tail -1

# Consumer:External prefix cache hit rate > 0 说明从 P 拉到了 KV
grep "External prefix cache hit rate" /tmp/mc_cons.log | tail -1
```

本次实测指标:
- Producer:`Num successful transfers=4, Avg xfer time=25.2ms, Throughput=31589 MB/s, Num failed transfers=0`
- Consumer:`External prefix cache hit rate: 40.0%`

## 关停

```bash
# 停 router + 两个 vllm 实例(及其 worker 子进程)
pkill -9 -f "vllm-router"
ps -ef | grep -E "vllm serve|EngineCore|Worker_TP" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

## 备注

- Mooncake 对 mamba/GDN 的 recurrent (ssm) 状态**不传**,而是 P 侧截断最后一个 prompt token、D 侧重算该 token  来重建 ssm(`_truncate_mamba_request_for_prefill`)。所以 KDA 层的正确性依赖这个重算路径——需用 gsm8k / 识图做精 度验证。
- 与 NIXL 的区别:NIXL 传 conv+ssm(需 `_mamba_region_indices` 修复才能 init);Mooncake 只传 conv + 重算 ssm,且 storage 级注册天然避开 NIXL 的 dedup/overflow 坑。
