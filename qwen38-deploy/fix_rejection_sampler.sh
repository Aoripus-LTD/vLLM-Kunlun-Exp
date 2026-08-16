#!/bin/bash
# fix_rejection_sampler.sh — 部署拼装版 RejectionSampler 到容器双份
# 用法: bash fix_rejection_sampler.sh <patches/rejection_sampler.py>
# 双份: site-packages（import 实际加载）+ /home/newdata/vLLM-Kunlun-0.25.1-dev 仓库根
# 各备份 .bak_rs
set -e
SRC=${1:?用法: bash fix_rejection_sampler.sh <patches/rejection_sampler.py>}
P=/opt/vllm_kunlun/lib/python3.10/site-packages
S=/home/newdata/vLLM-Kunlun-0.25.1-dev
PY=/opt/vllm_kunlun/bin/python

echo "== 0. 源文件 =="
wc -l "$SRC"
grep -n "triton\|tl\." "$SRC" && echo "!! 源文件含 triton 引用，拒绝部署" && exit 1 || true

echo "== 1. 备份双份（.bak_rs） =="
for T in "$P/vllm_kunlun/v1/sample/rejection_sampler.py" "$S/vllm_kunlun/v1/sample/rejection_sampler.py"; do
  if [ -f "$T" ]; then
    if [ ! -f "$T.bak_rs" ]; then
      cp "$T" "$T.bak_rs" && echo "备份: $T.bak_rs"
    else
      echo "备份已存在（跳过）: $T.bak_rs"
    fi
  else
    echo "!! 缺失: $T"; exit 1
  fi
done

echo "== 2. 覆盖双份 =="
cp "$SRC" "$P/vllm_kunlun/v1/sample/rejection_sampler.py"
cp "$SRC" "$S/vllm_kunlun/v1/sample/rejection_sampler.py"
echo "== 3. md5 三处一致校验 =="
md5sum "$SRC" "$P/vllm_kunlun/v1/sample/rejection_sampler.py" "$S/vllm_kunlun/v1/sample/rejection_sampler.py"

echo "== 4. py_compile 语法检查 =="
"$PY" -m py_compile "$P/vllm_kunlun/v1/sample/rejection_sampler.py" && echo "语法 OK"

echo "== 5. import + 接口签名检查 =="
"$PY" - <<'EOF'
import inspect
import vllm_kunlun.v1.sample.rejection_sampler as rs
print("module:", rs.__file__)
print("ctor:", inspect.signature(rs.RejectionSampler.__init__))
print("forward:", inspect.signature(rs.RejectionSampler.forward))
print("parse_output:", inspect.signature(rs.RejectionSampler.parse_output))
print("rejection_sample:", inspect.signature(rs.rejection_sample))
print("sample_recovered_tokens:", inspect.signature(rs.sample_recovered_tokens))
print("expand_batch_to_tokens:", inspect.signature(rs.expand_batch_to_tokens))
print("generate_uniform_probs:", inspect.signature(rs.generate_uniform_probs))
# triton 依赖检查
src = open(rs.__file__).read()
print("triton refs:", "triton" in src or "tl." in src)
# 构造验证（mock sampler）
class MockSampler:
    logprobs_mode = "raw_logits"
    def compute_logprobs(self, x): return x
    def gather_logprobs(self, x, n, ids): return None
    def __call__(self, **kw):
        from vllm.v1.outputs import SamplerOutput
        import torch
        return SamplerOutput(sampled_token_ids=torch.zeros(kw["logits"].shape[0], dtype=torch.int32, device=kw["logits"].device))
rs.RejectionSampler(MockSampler())
print("ctor OK")
EOF
echo "全部通过"
