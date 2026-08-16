#!/bin/bash
# fix_schema_idempotent.sh — vllm_kunlun/schema.py 自定义算子注册幂等化（双份）
# 第 16 雷：MTP 场景 target（qwen3_5 链）与 drafter（qwen3_next）在同一 worker
# 进程内注册同名 vllm::gdn_attention_core → RuntimeError duplicate registration。
# 修复：direct_register_custom_op 的 define/impl/fake 块捕获重复注册并跳过。
set -e
P=/opt/vllm_kunlun/lib/python3.10/site-packages
S=/home/newdata/vLLM-Kunlun-0.25.1-dev
PY=/opt/vllm_kunlun/bin/python

patch_py() {
  "$PY" - "$1" <<'EOF'
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

old = """    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)"""

new = """    my_lib = target_lib or vllm_lib
    try:
        my_lib.define(op_name + schema_str, tags=tags)
        my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)
    except RuntimeError as e:
        msg = str(e)
        if "same name and overload name multiple times" in msg \\
                or "already registered" in msg:
            # Idempotent: the same custom op can be registered by both the
            # target model and the drafter model inside one worker process
            # (MTP / self-speculative decoding). Skip re-registration.
            pass
        else:
            raise"""

assert src.count(old) == 1, f"{path}: pattern not found or not unique (count={src.count(old)})"
src = src.replace(old, new)
open(path, "w", encoding="utf-8").write(src)
print(f"patched: {path}")
EOF
}

for T in "$P/vllm_kunlun/schema.py" "$S/vllm_kunlun/schema.py"; do
  if [ ! -f "$T.bak_rs" ]; then
    cp "$T" "$T.bak_rs" && echo "备份: $T.bak_rs"
  fi
  patch_py "$T"
done

echo "== 验证 1: 补丁后 md5 =="
md5sum $P/vllm_kunlun/schema.py $S/vllm_kunlun/schema.py
echo "== 验证 2: py_compile =="
"$PY" -m py_compile $P/vllm_kunlun/schema.py && echo "语法 OK"
echo "== 验证 3: 幂等化逻辑存在 =="
grep -n "same name and overload name multiple times" $P/vllm_kunlun/schema.py
echo "== 验证 4: import + 注册两次不炸 =="
"$PY" - <<'EOF'
import torch

def op_impl(x: torch.Tensor, layer_name: str) -> None:
    return

def fake_impl(x: torch.Tensor, layer_name: str) -> None:
    return

from vllm_kunlun.schema import direct_register_custom_op
for i in range(2):
    direct_register_custom_op(
        op_name="gdn_attention_core_test",
        op_func=op_impl,
        mutates_args=[],
        fake_impl=fake_impl,
    )
print("double register OK (idempotent)")
EOF
echo "全部通过"
