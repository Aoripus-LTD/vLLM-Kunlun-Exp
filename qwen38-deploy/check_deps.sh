#!/bin/bash
# check_deps.sh — 容器内检查 web 层依赖实际版本（定位 instrumentator/fastapi 不匹配）
/opt/vllm_kunlun/bin/python - <<'PYEOF'
import importlib.metadata as md
for p in ("fastapi", "prometheus-fastapi-instrumentator", "starlette",
          "uvicorn", "openai", "pydantic", "prometheus-client"):
    try:
        print(p, "=", md.version(p))
    except Exception as e:
        print(p, "MISSING:", e)
PYEOF
