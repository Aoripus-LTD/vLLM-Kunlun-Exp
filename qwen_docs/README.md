# qwen_docs — Qwen3.8-27B on Kunlun P800 部署文档站

MkDocs Material 中文文档站，内容为昆仑芯 P800 部署 Qwen3.8-27B W8A8 INT8 的完整手册。

## 本地预览

```bash
cd qwen_docs
python -m venv .venv                    # 或任意 python 环境
pip install -r requirements.txt
mkdocs serve                            # http://127.0.0.1:8000
```

## 构建静态站点

```bash
mkdocs build                            # 输出 site/ 目录
```

## 部署方式

静态站点（`site/` 目录）可直接部署到任意静态托管：

- **GitHub Pages**：`mkdocs gh-deploy` 或 Actions 推送 `site/`
- **Nginx/任意 Web 服务器**：上传 `site/` 即可
- **Read the Docs**：项目根添加 `.readthedocs.yaml` 指向 `qwen_docs/` 目录

## 目录结构

```text
qwen_docs/
├── mkdocs.yml            # MkDocs 配置（Material 主题）
├── requirements.txt      # 依赖（mkdocs + material）
└── docs/
    ├── index.md          # 首页（状态 + 导航）
    ├── overview.md       # 项目概览（规格/版本/关键认知）
    ├── quickstart.md     # 快速开始（容器 → 安装 → 验证）
    ├── installation.md   # 安装详解
    ├── model.md          # 模型与适配（架构/量化/config）
    ├── launch.md         # 启动参数（权威参数原理）
    ├── performance.md    # 性能与压测（XCCL/带宽/形态决策）
    ├── troubleshooting.md# 故障排查
    └── assets.md         # 部署资产（qwen38-deploy/ 脚本）
```

## 内容维护

素材来源：内部部署记录与 03/04 验证脚本。更新文档后按仓库规范提交（qwen38-dev
分支）。
