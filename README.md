# vLLM Turing SM75 · 原生部署项目

面向 **4×Tesla T10 16GB / SM75** 的可审查、可复现部署工程：在官方 **vLLM 0.31.0** 上应用固定版本的 [vllm-Turing](https://github.com/javier-house/vllm-Turing/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6) overlay，运行本地 NVFP4/FP8 混合量化模型，并提供 OpenAI 兼容 API。

**这是社区部署集成项目，不是官方 vLLM 发行版，也不包含模型权重。** CUDA 内核与主要兼容补丁来自上游；本项目新增原生安装、配置隔离、服务管理、验证脚本及中文文档。

## 已固定的基线

- 官方 vLLM：`0.31.0`。
- vllm-Turing 固定提交：`db9527a46873454610df6dbedf79a36d6bf1a7f6`（以提交为唯一锚点，不断言分支名）。
- Python：`3.12`；torch：`2.13.0`；Transformers：`5.15.1`。
- FlashInfer：`0.7.0.post1`（含 flashinfer-cubin 0.7.0.post1），与 0.31 元数据要求一致，无依赖偏离。
- 验证硬件：4×Tesla T10 16GB；原环境使用 CUDA toolkit 13.1、GCC/G++ 13。

完整来源及逐文件 SHA-256 见 [`UPSTREAM.json`](UPSTREAM.json)。依赖快照不是跨平台、带 wheel 哈希的锁文件；详见 [部署说明](docs/DEPLOYMENT.md)。

## 默认推理方案

- 模型 API 名：`Qwen3.8-27B`，可通过 `SERVED_MODEL_NAMES` 配置多个别名。
- 本地模型：NVFP4/FP8 混合或纯 NVFP4 量化目录，由用户自行准备；混合结构默认 `--quantization modelopt_mixed`，纯 NVFP4 用 `modelopt_fp4`，其余可留空让 vLLM 自动识别。
- TP4，FP16 计算，FP8 E4M3 KV，GDN recurrent state 使用 FP32。
- 最大上下文 `262144`；显存预算 `0.92`。
- MTP 草稿数量 `3`；前缀缓存、异步调度、分块 prefill。
- B8 公平调度（0.31 队列适配）与 B1–B4 小请求 CUDA Graph 捕获，来自 `compat/` 本地适配模块。
- Firefly NVFP4 大输入 INT8 计算路径、Firefly All-Reduce。
- FlashQLA-SM75 GDN prefill；FlashInfer attention。
- 默认监听 `127.0.0.1:8000`，强制配置 API key；监控页和测速页默认关闭。

> 这些是已有部署的配置与经验，不是任意 SM75 显卡和任意模型都能直接使用的承诺。MTP3 不等于三倍速度，256K 不代表四条满长请求可以同时驻留。

## 快速开始

需要 Linux x86_64、可用的 NVIDIA 驱动、CUDA toolkit（包含 `nvcc`）、GCC/G++ 13、Python 3.10+ 用于管理脚本，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。安装脚本会创建 Python 3.12 的项目专属虚拟环境。

```bash
# 克隆你的仓库后进入项目目录。
cp .env.example .env
chmod 600 .env
# 编辑 .env：必须设置真实 MODEL_PATH 和 VLLM_API_KEY。
# 可使用 python3 -c 'import secrets; print(secrets.token_urlsafe(32))' 生成新密钥。

bash scripts/install.sh
python3 scripts/manage.py print-command
python3 scripts/manage.py start
python3 scripts/manage.py wait --timeout 1800
python3 scripts/smoke_test.py
```

- 安装仅写入本项目 `.venv/`、`.runtime/` 及 uv 自身用户缓存；不使用 sudo，不修改驱动，不启动模型。
- 安装需联网下载依赖，首次编译可能较慢；准备足够磁盘空间。
- `start` 表示进程已启动，**不等于模型已就绪**；请运行 `wait` 和 smoke test。
- 已存在 `.venv` 时安装脚本拒绝覆盖；不要将其他部署的环境软链接到这里。
- 首次使用的边界及故障恢复见 [DEPLOYMENT.md](docs/DEPLOYMENT.md)。

## 日常管理

```bash
python3 scripts/manage.py status
python3 scripts/manage.py health
python3 scripts/manage.py logs

# 停止会中断当前请求，请先安排维护窗口。
python3 scripts/manage.py stop --timeout 120

# 修改 .env 后，在维护窗口 stop，再 start、wait。
```

服务在后台运行，SSH 断开不会因终端挂断而退出；默认**没有开机自启或崩溃自重启**。可选的 [systemd 模板](deploy/vllm-turing.service.example) 需管理员审核后手动安装，不能与后台管理入口混用。

管理器使用锁、PID 启动时钟、系统 boot ID 和项目进程标识，拒绝误停其他项目进程。API key 通过环境变量传递，不写入启动命令和进程状态文件。

## New API 接入

在 New API 中添加或编辑 OpenAI 类型渠道：

- 模型：`Qwen3.8-27B`。
- 上游地址：`http://HOST_IP:8000`，不要在渠道地址重复追加 `/v1`。
- 上游密钥：与本项目 `.env` 的 `VLLM_API_KEY` 一致。
- 客户端地址：`http://NEWAPI_HOST:3000/v1`。
- 客户端使用 **New API 令牌**，不是上游服务密钥。

Docker 容器的 `127.0.0.1` 不等于宿主机；跨容器接入、TLS、防火墙及测试方式见 [NEWAPI.md](docs/NEWAPI.md)。本项目不自动读取或修改 New API 数据库。

## 项目结构

```text
.env.example                 无密钥的参数模板
scripts/                     安装、启动管理、环境检查、API测试、发布检查
requirements/                主依赖、显式覆盖、历史版本快照
compat/                      项目本地 SM75 适配器（PYTHONPATH 覆盖上游同名模块）
vendor/vllm-turing/           固定提交的必要 overlay 和 CUDA 源码
UPSTREAM.json                来源、原始校验和、本地改动记录
LICENSE / NOTICE.md          许可证及归属
deploy/                      可选 systemd 模板
.github/workflows/check.yml  无 GPU 的持续集成检查
tests/                       配置、隔离、安全、进程生命周期测试
docs/                        部署、New API、优化、验证边界、发布说明
```

没有复制上游 Git 历史、Docker 示例、截图、模型、二进制扩展、缓存或真实部署日志。`vendor` 是构建输入，**不是完整的官方 vLLM 源码树**；基础包由安装流程获取。

## 验证与发布

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/verify_repo.py
bash -n scripts/install.sh
```

私有 `.env` 配好之后，全工作树扫描会故意拒绝它；发布时只审查暂存的公开文件：

```bash
git add .
python3 -B scripts/verify_repo.py --tracked
git diff --cached --check
```

不要使用 `git add -f .env`。公开仓库准备与上传步骤见 [PUBLISH.md](docs/PUBLISH.md)，安全边界见 [SECURITY.md](SECURITY.md)。

### 验证范围

原部署完成过健康检查、普通/流式生成、MTP 接受计数、约 45K 输入请求；0.31 源码线已在目标 4×T10 自 2026-10 起生产运行（NVFP4 27B，MTP3）；本项目的整理验收覆盖无 GPU 单测、源码与发布安全检查。**没有把这个新目录从零重新下载依赖、重新编译并跑完整 GPU 推理，也没有做 256K 满长压力测试或受控性能 A/B。**

不要将原部署的历史成功测试当成这个发布包已完成全新机器端到端验收。详情见 [VALIDATION.md](docs/VALIDATION.md)。

## 来源与许可证

遵循 Apache-2.0，保留上游声明；FlashQLA 子目录保留其 MIT 许可证与来源记录。详见 [NOTICE.md](NOTICE.md)。模型权重、第三方依赖及其许可证不包含在本源码发行包中，也不由本项目重新授权。
