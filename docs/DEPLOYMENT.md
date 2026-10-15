# 原生部署与恢复

## 1. 前提与边界

验证基线为 Linux x86_64、4×Tesla T10 16GB（SM75）、Python 3.12、驱动 595.91.07、CUDA toolkit 13.1、GCC/G++ 13。这些是 0.31 目标机验证时的版本，不是推导出的最低版本。其他驱动组合需另行验证。

本项目不执行系统级驱动/工具链安装。开始前检查：

```bash
nvidia-smi
/usr/local/cuda-13.1/bin/nvcc --version
gcc-13 --version
g++-13 --version
uv --version
df -h .
```

4 张 GPU 必须可见。Firefly 会按实际拓扑选择 P2P 或 SHM；不要仅凭显卡型号假设 P2P 可用。模型文件要完整并有 `config.json`，当前方案要求模型结构及 MTP 权重与原部署相符；不能只改目录名套用任意 27B。

## 2. 依赖约束与已知冲突

- `runtime.txt` 固定主要版本；flashinfer-cubin 以带 SHA256 的官方发布 wheel 固定。
- `observed-constraints.txt` 是目标机 0.31 venv 的 204 项分发包版本快照，仅作为依赖解析约束，不携带本机路径、索引凭据或 wheel 文件。
- `overrides.txt` 固定 `flashinfer-python==0.7.0.post1`，与 vLLM 0.31 元数据要求一致；**0.31 基线没有依赖偏离**。
- `scripts/check_environment.py` 不豁免任何版本组合，任何依赖冲突或主版本偏离都直接失败。

快照不含 wheel 哈希，也不保证未来索引仍保留所有构建包。需要供应链可重现性时，应另行建立经过许可的内部 wheel 镜像及哈希清单；本仓库不会上传二进制依赖。

如需下载代理，操作者可以在当前终端设置 `HTTPS_PROXY`/`HTTP_PROXY`；不要把带凭据的索引或代理地址写进仓库。

## 3. 安装步骤

1. 复制 `.env.example` 为 `.env`，设置模型绝对路径和随机密钥，权限设为 600。
2. 检查 `CUDA_HOME`、`CC`、`CXX`、`GPU_IDS`。
3. 执行 `bash scripts/install.sh`。
4. 运行 `python3 scripts/manage.py print-command` 查看不含密钥的参数。
5. 在 GPU 无冲突、端口空闲时执行 `start`、`wait` 和 smoke test。

配置文件只支持白名单 `KEY=value` 和引号，不执行 shell，不展开 `$HOME`、`${VAR}` 或命令替换。路径可用 `~`；相对路径基于项目根目录。环境变量优先于 `.env`。

新仓库的 `.venv`、扩展、进程状态和缓存完全独立；不要软链接既有部署的虚拟环境，也不要让两个实例争用同一组 GPU。

## 4. 失败与重试

安装脚本遇到现有 `.venv` 会停止，避免在未知环境上重复覆盖。

- 下载阶段失败：优先在另一个新 checkout 重新安装，不自动删除旧环境。
- 已确认依赖安装完整、仅 overlay/编译阶段失败：先停止该项目的推理进程，检查原因后可运行：

```bash
.venv/bin/python scripts/install_overlay.py
.venv/bin/python scripts/check_environment.py
```

修补脚本校验基础版本和锚点，拒绝不匹配版本。更新依赖后不得沿用旧的成功标记推断补丁仍在，必须重新确认安装证据与 API 功能。

## 5. 服务管理与可选自启

`manage.py start` 使用独立会话后台启动，默认没有守护进程。`stop` 发出 SIGTERM，超时不会自动 SIGKILL；停止可能中断请求，应先进入维护窗口。

`deploy/vllm-turing.service.example` 是未安装的模板。由管理员修改 `User`、`WorkingDirectory` 和 `ExecStart` 后，按系统管理流程安装；该操作会改变系统配置，应另行审批。systemd 使用前台 `manage.py run`，用 `systemctl` 管理生命周期；它不写后台模式的 PID 文件，因此不要用 `manage.py stop/status` 管理这个实例，健康检查仍可用。

## 6. 调优提示

- `MTP_TOKENS=0` 关闭 MTP；默认 3。
- `FIREFLY_DIRECT=0` / `FIREFLY_AR=0` 可分别关闭计算/通信优化，需重启生效。
- 0.92 是显存预算比例，不是所有负载下的绝对显存上限保证。
- `MAX_NUM_SEQS=4` 不等于四条 256K 满长请求能够并发驻留。
- 增大上下文或并发之前先验证峰值显存、长请求稳定性与吞吐。
- 性能比较需同模型、同输入、同并发、同缓存状态和维护窗口，不能比较不同请求的瞬时 tokens/s。
