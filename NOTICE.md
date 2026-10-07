# 来源、归属与本地变更

本项目不是从零实现的推理框架，也不声称拥有上游优化的原创权。

## 主要来源

1. **vLLM** — https://github.com/vllm-project/vllm
   - 基础版本：0.30.0。
   - 上游基线提交：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
   - Apache-2.0；根目录 `LICENSE` 保留从 vllm-Turing 继承的许可证。
2. **vllm-Turing** — https://github.com/javier-house/vllm-Turing
   - 分支：`memcpypeer0.30.0`。
   - 固定提交：`37763f17f38a7b7cb26ab4fd17b8c11b5e945b27`。
   - 本项目只收录原生部署需要的 overlay、安装器及来源文件，没有收录完整仓库或完整官方 vLLM。
3. **FlashQLA SM75** — 来自 https://github.com/1CatAI/1Cat-vLLM
   - 提交：`187b932dbd11940f0bcf52fb3675dd47fd69f313`。
   - 原始目录：`flash_qla/ops/gated_delta_rule/chunk/sm70/`。
   - MIT 许可证及完整集成说明保留在 `vendor/vllm-turing/vllm/third_party/flash_qla_sm75/`。

## 本项目新增内容

项目专属原生安装流程、配置与进程管理、环境偏离检查、安全发布检查、无 GPU 单元测试、CI 和中文文档。新增代码以 Apache-2.0 分发。

## 对收录上游文件的修改

- `docker/patch_transformers_startup.py`：从安装包元数据定位 Transformers，替代 Docker 固定目录；允许已完成的精确替换重复运行。
- `vllm/entrypoints/serve/instrumentator/test.html`：将上游示例中的私人网段地址替换为 localhost，并清理行尾空白。

其他收录文件保持所标注提交的字节内容。`UPSTREAM.json` 记录全部原始 SHA-256、修改原因及修改后 SHA-256。更新 vendor 时应同步来源、差异说明和校验和，不能静默漂移。

API 名称 `Qwen3.8-27B` 是部署别名，不构成对模型发布来源、商标或权重真实性的认证。
