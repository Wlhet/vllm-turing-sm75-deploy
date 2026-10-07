# 接入 New API

本项目只提供 OpenAI 兼容上游，不打包 New API、不迁移其数据库，也不自动创建用户或渠道。

## 配置渠道

1. 确保后端直接请求已经通过 `scripts/smoke_test.py`。
2. 在 New API 管理页面创建或复用 OpenAI 类型渠道。
3. 上游地址填写 `http://HOST_IP:8000`；模型填写 `Qwen3.8-27B`。
4. 渠道上游密钥与 `.env` 中 `VLLM_API_KEY` 保持一致。
5. 在 New API 内创建客户端令牌；调用 New API 时使用这个令牌。

客户端 OpenAI Base URL 是 `http://NEWAPI_HOST:3000/v1`。本例不设置模型映射；如果修改服务别名，渠道模型名称必须同步。

## Docker 网络注意事项

容器内的 localhost 是容器自己。New API 容器访问宿主机后端时，需要使用可达的宿主机地址；Linux Docker 可按实际网络配置 host-gateway。

默认 `.env` 仅监听 localhost。需要容器/局域网访问时，在维护窗口将 `HOST` 改成合适的接口地址或 `0.0.0.0`，并配置防火墙，只允许预期的 New API 来源访问 8000。对公网提供服务应使用 TLS、鉴权和网络访问控制，不要裸露推理端口。

## 测试

在目标机本地通过 New API 测试：

```bash
# read -s 避免令牌写入 shell 历史；不要把真实令牌贴进 GitHub Issue。
read -rsp 'New API token: ' NEWAPI_API_KEY; echo
export NEWAPI_API_KEY
python3 scripts/smoke_test.py \
  --base-url http://127.0.0.1:3000/v1 \
  --key-env NEWAPI_API_KEY --model Qwen3.8-27B
unset NEWAPI_API_KEY
```

该命令会产生实际推理请求。远程 URL 需要显式 `--allow-remote`，表示同意把指定令牌发送到该地址；使用前核对主机及 TLS。

不要把上游服务密钥当成 New API 客户端令牌。不要为了接入而直接修改 SQLite 数据库，避免绕过 New API 自身校验或影响其他用户。
