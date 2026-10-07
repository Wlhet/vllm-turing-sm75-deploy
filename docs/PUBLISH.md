# 上传 GitHub 前的检查

## 发布内容

只发布源码、配置模板、许可证、文档、单元测试和 CI。不要发布：

- `.env`、服务密钥、SSH 密码、New API 令牌/数据库。
- 模型权重、`.venv`、缓存、日志、PID、二进制扩展、wheel。
- 旧部署脚本中的个人路径和硬编码密钥。
- 旧仓库 `.git` 历史；当前包从固定提交的选定文件生成。

`.gitignore` 不是绝对安全保障；`git add -f` 和已经受追踪的文件能够绕过它。

## 验证

```bash
python3 -B -m unittest discover -s tests -v
bash -n scripts/install.sh
# 如果尚未初始化 Git：git init -b main
git add .
python3 -B scripts/verify_repo.py --tracked
git diff --cached --check
git status --short
```

查看待提交文件，确认 `.env` 没有出现。扫描是启发式防线，不是“不可能泄漏”的保证，发布前仍需人工复核。

## 首次推送

先在 GitHub 创建一个空仓库，例如 `vllm-turing-sm75-deploy`。作者信息使用你自己的真实配置，不要让工具代为编造身份。

```bash
# 仅在尚未配置提交身份时，设置当前仓库的 user.name / user.email。
git commit -m "初始化 SM75 原生部署项目" \
  -m "固定 vllm-Turing 上游提交；整理独立安装、启动管理、配置模板与 New API 文档；加入来源追踪、无 GPU 测试和发布安全检查。"
git remote add origin git@github.com:YOUR_GITHUB_USER/vllm-turing-sm75-deploy.git
git push -u origin main
```

本项目整理操作不会创建 GitHub 仓库、配置远端凭据或自动推送。如果已有 origin，先核对 URL，不要盲目覆盖。

## 源码压缩包

若提供配套 source ZIP，它只由已审查的 Git 暂存文件生成，不包含 `.git`、`.env` 或运行环境。解压后可重新 `git init`；不要直接压缩整个含私有配置的部署目录上传。

## 更新上游

选定新的上游提交，审查差异，更新 vendor 与 `UPSTREAM.json`，同步许可证/NOTICE。重新检查依赖、锚点、单元测试及 GPU 端到端验证。不要用无审查的 `git pull` 自动覆盖运行环境。
