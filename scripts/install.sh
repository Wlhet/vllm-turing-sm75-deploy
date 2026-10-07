#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 只创建项目内的 .venv/.runtime；不使用 sudo，不安装驱动，不启动模型。
set -euo pipefail
umask 077
unset PYTHONHOME PYTHONPATH
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo 'Install uv first; see docs/DEPLOYMENT.md' >&2; exit 1; }
# 配置由 Python 严格解析；检查无需 GPU 推理，也不会占用模型显存。
python3 scripts/install_overlay.py --preflight
if [[ -e .venv ]]; then
  echo 'Refusing existing .venv. Use a new checkout for a clean install; see docs/DEPLOYMENT.md.' >&2
  exit 1
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --override requirements/overrides.txt \
  --constraint requirements/observed-constraints.txt \
  --requirement requirements/runtime.txt
.venv/bin/python scripts/install_overlay.py
.venv/bin/python scripts/check_environment.py
echo 'Installation finished. Configure .env, then use scripts/manage.py start.'
