# SPDX-License-Identifier: Apache-2.0
"""无 GPU 单元测试：配置、参数、安全边界和 Linux 进程生命周期。"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import common
import manage
import verify_repo


def config():
    cfg = dict(common.DEFAULTS)
    cfg.update(MODEL_PATH="/tmp/test-model", VLLM_API_KEY="unit-test-placeholder-not-real")
    return cfg


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        common.validate(config())

    def test_missing_secret_rejected(self):
        cfg = config(); cfg["VLLM_API_KEY"] = "replace-with-a-random-secret"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_model_placeholder_rejected(self):
        cfg = config(); cfg["MODEL_PATH"] = "/path/to/model"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_mtp_range(self):
        for value in ("-1", "9"):
            cfg = config(); cfg["MTP_TOKENS"] = value
            with self.assertRaises(ValueError): common.validate(cfg)

    def test_memory_range(self):
        for value in ("0", "1", "nan", "inf"):
            cfg = config(); cfg["GPU_MEMORY_UTILIZATION"] = value
            with self.assertRaises(ValueError): common.validate(cfg)

    def test_gpu_count(self):
        cfg = config(); cfg["GPU_IDS"] = "0,1"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_duplicate_gpu(self):
        cfg = config(); cfg["GPU_IDS"] = "0,1,2,2"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_port_range(self):
        cfg = config(); cfg["PORT"] = "65536"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_threshold(self):
        cfg = config(); cfg["LONG_PREFILL_TOKEN_THRESHOLD"] = "8192"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_backend(self):
        cfg = config(); cfg["FIREFLY_AR_BACKEND"] = "unknown"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_env_overrides_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config"
            path.write_text('PORT=9000\nMODEL_PATH="/tmp/path with spaces"\n', encoding="utf-8")
            cfg = common.load_config(path, {"PORT": "8001"})
            self.assertEqual(cfg["PORT"], "8001")
            self.assertEqual(cfg["MODEL_PATH"], "/tmp/path with spaces")

    def test_no_shell_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config"
            path.write_text("MODEL_PATH='$(do-not-execute)'\n", encoding="utf-8")
            self.assertEqual(common.load_config(path, {})["MODEL_PATH"], "$(do-not-execute)")

    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config"; path.write_text("UNKNOWN=value\n")
            with self.assertRaises(ValueError): common.load_config(path, {})

    def test_mtp_command(self):
        cmd = common.build_command(config())
        spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
        self.assertEqual(spec, {"method": "mtp", "num_speculative_tokens": 3})
        graph = json.loads(cmd[cmd.index("--compilation-config") + 1])
        # 目标机实机参数：(MTP+1) × B1..B4 连续捕获。
        self.assertEqual(graph["cudagraph_capture_sizes"], [4, 8, 12, 16])

    def test_mtp_scheduler_is_fair031(self):
        cmd = common.build_command(config())
        self.assertEqual(cmd[cmd.index("--scheduler-cls") + 1],
                         "scheduler_sm75_fair031.SM75FairScheduler031")

    def test_served_model_names(self):
        cfg = config()
        self.assertEqual(common.build_command(cfg).count("--served-model-name"), 1)
        cfg["SERVED_MODEL_NAMES"] = "Qwen3.8-27B Nvidia-Qwen3.8-27B"
        cmd = common.build_command(cfg)
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "Qwen3.8-27B")
        self.assertEqual(cmd[cmd.index("--served-model-name", 5) + 1], "Nvidia-Qwen3.8-27B")

    def test_quantization_flag(self):
        cmd = common.build_command(config())
        self.assertEqual(cmd[cmd.index("--quantization") + 1], "modelopt_mixed")
        cfg = config(); cfg["QUANTIZATION"] = ""
        self.assertNotIn("--quantization", common.build_command(cfg))

    def test_quantization_looks_like_flag_rejected(self):
        cfg = config(); cfg["QUANTIZATION"] = "--x"
        with self.assertRaises(ValueError): common.validate(cfg)

    def test_mtp_off(self):
        cfg = config(); cfg["MTP_TOKENS"] = "0"
        cmd = common.build_command(cfg)
        self.assertNotIn("--speculative-config", cmd)
        self.assertNotIn("--scheduler-cls", cmd)
        self.assertEqual(common.runtime_env(cfg)["VLLM_SPEC_DECODE"], "0")

    def test_secret_not_in_command(self):
        cfg = config()
        self.assertNotIn(cfg["VLLM_API_KEY"], " ".join(common.build_command(cfg)))
        self.assertEqual(common.runtime_env(cfg)["VLLM_API_KEY"], cfg["VLLM_API_KEY"])

    def test_python_environment_isolation(self):
        with mock.patch.dict(os.environ, {"PYTHONHOME": "/bad", "PYTHONPATH": "/bad"}):
            env = common.runtime_env(config())
        self.assertNotIn("PYTHONHOME", env)
        # 只有项目内 compat 目录可覆盖上游模块。
        self.assertEqual(env["PYTHONPATH"], str(ROOT / "compat"))
        self.assertEqual(env["SM75_FA2_SMALLQ_GRAPH"], "1")

    def test_health_wildcard(self):
        cfg = config(); cfg["HOST"] = "0.0.0.0"
        self.assertEqual(common.health_url(cfg), "http://127.0.0.1:8000")

    def test_health_ipv6(self):
        cfg = config(); cfg["HOST"] = "::"
        self.assertEqual(common.health_url(cfg), "http://[::1]:8000")


class SecurityTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "Git required for ignore-rule regression")
    def test_model_weights_ignored_but_model_source_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(["git", "init", "-q", temp], check=True)
            shutil.copyfile(ROOT / ".gitignore", Path(temp) / ".gitignore")
            for path, ignored in [("models/model.safetensors", True),
                                  ("vendor/vllm-turing/vllm/models/qwen4_exp/nvidia/qsa.py", False),
                                  ("vendor/vllm-turing/docker/speculative/vllm/model_executor/models/qwen3_dflash.py", False)]:
                result = subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=temp)
                self.assertEqual(result.returncode, 0 if ignored else 1)

    def test_env_excluded(self):
        self.assertIn("private environment file", verify_repo.problems(Path(".env"), b"A=B"))

    def test_weights_excluded(self):
        self.assertIn("runtime/binary/secret artifact", verify_repo.problems(Path("model.safetensors"), b"x"))

    def test_synthetic_key_detected(self):
        secret = ("sk" + "-" + "x" * 32).encode()
        self.assertIn("token-shaped string", verify_repo.problems(Path("readme.md"), secret))

    def test_stale_identity_never_stopped(self):
        with mock.patch.object(manage, "process_identity", return_value=None):
            self.assertFalse(manage.owned({"pid": 123}))

    def test_reused_pid_never_owned(self):
        with mock.patch.object(manage, "process_identity", return_value={"pid": 123, "start_ticks": "new", "boot_id": "a"}):
            self.assertFalse(manage.owned({"pid": 123, "start_ticks": "old", "boot_id": "a"}))

    def test_transformers_patch_is_idempotent(self):
        # 静态执行只依赖 pathlib 的 replace_once，避免本机需要安装 transformers。
        import ast
        tree = ast.parse((ROOT / "vendor/vllm-turing/docker/patch_transformers_startup.py").read_text(encoding="utf-8"))
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "replace_once")
        scope = {"Path": Path}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "patch", "exec"), scope)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.py"; path.write_text("before")
            scope["replace_once"](path, "before", "after")
            scope["replace_once"](path, "before", "after")
            self.assertEqual(path.read_text(), "after")
            with self.assertRaises(RuntimeError): scope["replace_once"](path, "missing", "other")


@unittest.skipUnless(sys.platform == "linux", "Linux /proc and process groups required")
class LifecycleTests(unittest.TestCase):
    def test_start_duplicate_and_stop_dummy_process(self):
        # 仅启动标准库 sleep 子进程；绝不加载 vLLM 或访问 GPU。
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            with mock.patch.object(manage, "ROOT", Path(temp)), mock.patch.object(manage, "RUNTIME", runtime), \
                 mock.patch.object(manage, "runtime_env", return_value={**os.environ, "SM75_PROJECT_ROOT": temp}), \
                 mock.patch.object(manage, "STATE", runtime / "server.json"), \
                 mock.patch.object(manage, "validate"), mock.patch.object(manage, "port_available"), \
                 mock.patch.object(manage, "build_command", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]):
                try:
                    manage.start(config())
                    state = manage.read_state()
                    self.assertTrue(manage.owned(state))
                    with self.assertRaises(RuntimeError): manage.start(config())
                    manage.stop(5)
                    self.assertFalse(manage.owned(state))
                    manage.stop(1)
                finally:
                    state = manage.read_state()
                    if manage.owned(state):
                        os.killpg(state["pid"], signal.SIGTERM)

    def test_port_collision_detected(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0)); server.listen()
            cfg = config(); cfg["PORT"] = str(server.getsockname()[1])
            with self.assertRaises(OSError): manage.port_available(cfg)


if __name__ == "__main__":
    unittest.main()
