import subprocess
import uuid
from pathlib import Path

import gymnasium as gym
import yaml
from gymnasium import spaces

from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.pytest_parser import PytestParser

TASKS_ROOT = Path.home() / ".cache/terminal-bench/terminal-bench-core/0.1.1"
TEST_DIR = "/tests"


def _docker(*args, input=None, timeout=None):
    return subprocess.run(
        ["docker", *args], input=input, capture_output=True, text=True, timeout=timeout
    )


def is_resolved(parser_results):
    # Mirrors harness._is_resolved: None -> unresolved, else every test must pass
    if parser_results is None:
        return False
    return all(s == UnitTestStatus.PASSED for s in parser_results.values())


class TBEnv(gym.Env):
    """
    mode="shell":  each action is one bash command; "DONE" ends the episode.
    mode="script": each action is a full bash script; one step, then verify.
    Reward is 0.0 on intermediate steps and 1.0/0.0 from the harness rule at the end.
    """

    def __init__(self, task_id, image="tb-hello-world", mode="script",
                 max_steps=30, cmd_timeout=60):
        super().__init__()
        self.task_id = task_id
        self.task_dir = TASKS_ROOT / task_id
        self.image = image
        self.mode = mode
        self.max_steps = max_steps
        self.cmd_timeout = cmd_timeout
        self.action_space = spaces.Text(max_length=8192)
        self.observation_space = spaces.Text(max_length=65536)
        self.parser = PytestParser()
        self.name = None
        self.steps = 0
        self.last_parser_results = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.name = f"tb-rl-{uuid.uuid4().hex[:8]}"
        r = _docker("run", "-d", "--name", self.name, self.image,
                    "sleep", "infinity", timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"Container failed to start: {r.stderr}")
        self.steps = 0
        self.last_parser_results = None
        task = yaml.safe_load((self.task_dir / "task.yaml").read_text())
        return task["instruction"], {}

    def _exec(self, cmd):
        r = _docker("exec", self.name, "bash", "-lc", cmd, timeout=self.cmd_timeout)
        return f"exit={r.returncode}\n{r.stdout}{r.stderr}"

    def _exec_script(self, script):
        # Feed the script on stdin so no temp files are needed in the container
        return _docker("exec", "-i", self.name, "bash", input=script,
                       timeout=self.cmd_timeout)

    def step(self, action):
        self.steps += 1
        if self.mode == "shell":
            if action.strip() == "DONE" or self.steps >= self.max_steps:
                return "", self._verify(), True, False, {}
            try:
                obs = self._exec(action)
            except subprocess.TimeoutExpired:
                obs = "timeout"
            return obs, 0.0, False, False, {}

        # script mode: run the whole script, then the verifier decides
        try:
            self._exec_script(action)
        except subprocess.TimeoutExpired:
            pass  # a timed-out script is still judged by the verifier
        return "", self._verify(), True, False, {}

    def _verify(self) -> float:
        # Copy tests in only now, so the policy never sees them during the episode
        _docker("exec", self.name, "mkdir", "-p", TEST_DIR)
        _docker("cp", f"{self.task_dir / 'tests'}/.", f"{self.name}:{TEST_DIR}")
        _docker("cp", str(self.task_dir / "run-tests.sh"),
                f"{self.name}:{TEST_DIR}/run-tests.sh")
        r = _docker("exec", "-e", f"TEST_DIR={TEST_DIR}", self.name,
                    "bash", f"{TEST_DIR}/run-tests.sh", timeout=120)
        try:
            results = self.parser.parse(r.stdout + r.stderr)
        except Exception:
            results = None  # harness treats a parse failure as unresolved
        self.last_parser_results = results
        return 1.0 if is_resolved(results) else 0.0

    def close(self):
        if self.name:
            _docker("rm", "-f", self.name)
            self.name = None
