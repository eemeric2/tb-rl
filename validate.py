from pathlib import Path

from tb_env import TBEnv, TASKS_ROOT

TASK = "hello-world"
SOLUTION = (TASKS_ROOT / TASK / "solution.sh").read_text()

CASES = [
    ("oracle replay", SOLUTION, 1.0),
    ("do nothing", "", 0.0),
    ("wrong content", "printf 'wrong\\n' > /app/hello.txt\n", 0.0),
    ("extra file (benchmark gap)", SOLUTION + "\ntouch /app/extra.txt\n", 1.0),
]


def run_case(name, script, expected):
    env = TBEnv(TASK, mode="script")
    try:
        env.reset()
        _, reward, terminated, _, _ = env.step(script)
        ok = reward == expected and terminated
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: reward={reward} expected={expected}")
        if env.last_parser_results is not None:
            for test, result in env.last_parser_results.items():
                print(f"        {test}: {result.value}")
        else:
            print("        parser_results is None (parse error or no output)")
        return ok
    finally:
        env.close()


if __name__ == "__main__":
    results = [run_case(*case) for case in CASES]
    print(f"\n{sum(results)}/{len(results)} cases matched expectations")
