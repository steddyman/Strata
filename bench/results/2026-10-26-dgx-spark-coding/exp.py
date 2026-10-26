"""One engine experiment on the Spark: a single-stream copy of the NVFP4 config with the given environment variables and
extra engine arguments, served on 127.0.0.1:8180; runs decode_bench.py against it, keeps the engine's timing lines,
then stops the server. The user's own config is read, never written.

    python3 exp.py NAME [--env K=V ...] [--arg ARG ...] [--exe PATH] [--base CONFIG] [--rounds N] [--parallel N]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HOME = os.path.expanduser("~")
STRATA = f"{HOME}/Strata"
DEV = f"{HOME}/strata-dev"


def ready(port, secs=900):
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read()
            return True
        except Exception:
            time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--exe", default="")
    ap.add_argument("--base", default=f"{STRATA}/strata-nvfp4-nvfp4.json")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=0)
    ap.add_argument("--port", type=int, default=8180)
    ap.add_argument("--bench", default="decode", choices=["decode", "agents", "prompt", "coding"])
    ap.add_argument("--dense", default="", help="a dense GGUF for --native and --native-dense-gguf")
    ap.add_argument("--config-only", action="store_true")
    ap.add_argument("--drop-mtp-q4", action="store_true", help="remove --mtp-q4 X from the config args")
    a = ap.parse_args()
    c = json.load(open(a.base, encoding="utf-8-sig"))
    for k in ("parallel", "host", "api_key"):
        c.pop(k, None)
    if a.parallel:
        c["parallel"] = a.parallel
    c["port"] = a.port
    c["log"] = f"/tmp/exp-{a.name}.log"
    env = dict(c.get("env") or {})
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    c["env"] = env
    args0 = list(c["args"])
    if a.drop_mtp_q4 and "--mtp-q4" in args0:
        i = args0.index("--mtp-q4"); del args0[i:i + 2]
    c["args"] = args0 + a.arg
    if a.exe:
        c["exe"] = a.exe
    if a.dense:
        for i, x in enumerate(c["args"][:-1]):
            if x in ("--native", "--native-dense-gguf"):
                c["args"][i + 1] = a.dense
    cfg = f"/tmp/exp-{a.name}.json"
    json.dump(c, open(cfg, "w"), indent=1)
    if a.config_only:
        print(cfg)
        return
    if os.path.exists(c["log"]):
        os.remove(c["log"])
    rc = 0
    srv = subprocess.Popen([f"{STRATA}/.venv/bin/python", f"{STRATA}/serve/server.py", "--engine", "strata", "--config", cfg,
                            "--port", str(a.port)], cwd=STRATA, stdout=open(f"/tmp/exp-{a.name}.server.log", "w"),
                           stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not ready(a.port):
            print("NOT READY; server log tail:")
            print(open(f"/tmp/exp-{a.name}.server.log").read()[-2000:])
            return 2
        sys.stdout.flush()
        if a.bench == "agents":                                       # 8 concurrent 6-turn code-review sessions (agents_bench.py)
            files = open(f"{DEV}/agent_files.txt").read().split()
            r = subprocess.run([sys.executable, f"{DEV}/agents_bench.py", "--url", f"http://127.0.0.1:{a.port}/v1", "--agents",
                                str(a.parallel or 1), "--sessions", "8", "--files", *files, "--out", f"/tmp/exp-{a.name}.jsonl"],
                               check=False, capture_output=True, text=True)
            r.stdout = r.stdout.replace("SUMMARY ", "RESULT ")
        elif a.bench == "coding":
            r = subprocess.run([sys.executable, f"{DEV}/coding_bench.py", "--url", f"http://127.0.0.1:{a.port}/v1", "--rounds", str(a.rounds)],
                               check=False, capture_output=True, text=True)
        elif a.bench == "prompt":
            r = subprocess.run([sys.executable, f"{DEV}/prompt_bench.py", "--url", f"http://127.0.0.1:{a.port}/v1"],
                               check=False, capture_output=True, text=True)
        else:
            r = subprocess.run([sys.executable, f"{DEV}/decode_bench.py", "--url", f"http://127.0.0.1:{a.port}/v1",
                                "--long", f"{DEV}/prompt-4k.txt", "--rounds", str(a.rounds)], check=False, capture_output=True, text=True)
        print(r.stdout, end="")
        if r.returncode != 0:                                          # a failed bench: its error and exit status, not silence
            sys.stderr.write(r.stderr)
            rc = r.returncode
        with open(f"{DEV}/results.txt", "a") as f:                    # one line per experiment, kept for comparison
            res = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            f.write(f"{a.name}\t{' '.join(a.env)}\t{' '.join(a.arg)}\t{res[-1][7:] if res else 'NO RESULT'}\n")
    finally:
        os.killpg(srv.pid, signal.SIGTERM)
        try:
            srv.wait(900)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
    keep = ("decode timing", "GPU stages", "expert cache", "INFO ", "batch:", "verify:", "mtp", "spec")
    for line in open(c["log"], errors="replace"):
        if any(k in line for k in keep):
            print("LOG " + line.rstrip()[:400])
    return rc


if __name__ == "__main__":
    sys.exit(main())
