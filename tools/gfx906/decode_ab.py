#!/usr/bin/env python3
"""Decode A/B on a local Strata install: one `strata.py serve` per variant, the same fixed prompts, tok/s from the engine log.

    python tools/gfx906/decode_ab.py --config RUN.json [--port 18600] [--runs 3] [--profile] \
        --variant base --variant fuse=STRATA_EXP_MODE=8,STRATA_QFUSE=1 ...

A variant's ENGINE=<engine package folder> runs that engine (its "strata") instead of the one Control picks.
RUN.json is a run config as Control writes it (strata-<model>.json or a native instance config with "api_key"); its
"log" is replaced by one log per variant next to --out-dir.  Each variant is NAME or NAME=K=V,K=V (the environment the
server and its engine start with).  Per variant: a long run (a ~8K-token prompt with a fresh tag, so nothing is reused,
then 256 tokens) and a short run (an ~85-token prompt, 400 tokens), --runs times each, temperature 0.  --profile adds
STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 and prints the last per-stage total (it changes the graph: the
shared-expert side stream and some fusions are off while profiling, so compare tok/s without it).
Nothing of the installed app is changed; the GPUs must be free (close the Strata app first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ \d+ read in \d+ ms \(([\d.]+) tok/s\), (\d+) generated in \d+ ms "
                  r"\(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def control_python() -> str:
    staged = ROOT / "apps/desktop/src-tauri/resources/control/python/bin/python3"
    return str(staged) if staged.exists() else sys.executable


def ask(cfg: dict, port: int, log: pathlib.Path, prompt: str, n: int):
    body = json.dumps({"model": cfg["model_name"], "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + cfg.get("api_key", "")})
    pos = log.stat().st_size
    reply = json.loads(urllib.request.urlopen(req, timeout=900).read())
    msg = reply["choices"][0]["message"]
    text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")   # a thinking model fills the first
    time.sleep(1.0)
    with open(log, encoding="utf-8", errors="replace") as f:
        f.seek(pos)
        found = LINE.findall(f.read())
    return (found[-1] if found else None), hashlib.sha256(text.encode()).hexdigest()[:12]


def run_variant(name: str, env_extra: dict, a, base: dict, doc: str) -> dict:
    out = pathlib.Path(a.out_dir)
    cfg = dict(base, log=str(out / f"{name}.log"))
    engine_dir = env_extra.pop("ENGINE", None)       # NAME=ENGINE=<package folder>,...: run that engine instead of Control's pick
    if engine_dir:
        cfg["exe"] = str(pathlib.Path(engine_dir).expanduser() / "strata")
    cfg_path = out / f"{name}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    log = pathlib.Path(cfg["log"])
    log.write_text("")
    ready = out / f"{name}.ready.json"
    ready.unlink(missing_ok=True)
    env = dict(os.environ, **env_extra)
    env.setdefault("STRATA_DECODE_TIMING", "1")   # host-side per-request timing; leaves the graph as it is
    if a.profile:
        env.update(STRATA_DECODE_TIMING="1", STRATA_VERIFY_PROFILE="1")
    proc = subprocess.Popen([control_python(), "strata.py", "serve", "--config", str(cfg_path), "--port", str(a.port),
                             "--ready-file", str(ready)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        t0 = time.time()
        while not (ready.exists() and ready.stat().st_size):
            if proc.poll() is not None or time.time() - t0 > 900:
                tail = log.read_text(errors="replace").splitlines()[-15:]
                raise SystemExit(f"{name}: the server did not start\n" + "\n".join(tail))
            time.sleep(2)
        engine = re.search(r"components/engine/([^/ ]+)", log.read_text(errors="replace"))
        long_d, short_d, acc = [], [], []
        reused = 0
        digests = []
        for i in range(a.runs):
            tag = f"{a.tag}-{i}"          # the same prompts for every variant: the same greedy text, the same drafts
            for kind, prompt, n, into in (("long", f"[run {tag}]\n\n{doc}\n\nSummarise the document above in detail.", 256, long_d),
                                          ("short", f"[run {tag}] Write a 400-word story about a lighthouse keeper who finds a map.", 400, short_d)):
                r, digest = ask(cfg, a.port, log, prompt, n)
                digests.append(digest)
                if r is None:
                    raise SystemExit(f"{name}: no decode line in {log}")
                into.append(float(r[4]))
                reused += int(r[1])
                acc.append(int(r[5]) / max(1, int(r[6])))
        text = log.read_text(errors="replace")
        stages = re.findall(r"total ([\d.]+) ms/window", text)
        # "strata decode timing: N windows, ... X ms/window": the window's cost, which draft acceptance barely moves
        win = [(int(n), float(ms)) for n, ms in re.findall(r"strata decode timing: (\d+) windows, .*? ([\d.]+) ms/window =", text)]
        return {"name": name, "engine": engine.group(1) if engine else "?", "long": long_d, "short": short_d,
                "accept": statistics.mean(acc), "reused": reused, "texts": ",".join(digests), "stage_ms": float(stages[-1]) if stages else None,
                "ms_window": sum(n * ms for n, ms in win) / max(1, sum(n for n, _ in win)) if win else None}
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--port", type=int, default=18600)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--tag", default="ab", help="prompt tag prefix; change it to get prompts no server has cached")
    ap.add_argument("--out-dir", default=str(pathlib.Path.home() / "strata-build" / "decode-ab"))
    a = ap.parse_args()
    pathlib.Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    base = json.loads(pathlib.Path(a.config).read_text())
    doc = (ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8")[:27000]
    results = []
    for v in a.variant:
        name, _, spec = v.partition("=")
        env = dict(kv.split("=", 1) for kv in spec.split(",") if kv)
        r = run_variant(name, env, a, base, doc)
        results.append(r)
        mean = statistics.mean(r["long"] + r["short"])
        print(f"{r['name']:24s} {r['engine']:34s} " + (f"{r['ms_window']:6.2f} ms/window  " if r["ms_window"] else "") + f"decode {mean:5.1f} tok/s (long {statistics.mean(r['long']):5.1f} "
              f"short {statistics.mean(r['short']):5.1f}, min {min(r['long'] + r['short']):5.1f} max {max(r['long'] + r['short']):5.1f}) "
              f"accept {r['accept']:.2f}" + (f"  REUSED {r['reused']} tokens" if r["reused"] else "") + f"  texts {r['texts']}" + (f"  stage {r['stage_ms']:.2f} ms/window" if r["stage_ms"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
