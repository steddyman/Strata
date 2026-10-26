"""Coding speed: five coding tasks, thinking off, temperature 0, up to 512 tokens each, prompt cache off, the server's own
timings; --rounds passes over the set.  Prints per-task tok/s and RESULT with the total (tokens / decode time)."""
import argparse, json, statistics, time, urllib.request

TASKS = [
    ("lru", "Write a Python module that implements an LRU cache with a TTL per entry, with type hints, docstrings and a short example at the end."),
    ("dataclass", "Write a Python dataclass for a Customer with 12 fields (id, name, email, phone, address lines, city, postcode, country, "
                  "created and updated timestamps, and an active flag), then a to_dict() method and a from_dict() classmethod that handle every field."),
    ("tests", "Write pytest unit tests for a function parse_duration(s: str) -> int that turns strings like '1h30m', '45s', '2d' and '1h 5m 10s' "
              "into seconds and raises ValueError on bad input. Cover normal cases, edge cases and errors."),
    ("api", "Write a FastAPI app with Pydantic models for a todo list: create, list (with paging), get, update and delete endpoints, "
            "an in-memory store, proper status codes and 404 handling."),
    ("ts", "Convert this to TypeScript with full types and JSDoc comments: a function that groups an array of order objects "
           "{id, customer, total, status, createdAt} by customer, sums the totals per customer, and returns the top N customers sorted by total."),
]

def chat(url, text):
    body = {"model": "x", "max_tokens": 512, "temperature": 0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"[{time.time()}] {text}"}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    return r["timings"]

ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--rounds", type=int, default=2)
a = ap.parse_args()
chat(a.url, "Say hello.")   # warm-up
per, n_all, ms_all = {k: [] for k, _ in TASKS}, 0, 0.0
for r in range(a.rounds):
    for k, text in TASKS:
        t = chat(a.url, text)
        per[k].append(t["predicted_per_second"]); n_all += t["predicted_n"]; ms_all += t["predicted_ms"]
        print(f"  r{r} {k}: {t['predicted_n']} tok at {t['predicted_per_second']:.1f} tok/s", flush=True)
res = {"coding_total": round(n_all / ms_all * 1000, 2), **{k: round(statistics.median(v), 1) for k, v in per.items()}}
print("RESULT " + json.dumps(res))
