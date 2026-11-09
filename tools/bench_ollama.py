"""Measure what the chat architecture costs on YOUR machine/models (needs a running Ollama).

    python tools/bench_ollama.py gemma3:4b qwen3:8b

For each model: time to first token (TTFT) and total time for a plain chat message, the
same message with the forex tool schema attached, and the time until the model emits a tool
call for a chart request. Median of N runs after one warm-up (so model loading is excluded).
"""
import json
import statistics
import sys
import time
import urllib.request

URL = "http://127.0.0.1:11434/api/chat"
sys.path.insert(0, ".")
from local_chat.server import _FOREX_TOOLS  # noqa: E402


def run(model, text, tools=None, options=None):
    body = {"model": model, "stream": True, "keep_alive": "10m",
            "messages": [{"role": "user", "content": text}],
            "options": {"num_predict": 80, "temperature": 0, **(options or {})}}
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start, first, called = time.perf_counter(), None, None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            packet = json.loads(line)
            message = packet.get("message") or {}
            if first is None and (message.get("content") or message.get("tool_calls") or message.get("thinking")):
                first = time.perf_counter() - start
            if message.get("tool_calls"):
                called = message["tool_calls"][0]["function"]["name"]
    return first, time.perf_counter() - start, called


def median(model, text, tools=None, runs=5):
    run(model, text, tools)  # warm-up
    samples = [run(model, text, tools) for _ in range(runs)]
    return (statistics.median(s[0] or 0 for s in samples), statistics.median(s[1] for s in samples),
            [s[2] for s in samples])


if __name__ == "__main__":
    for model in sys.argv[1:] or ["gemma3:4b"]:
        print(f"== {model}")
        for label, text, tools in (("plain chat, no tools", "سلام، یه جمله دربارهٔ طلا بگو", None),
                                   ("plain chat, with tools", "سلام، یه جمله دربارهٔ طلا بگو", _FOREX_TOOLS),
                                   ("chart request, with tools", "یه اسکرین‌شات از xauusd بگیر و fvg مهم رو مشخص کن", _FOREX_TOOLS),
                                   ("analysis request, with tools", "تحلیل XAUUSD در H1 چیه؟", _FOREX_TOOLS)):
            ttft, total, calls = median(model, text, tools)
            print(f"{label:32s} TTFT {ttft*1000:7.0f} ms   total {total*1000:7.0f} ms   tool calls: {calls}")
