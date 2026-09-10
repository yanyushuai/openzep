#!/usr/bin/env python3
"""Throughput benchmark for the two models OpenZep depends on.

    LLM       : qwen3.8-27b        (structured entity extraction, thinking off)
    Embedder  : WeMM-Embedding-2B  (2048-dim)

Runs inside the openzep container; reads keys from env.
Measures single-call latency, generation speed (tok/s), and aggregated
throughput under concurrency. Mirrors OpenZep's real call shape:
  - LLM: OpenAI-compatible /chat/completions with response_format=json_object
         and chat_template_kwargs.enable_thinking=false
  - Embedder: OpenAI-compatible /embeddings
"""
import asyncio
import json
import os
import statistics
import time

import httpx

LLM_URL = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
LLM_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
EMB_URL = os.environ["EMBEDDER_BASE_URL"].rstrip("/") + "/embeddings"
EMB_KEY = os.environ["EMBEDDER_API_KEY"]
EMB_MODEL = os.environ["EMBEDDER_MODEL"]

EXTRACT_SYSTEM = (
    "You are an entity and relationship extraction engine. "
    "Extract named entities and the relationships between them from the text. "
    "Return a JSON object with keys 'entities' and 'relationships'."
)
EXTRACT_USER = (
    "用户 Alice 和 Bob 在 Twitter 上讨论 AI 芯片。"
    "Alice 发布了一条关于 Nvidia H100 的帖子，提到其 3.35TB/s 的显存带宽。"
    "Bob 回复说 AMD 的 MI300X 有 192GB 显存，并在 Reddit 上创建了讨论帖。"
    "随后 Carol 点赞了 Bob 的帖子。Alice 又提到 OpenAI 正在训练 GPT-5。"
)

LLM_HEADERS = {"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"}
EMB_HEADERS = {"Authorization": f"Bearer {EMB_KEY}", "Content-Type": "application/json"}


async def llm_once(client: httpx.AsyncClient):
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": EXTRACT_USER},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = await client.post(LLM_URL, json=body, headers=LLM_HEADERS)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return {"error": r.status_code, "body": r.text[:300], "latency_s": dt}
    j = r.json()
    usage = j.get("usage", {})
    return {
        "latency_s": dt,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def emb_once(client: httpx.AsyncClient, texts):
    body = {"model": EMB_MODEL, "input": texts}
    t0 = time.perf_counter()
    r = await client.post(EMB_URL, json=body, headers=EMB_HEADERS)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return {"error": r.status_code, "body": r.text[:300], "latency_s": dt}
    j = r.json()
    data = j.get("data", [])
    dim = len(data[0]["embedding"]) if data else 0
    return {"latency_s": dt, "n": len(data), "dim": dim}


def fmt_tok_per_s(tokens: int, seconds: float) -> str:
    return f"{tokens / seconds:.1f} tok/s" if seconds > 0 else "-"


async def bench_llm():
    print("=" * 70)
    print(f"LLM  {LLM_MODEL}  ({LLM_URL})")
    print("=" * 70)
    async with httpx.AsyncClient(timeout=180.0) as c:
        # warm-up (cold channel / gateway warmup)
        w = await llm_once(c)
        print(f"[warm-up]            {w.get('latency_s', 0):6.2f}s  "
              f"out={w.get('completion_tokens', '?')} tok")
        if "error" in w:
            print("!! LLM warm-up error:", w)
            return

        # single-call (3 runs)
        lat, ctoks, ptoks = [], [], []
        for _ in range(3):
            r = await llm_once(c)
            lat.append(r["latency_s"])
            ctoks.append(r["completion_tokens"])
            ptoks.append(r["prompt_tokens"])
        avg = statistics.mean(lat)
        print(f"[single x3]          avg={avg:6.2f}s  "
              f"min={min(lat):.2f}s  max={max(lat):.2f}s")
        print(f"                     in={statistics.mean(ptoks):.0f} tok  "
              f"out={statistics.mean(ctoks):.0f} tok  "
              f"gen={fmt_tok_per_s(int(statistics.mean(ctoks)), avg)}  "
              f"overall={fmt_tok_per_s(int(statistics.mean(ptoks) + statistics.mean(ctoks)), avg)}")

        # concurrency sweep
        for concurrency in (5, 10):
            t0 = time.perf_counter()
            results = await asyncio.gather(*(llm_once(c) for _ in range(concurrency)))
            wall = time.perf_counter() - t0
            lats = [r["latency_s"] for r in results]
            out_tok = sum(r["completion_tokens"] for r in results)
            in_tok = sum(r["prompt_tokens"] for r in results)
            print(f"[concurrency {concurrency:>2}]      wall={wall:6.2f}s  "
                  f"req/s={concurrency / wall:.2f}  "
                  f"avg_lat={statistics.mean(lats):.2f}s  "
                  f"max_lat={max(lats):.2f}s")
            print(f"                     agg_out={fmt_tok_per_s(out_tok, wall)}  "
                  f"agg_total={fmt_tok_per_s(in_tok + out_tok, wall)}")


async def bench_embedder():
    print()
    print("=" * 70)
    print(f"EMB  {EMB_MODEL}  ({EMB_URL})")
    print("=" * 70)
    text = "OpenZep knowledge graph entity and relationship embedding benchmark text."
    async with httpx.AsyncClient(timeout=30.0) as c:
        # single x5
        lats = []
        for _ in range(5):
            r = await emb_once(c, text)
            lats.append(r["latency_s"])
        avg = statistics.mean(lats)
        dim = r.get("dim", 0)
        print(f"[single x5]          avg={avg * 1000:7.1f}ms  "
              f"min={min(lats) * 1000:.1f}ms  max={max(lats) * 1000:.1f}ms  "
              f"dim={dim}  {1 / avg:.1f} req/s")

        # batch sizes
        for n in (10, 50):
            r = await emb_once(c, [f"{text} #{i}" for i in range(n)])
            print(f"[batch {n:>2}]           lat={r['latency_s'] * 1000:7.1f}ms  "
                  f"items/s={n / r['latency_s']:.1f}  per_item={r['latency_s'] / n * 1000:.1f}ms")

        # concurrency 10 single-item
        t0 = time.perf_counter()
        results = await asyncio.gather(*(emb_once(c, text) for _ in range(10)))
        wall = time.perf_counter() - t0
        lats = [r["latency_s"] for r in results]
        print(f"[concurrency 10]     wall={wall * 1000:7.1f}ms  "
              f"req/s={10 / wall:.1f}  "
              f"avg_lat={statistics.mean(lats) * 1000:.1f}ms  "
              f"max_lat={max(lats) * 1000:.1f}ms")


async def main():
    await bench_llm()
    await bench_embedder()


if __name__ == "__main__":
    asyncio.run(main())
