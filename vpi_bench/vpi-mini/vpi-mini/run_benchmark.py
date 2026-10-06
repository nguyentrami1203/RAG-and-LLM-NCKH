#!/usr/bin/env python3
"""Chạy 1 agent trên toàn bộ test case, lưu trace vào results/<run>/traces/*.json"""
import argparse, json
from pathlib import Path
from env import Sandbox
from agents import make_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cases.jsonl")
    ap.add_argument("--agent", required=True, help="vd: sim:0.8 | anthropic:<model> | openai:<model>")
    ap.add_argument("--defense", action="store_true", help="thêm system-prompt defense")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="số lần chạy mỗi case (paper dùng 3)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(a.data)][: a.limit]
    agent = make_agent(a.agent, a.defense)
    out = Path("results") / a.run_name / "traces"; out.mkdir(parents=True, exist_ok=True)
    for c in cases:
        for r in range(a.repeat):
            sb = Sandbox(c)
            try:
                extra = agent.run(c, sb)
            except Exception as e:
                extra = {"error": repr(e)}
            trace = dict(case_id=c["id"], run=r, agent=agent.name, calls=sb.calls, finished=sb.finished,
                         answer=sb.answer, **extra)
            (out / f"{c['id']}__r{r}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1))
        print(f"done {c['id']}")
    print(f"Traces -> {out}")


if __name__ == "__main__":
    main()
