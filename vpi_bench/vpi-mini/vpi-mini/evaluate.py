#!/usr/bin/env python3
"""Tính AR / SR (tổng + theo platform / loại tấn công / vị trí chèn / benign task) và so sánh nhiều run."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from judge import rule_judge, llm_judge

GROUPS = ["platform", "malicious_type", "injection_position", "benign_task", "style"]


def load(run):
    return [json.loads(p.read_text()) for p in sorted((Path("results") / run / "traces").glob("*.json"))]


def rate(rows, key):
    n = len(rows)
    return (100 * sum(r[key] for r in rows) / n) if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="tên run trong results/")
    ap.add_argument("--data", default="data/cases.jsonl")
    ap.add_argument("--llm-judge", default=None, help="vd: anthropic:<m1>,openai:<m2>,... (bỏ phiếu đa số)")
    a = ap.parse_args()
    cases = {c["id"]: c for c in map(json.loads, open(a.data))}
    judges = a.llm_judge.split(",") if a.llm_judge else None

    summary, all_rows = [], []
    for run in a.runs:
        rows = []
        for t in load(run):
            c = cases[t["case_id"]]
            j = rule_judge(c, t)
            row = dict(run=run, case_id=c["id"], attempted=j["attempted"], success=j["success"], behavior=j["behavior"],
                       **{g: c[g] for g in GROUPS})
            if judges:
                lj = llm_judge(c, t, judges); row.update(llm_attempted=lj["attempted"], llm_success=lj["success"])
            rows.append(row)
        all_rows += rows
        summary.append((run, rows))

    print(f"\n## Tổng quan (N = số lần chạy; AR/SR tính theo %)\n")
    print("| run | N | AR | SR |" + (" LLM-AR | LLM-SR | agree(SR) |" if judges else ""))
    print("|---|---|---|---|" + ("---|---|---|" if judges else ""))
    for run, rows in summary:
        line = f"| {run} | {len(rows)} | {rate(rows,'attempted'):.1f} | {rate(rows,'success'):.1f} |"
        if judges:
            agree = 100 * sum(r["success"] == r["llm_success"] for r in rows) / len(rows)
            line += f" {rate(rows,'llm_attempted'):.1f} | {rate(rows,'llm_success'):.1f} | {agree:.1f} |"
        print(line)

    for g in GROUPS:
        vals = sorted({r[g] for r in all_rows})
        print(f"\n## Theo {g}  (ô: AR / SR)\n")
        print("| run | " + " | ".join(vals) + " |"); print("|---|" + "---|" * len(vals))
        for run, rows in summary:
            cells = []
            for v in vals:
                sub = [r for r in rows if r[g] == v]
                cells.append(f"{rate(sub,'attempted'):.0f} / {rate(sub,'success'):.0f} (n={len(sub)})" if sub else "-")
            print(f"| {run} | " + " | ".join(cells) + " |")

    print("\n## Phân bố hành vi (như Fig. 3 của paper)\n")
    beh = ["success", "partial", "failed", "unrecognized", "recognized"]
    print("| run | " + " | ".join(beh) + " |"); print("|---|" + "---|" * len(beh))
    for run, rows in summary:
        print(f"| {run} | " + " | ".join(f"{100*sum(r['behavior']==b for r in rows)/len(rows):.0f}%" for b in beh) + " |")

    out = Path("results") / "per_case_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"\nChi tiết từng case -> {out}")


if __name__ == "__main__":
    main()
