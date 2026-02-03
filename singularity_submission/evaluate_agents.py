import json
import re
import unicodedata
from collections import defaultdict
from math_verify import parse, verify

_SUP_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', 'ⁿ': 'n', 'ⁱ': 'i',
}
_SUB_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
}
_SUP_RE = re.compile('[' + re.escape(''.join(_SUP_MAP.keys())) + ']+')
_SUB_RE = re.compile('[' + re.escape(''.join(_SUB_MAP.keys())) + ']+')

def _normalize_math(s: str) -> str:
    s = s.strip()
    s = s.replace('\\\\', '\\')
    s = _SUP_RE.sub(lambda m: '^{' + ''.join(_SUP_MAP[c] for c in m.group(0)) + '}', s)
    s = _SUB_RE.sub(lambda m: '_{' + ''.join(_SUB_MAP[c] for c in m.group(0)) + '}', s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'^\$\$(.*)\$\$$', r'$\1$', s, flags=re.DOTALL)
    if not re.search(r'[\$]', s):
        s = f'${s}$'
    return s

def check(solution: str, output: str) -> bool:
    for a1, a2 in [(solution, output),
                    (_normalize_math(solution), _normalize_math(output))]:
        try:
            if verify(parse(a1), parse(a2)):
                return True
        except Exception:
            continue
    return solution.strip() == output.strip()

# Load ground truth
with open("/home/ach18346zf/dev.jsonl") as f:
    golds = {item["id"]: item["solution"] for item in map(json.loads, f)}

base_dir = "/home/ach18346zf/ft-llm-2026/singularity_submission"
agent_labels = {
    0: "Agent 0 (v2)",
    1: "Agent 1 (v2)",
    2: "Agent 2 (v5)",
    3: "Agent 3 (v5)",
}

for agent_id in range(4):
    path = f"{base_dir}/agent_{agent_id}_final.jsonl"
    with open(path) as f:
        samples = [json.loads(line) for line in f]

    # Group by problem id
    by_problem = defaultdict(list)
    for s in samples:
        by_problem[s["id"]].append(s["output"])

    correct = 0
    total = len(by_problem)
    wrong_ids = []

    for pid in sorted(by_problem.keys()):
        sol = golds[pid]
        votes = by_problem[pid]
        # Majority vote within this agent
        from collections import Counter
        # Simple: check each sample, take majority
        correct_count = sum(1 for v in votes if check(sol, v))
        # Problem is "correct" if majority of samples are correct
        if correct_count > len(votes) / 2:
            correct += 1
        else:
            wrong_ids.append((pid, correct_count, len(votes)))

    print(f"=== {agent_labels[agent_id]}: {correct}/{total} ({correct/total*100:.1f}%) ===")
    if wrong_ids:
        for pid, cc, tv in wrong_ids:
            print(f"  #{pid:3d}: {cc}/{tv} samples correct")
    print()

# Also show per-sample accuracy (raw hit rate)
print("=" * 60)
print("Per-sample accuracy (raw hit rate across all samples):")
print("=" * 60)
for agent_id in range(4):
    path = f"{base_dir}/agent_{agent_id}_final.jsonl"
    with open(path) as f:
        samples = [json.loads(line) for line in f]

    hits = sum(1 for s in samples if check(golds[s["id"]], s["output"]))
    total = len(samples)
    print(f"  {agent_labels[agent_id]}: {hits}/{total} ({hits/total*100:.1f}%)")
