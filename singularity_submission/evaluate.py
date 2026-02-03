import json
import re
import unicodedata
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

# Load files
with open("/home/ach18346zf/dev.jsonl") as f:
    golds = [json.loads(line) for line in f]
with open("/home/ach18346zf/ft-llm-2026/singularity_submission/output.jsonl") as f:
    preds = [json.loads(line) for line in f]

correct = 0
wrong = []
errors = []

for gold, pred in zip(golds, preds):
    assert gold["id"] == pred["id"]
    sol = gold["solution"]
    out = pred["output"]
    try:
        if check(sol, out):
            correct += 1
        else:
            wrong.append((gold["id"], gold["category"], gold["unit"], sol, out))
    except Exception as e:
        errors.append((gold["id"], str(e), sol, out))

total = len(golds)
print(f"=== Results: {correct}/{total} ({correct/total*100:.1f}%) ===\n")

if wrong:
    print(f"--- Wrong ({len(wrong)}) ---")
    for id_, cat, unit, sol, out in wrong:
        print(f"  #{id_:3d} [{cat}/{unit}]  gold={sol!r}  pred={out!r}")

if errors:
    print(f"\n--- Errors ({len(errors)}) ---")
    for id_, err, sol, out in errors:
        print(f"  #{id_:3d} {err}  gold={sol!r}  pred={out!r}")
