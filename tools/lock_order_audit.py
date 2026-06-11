#!/usr/bin/env python3
"""Lock-order audit: every same-scope HELD-guard acquisition sequence must
follow ChittaField's struct declaration order (THEORY.md §7 invariant 4,
production deadlocks 2026-06-11). Exit 1 on any opposite-order pair among
tool-reachable sites. sync_foreign's canonical sweep runs under the C++
EXCLUSIVE rpc lock and is exempt (it cannot interleave with tool paths)."""
import re, sys, pathlib

root = pathlib.Path(__file__).resolve().parent.parent / "src"
field_src = (root / "field.rs").read_text()
struct_body = re.search(r'pub struct ChittaField \{(.*?)\n\}', field_src, re.S).group(1)
ORDER = {m.group(1): i for i, m in enumerate(
    re.finditer(r'pub(?:\(crate\))? (\w+):\s*(?:parking_lot::)?RwLock<', struct_body))}

# Held-guard binding only: `let g = <recv>.<field>.read();` (statement ends at guard)
BIND = re.compile(r'let\s+(?:mut\s+)?(\w+)\s*=\s*(?:self|handle\.field|field|h\.field)\.(\w+)\.(read|write)\(\);')

def sync_foreign_span(text):
    m = re.search(r'fn sync_foreign.*?\n    \}', text, re.S)
    return (text[:m.start()].count('\n'), text[:m.end()].count('\n')) if m else (-1, -1)

pairs, sites = {}, {}
bad = []
for fname in ["store.rs", "field.rs", "ffi.rs"]:
    text = (root / fname).read_text()
    lines = text.splitlines()
    exempt = sync_foreign_span(text) if fname == "field.rs" else (-1, -1)
    for i, line in enumerate(lines):
        m1 = BIND.search(line)
        if not m1 or m1.group(2) not in ORDER: continue
        if exempt[0] <= i <= exempt[1]: continue
        guard_var = m1.group(1)
        depth = 0
        for j in range(i + 1, min(i + 60, len(lines))):
            depth += lines[j].count('{') - lines[j].count('}')
            if depth < 0: break
            if re.search(r'\bdrop\(' + re.escape(guard_var) + r'\)', lines[j]):
                break  # guard explicitly released before any later acquisition
            m2 = BIND.search(lines[j])
            if m2 and m2.group(2) in ORDER and m2.group(2) != m1.group(2):
                if ORDER[m2.group(2)] < ORDER[m1.group(2)]:
                    bad.append(f"{fname}:{i+1} holds {m1.group(2)}.{m1.group(3)} "
                               f"then takes earlier-ordered {m2.group(2)} at :{j+1}")
                break
for b in bad: print("LOCK-ORDER VIOLATION:", b)
print(f"{len(bad)} violations ({len(ORDER)} ordered locks)")
sys.exit(1 if bad else 0)
