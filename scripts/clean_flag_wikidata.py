#!/usr/bin/env python3
"""Flag Wikidata probes whose bare entity label is ambiguous and whose question
does not disambiguate it.

- label ambiguity: >=2 Wikidata items share the exact English label.
- grounding: the question contains a disambiguating qualifier beyond the bare
  name (a comma-qualifier, or "in <Capitalized place>"), which the audit added
  to many probes. Grounded probes are NOT flagged even if the label is shared.

A probe is flagged AMBIGUOUS only if (label shared by >=2 items) AND (not grounded).
"""
import json, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "probes" / "wikidata_ambiguity.json"
UA = {"User-Agent": "IKP-dataset-clean/1.0 (research; bojieli@gmail.com)"}
API = "https://www.wikidata.org/w/api.php"

probes = json.load(open(ROOT / "data" / "probes" / "final_probe_set_v8.json"))
wd = [x for x in probes if x.get("source_type") == "wikidata" and x.get("wikidata_id")]

def get(params):
    for attempt in range(4):
        try:
            r = httpx.get(API, params={**params, "format": "json"}, timeout=30, headers=UA)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {}

# 1) batch labels (50 ids/call)
labels = {}
ids = [x["wikidata_id"] for x in wd]
for i in range(0, len(ids), 50):
    chunk = ids[i:i+50]
    d = get({"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels", "languages": "en"})
    for qid, ent in d.get("entities", {}).items():
        labels[qid] = ent.get("labels", {}).get("en", {}).get("value")
    time.sleep(0.15)
print(f"fetched {sum(1 for v in labels.values() if v)}/{len(ids)} labels")

# 2) same-label entities with descriptions (dedup by label)
uniq = sorted({l for l in labels.values() if l})
same_ents = {}   # label -> list of {id, description}
lock = threading.Lock(); done = [0]
def fetch_label(lbl):
    d = get({"action": "wbsearchentities", "search": lbl, "language": "en", "limit": 50})
    ents = [{"id": x.get("id"), "description": (x.get("description") or "")}
            for x in d.get("search", []) if x.get("label", "").lower().strip() == lbl.lower().strip()]
    with lock:
        done[0] += 1
        if done[0] % 100 == 0: print(f"  labels {done[0]}/{len(uniq)}", flush=True)
    return lbl, ents
with ThreadPoolExecutor(max_workers=6) as ex:
    for lbl, ents in ex.map(fetch_label, uniq):
        same_ents[lbl] = ents

STOP = {"the","a","an","of","in","and","or","by","for","to","at","on","from","with"}
def dwords(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}

def type_collisions(qid, lbl):
    """Count same-label entities that share a description keyword with the intended entity
    (i.e. are plausibly the same TYPE) — excludes unrelated homonyms (song/ship named 'Canada')."""
    ents = same_ents.get(lbl, [])
    intended = next((e for e in ents if e["id"] == qid), None)
    if intended is None:
        # intended not surfaced by search; fall back to same-description-cluster largest group
        idesc = set()
    else:
        idesc = dwords(intended["description"])
    others = [e for e in ents if e["id"] != qid]
    if not idesc:
        # no intended description: count others that share a keyword with ANY other (loose)
        return len(others) if len(ents) >= 2 else 0, others
    coll = [e for e in others if dwords(e["description"]) & idesc]
    return len(coll), coll

# 3) grounding heuristic
COUNTRY_ALIAS = {
    "czech republic": {"czechia", "czech"},
    "united kingdom": {"uk", "britain", "england", "scotland", "wales"},
    "united states": {"usa", "states", "american"},
    "people's republic of china": {"china", "chinese"},
}
COUNTRY_RE = re.compile(r"\bin\s+([A-Z][a-zA-Z.\-]+(?:\s+[A-Z][a-zA-Z.\-]+)*)")
def grounded(q, label, coll=None):
    body = q
    if label:  # strip the entity name so "in <place>" refers to a qualifier, not the name
        body = re.sub(re.escape(label), "", body, flags=re.IGNORECASE)
    if "," in body:  # comma qualifier e.g. "Putnam, Connecticut"
        return True
    m = COUNTRY_RE.search(body)
    if m:  # "... in Connecticut ..."
        place = m.group(1).lower().strip()
        place_tokens = {w for w in re.findall(r"[a-z]+", place) if len(w) > 2} | COUNTRY_ALIAS.get(place, set())
        # If colliding entities are already in that same country/place (e.g. multiple
        # rivers named Bělá inside Czech Republic), country-level "in <Place>" does not disambiguate.
        if coll and place_tokens:
            same_place_coll = [c for c in coll if dwords(c.get("description", "")) & place_tokens]
            if same_place_coll:
                return False
        return True
    return False

out = []
for x in wd:
    lbl = labels.get(x["wikidata_id"])
    n_type, coll = type_collisions(x["wikidata_id"], lbl) if lbl else (0, [])
    g = grounded(x["question"], lbl or "", coll)
    ambiguous = (n_type >= 1) and (not g)   # >=1 OTHER same-type same-label entity, ungrounded
    out.append({"probe_id": x["id"], "tier": x["tier"], "wikidata_id": x["wikidata_id"],
                "label": lbl, "same_type_collisions": n_type,
                "collision_desc": [c["description"] for c in coll][:10],
                "grounded": g, "ambiguous": ambiguous, "question": x["question"]})
json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
flagged = [r for r in out if r["ambiguous"]]
shared = [r for r in out if r["same_type_collisions"] >= 1]
print(f"\n{len(shared)}/{len(out)} have a same-TYPE same-label collision; "
      f"{len(shared)-len(flagged)} of those are grounded (kept).")
print(f"{len(flagged)}/{len(out)} flagged AMBIGUOUS (same-type collision AND ungrounded) = {100*len(flagged)/len(out):.1f}%")
from collections import Counter
print("by tier:", dict(Counter(r["tier"] for r in flagged)))
for r in flagged[:10]:
    print(f"  {r['probe_id']} [{r['tier']}] {r['label']!r} x{r['same_type_collisions']} {r['collision_desc']} | {r['question'][:55]}")
print(f"wrote {OUT}")
