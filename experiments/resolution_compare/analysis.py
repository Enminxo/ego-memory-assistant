"""Analysis of the resolution test (endpoints in run.py, fixed before any run).

    facts    test A: split the six descriptions of every Neon window into facts
    review   test A: page where the facts are judged by hand, blind to size
    score-a  test A: facts lost at 720 and 360 against 1600, beyond seed noise
    score-b  test B: QAEgo4D accuracy at 270 against 540

Details decided here, before any of the new diaries were read:
  - The six descriptions of a window are shown to the fact extractor under
    letters A-F in an order shuffled per window, so neither the extractor nor
    the page knows which size wrote what. `facts --perm-seed 1` repeats the
    extraction with another order, to check that it does not move the counts.
  - Statements of absence ("no other people are present") are kept apart as
    kind "absence" and left out of the review and of every count: the
    pre-registered kinds are objects, colours, counts, text or brands, places
    and actions.
  - Facts stated by all six descriptions are not reviewed: they cannot be
    missed by any size, so they do not enter any difference.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from egomem.config import load_config
from egomem.llm import ChatClient
from egomem.memory import TextualMemory

from run import CONFIG, NEON_VIDEOS, NEON_WIDTHS, OUT, EVAL_ROOT, SegmentSource, window_frames

SEEDS = (0, 1)
CONDITIONS = [f"w{w}_s{s}" for w in NEON_WIDTHS for s in SEEDS]
LETTERS = "ABCDEF"
KINDS = ["object", "count", "text_or_brand", "colour", "action", "place", "absence", "other"]
NEON = OUT / "neon"
REVIEW_FILE = NEON / "review.json"
BOOTSTRAP = 10_000

FACT_PROMPT = """Below are {n} diary entries written about the same {secs}-second clip of an \
egocentric video, labelled {labels}. Different runs wrote them, so they mention \
different things and word them differently.

Break them into atomic facts that someone could check by looking at the clip: \
one object, colour, count, piece of text or brand, place or action per fact, \
written as one short sentence (e.g. "The 3D printer is a Bambu Lab", "There are \
nine printed objects on the build plate", "I pick up a pink scraper").

Rules:
- Merge facts that say the same thing in different words into one fact, and list \
every entry that states it.
- If entries disagree on a detail (nine vs ten objects, pink vs red), write one \
fact per version, each listing only the entries that state that version.
- An entry states a fact only if its text says it; do not infer.
- Statements that something did not happen or is not there ("no other people \
are present", "no medication was taken") get kind "absence".
- kind is one of: {kinds}.

Entries:
{entries}"""

FACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"fact": {"type": "string"},
                       "kind": {"type": "string", "enum": KINDS},
                       "stated_by": {"type": "array",
                                     "items": {"type": "string", "enum": list(LETTERS)}}},
        "required": ["fact", "kind", "stated_by"], "additionalProperties": False}}},
    "required": ["facts"], "additionalProperties": False}


# --- Test A: facts -------------------------------------------------------------

def load_neon_diaries() -> dict[tuple[str, int], dict]:
    """(video, window index) -> {"t0", "t1", "texts": {condition: text}}.

    Checks the join before anything is computed: every condition must have
    the same windows, with the same times, for every video.
    """
    windows: dict[tuple[str, int], dict] = {}
    for video in NEON_VIDEOS:
        reference: list[tuple[int, float, float]] | None = None
        for condition in CONDITIONS:
            path = NEON / condition / video / "memory.jsonl"
            if not path.exists():
                raise SystemExit(f"missing diary {path}; is run_all.sh done?")
            entries = sorted(TextualMemory.load(path).entries(), key=lambda e: e.index)
            keys = [(e.index, e.t0, e.t1) for e in entries]
            if reference is None:
                reference = keys
            elif keys != reference:
                raise SystemExit(f"{condition}/{video} has other windows than {CONDITIONS[0]}")
            for e in entries:
                row = windows.setdefault((video, e.index), {"t0": e.t0, "t1": e.t1, "texts": {}})
                row["texts"][condition] = e.text
    return windows


def extract_facts(perm_seed: int) -> Path:
    config = load_config(CONFIG)
    client = ChatClient(config["base_url"], config["reader"]["model"])
    out = NEON / f"facts_p{perm_seed}.jsonl"
    done = set()
    if out.exists():
        done = {(r["video"], r["index"]) for r in map(json.loads, out.open())}
    windows = load_neon_diaries()
    for (video, index), row in sorted(windows.items()):
        if (video, index) in done:
            continue
        order = list(CONDITIONS)
        random.Random(f"{perm_seed}:{video}:{index}").shuffle(order)
        letters = {LETTERS[i]: condition for i, condition in enumerate(order)}
        entries = "\n".join(f"{letter}: {row['texts'][c]}" for letter, c in letters.items())
        prompt = FACT_PROMPT.format(n=len(order), secs=round(row["t1"] - row["t0"]),
                                    labels=", ".join(letters), kinds=", ".join(KINDS),
                                    entries=entries)
        result = client.chat([{"role": "user", "content": prompt}], max_tokens=2000,
                             temperature=0.0,
                             response_format={"type": "json_schema", "json_schema": {
                                 "name": "facts", "strict": True, "schema": FACT_SCHEMA}})
        facts = json.loads(result.text)["facts"]
        for k, fact in enumerate(facts):
            fact["id"] = f"{video}:{index}:{k}"
            fact["conditions"] = sorted({letters[x] for x in fact["stated_by"]})
        record = {"video": video, "index": index, "t0": row["t0"], "t1": row["t1"],
                  "letters": letters, "texts": row["texts"], "facts": facts}
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{video} {index:2d}: {len(facts)} facts ({result.latency_s:.0f}s)", flush=True)
    return out


def load_facts(perm_seed: int) -> list[dict]:
    return [json.loads(line) for line in (NEON / f"facts_p{perm_seed}.jsonl").open()]


def to_review(record: dict) -> list[dict]:
    """Facts of one window that go to the reviewer: not absences, not stated by
    all six, and stated by at least one (an empty list is an extraction slip)."""
    return [f for f in record["facts"]
            if f["kind"] != "absence" and 0 < len(f["conditions"]) < len(CONDITIONS)]


# --- Test A: review page --------------------------------------------------------

def frame_path(video: str, index: int, k: int) -> Path:
    return NEON / "frames" / video / f"{index:02d}_{k}.jpg"


def prepare_frames(records: list[dict]) -> None:
    """The 8 frames of every window at 1600 px, as the reviewer sees them."""
    config = load_config(CONFIG)["describer"]
    for record in records:
        video, index = record["video"], record["index"]
        if frame_path(video, index, config["n_frames"] - 1).exists():
            continue
        source = SegmentSource(NEON_VIDEOS[video])
        _, _, frames = window_frames(source, index, config["clip_sec"], config["n_frames"])
        for k, frame in enumerate(frames):
            path = frame_path(video, index, k)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(path, quality=92)


PAGE = """<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revisión a ciegas</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddd;--yes:#2f7d4f;--no:#b8432f;--unk:#8a7a2a}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
header{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
h1{font-size:20px;margin:0}.muted{color:var(--muted)}
.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:12px 0}
.frames img{width:100%;cursor:zoom-in;border-radius:4px}
.fact{display:flex;gap:10px;align-items:center;padding:8px;border-bottom:1px solid var(--line)}
.fact.cur{background:#f0efe8}.fact p{flex:1;margin:0}
button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
button.on[data-v=true]{background:var(--yes);color:#fff}button.on[data-v=false]{background:var(--no);color:#fff}
button.on[data-v=unclear]{background:var(--unk);color:#fff}
nav{display:flex;gap:8px;margin:12px 0}
#zoom{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center}
#zoom img{max-width:100vw;max-height:100vh}
@media(max-width:700px){.frames{grid-template-columns:repeat(2,1fr)}}
</style>
<main>
<header><h1>Revisión a ciegas</h1><span class="muted" id="where"></span><span class="muted" id="progress"></span></header>
<p class="muted">Cada frase la escribió una de las versiones del diario y no se dice cuál. Mira las 8 fotos del tramo
(pincha para verlas a tamaño completo) y marca si la frase es <b>verdad</b> en este tramo, <b>falsa</b>, o si las fotos
<b>no bastan</b> para saberlo. Teclas: 1 verdad, 2 falso, 3 no se ve; flechas arriba/abajo para moverte, N/P tramo siguiente/anterior.</p>
<div class="frames" id="frames"></div>
<div id="facts"></div>
<nav><button id="prev">← Tramo anterior</button><button id="next">Tramo siguiente →</button>
<button id="todo">Ir al primero sin terminar</button></nav>
</main>
<div id="zoom"><img alt=""></div>
<script>
let W=[],A={},w=0,cur=0;
const $=s=>document.querySelector(s);
async function load(){W=await (await fetch('windows')).json();A=await (await fetch('answers')).json();show(firstTodo())}
function firstTodo(){const i=W.findIndex(x=>x.facts.some(f=>!A[f.id]));return i<0?0:i}
function show(i){w=Math.max(0,Math.min(W.length-1,i));cur=0;const x=W[w];
 $('#where').textContent=`tramo ${w+1} de ${W.length}`;
 $('#frames').innerHTML=x.frames.map(u=>`<img src="${u}" loading="lazy" alt="">`).join('');
 $('#frames').querySelectorAll('img').forEach(im=>im.onclick=()=>{$('#zoom img').src=im.src;$('#zoom').style.display='flex'});
 render();window.scrollTo(0,0)}
function render(){const x=W[w];
 $('#facts').innerHTML=x.facts.map((f,i)=>`<div class="fact${i==cur?' cur':''}" data-i="${i}"><p>${f.text}</p>`+
  ['true','false','unclear'].map((v,j)=>`<button data-v="${v}" class="${A[f.id]==v?'on':''}">${['Verdad','Falso','No se ve'][j]}</button>`).join('')+'</div>').join('');
 $('#facts').querySelectorAll('.fact').forEach(el=>el.querySelectorAll('button').forEach(b=>b.onclick=()=>{cur=+el.dataset.i;answer(b.dataset.v)}));
 const all=W.reduce((n,x)=>n+x.facts.length,0),done=Object.keys(A).length;$('#progress').textContent=`· ${done} de ${all} frases`}
async function answer(v){const f=W[w].facts[cur];A[f.id]=v;
 await fetch('answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:f.id,value:v})});
 if(cur<W[w].facts.length-1)cur++;render()}
$('#prev').onclick=()=>show(w-1);$('#next').onclick=()=>show(w+1);$('#todo').onclick=()=>show(firstTodo());
$('#zoom').onclick=()=>$('#zoom').style.display='none';
document.onkeydown=e=>{if(e.key=='1')answer('true');else if(e.key=='2')answer('false');else if(e.key=='3')answer('unclear');
 else if(e.key=='ArrowDown'){cur=Math.min(W[w].facts.length-1,cur+1);render();e.preventDefault()}
 else if(e.key=='ArrowUp'){cur=Math.max(0,cur-1);render();e.preventDefault()}
 else if(e.key=='n'||e.key=='N')show(w+1);else if(e.key=='p'||e.key=='P')show(w-1);else if(e.key=='Escape')$('#zoom').style.display='none'};
load();
</script></html>"""


def review_windows(records: list[dict]) -> list[dict]:
    """What the page shows: windows in a shuffled order, facts shuffled inside,
    and nothing that says which description or size a fact came from."""
    n_frames = load_config(CONFIG)["describer"]["n_frames"]
    shown = []
    for record in records:
        facts = to_review(record)
        if not facts:
            continue
        random.Random(f"review:{record['video']}:{record['index']}").shuffle(facts)
        shown.append({"facts": [{"id": f["id"], "text": f["fact"]} for f in facts],
                      "frames": [f"frame/{record['video']}/{record['index']:02d}_{k}.jpg"
                                 for k in range(n_frames)]})
    random.Random("review-order").shuffle(shown)
    return shown


def serve_review(bind: str, port: int) -> None:
    records = load_facts(0)
    prepare_frames(records)
    windows = json.dumps(review_windows(records), ensure_ascii=False).encode()
    answers: dict[str, str] = json.loads(REVIEW_FILE.read_text()) if REVIEW_FILE.exists() else {}
    valid_ids = {f["id"] for r in records for f in to_review(r)}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            if self.path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/windows":
                self._send(windows, "application/json")
            elif self.path == "/answers":
                self._send(json.dumps(answers).encode(), "application/json")
            elif self.path.startswith("/frame/"):
                parts = self.path.split("/")
                if len(parts) != 4 or parts[2] not in NEON_VIDEOS or ".." in parts[3]:
                    self._send(b"not found", "text/plain", 404)
                    return
                path = NEON / "frames" / parts[2] / parts[3]
                if not path.is_file():
                    self._send(b"not found", "text/plain", 404)
                    return
                self._send(path.read_bytes(), "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/answer":
                self._send(b"not found", "text/plain", 404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("id") not in valid_ids or body.get("value") not in ("true", "false", "unclear"):
                self._send(b"bad answer", "text/plain", 400)
                return
            answers[body["id"]] = body["value"]
            tmp = REVIEW_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(answers, indent=1))
            tmp.replace(REVIEW_FILE)
            self._send(b"ok", "text/plain")

        def log_message(self, *args: object) -> None:
            pass

    n_facts = len(valid_ids)
    print(f"review: {n_facts} facts, {len(answers)} answered; http://{bind}:{port}/", flush=True)
    ThreadingHTTPServer((bind, port), Handler).serve_forever()


# --- Test A: scoring --------------------------------------------------------------

def stated(record: dict, condition: str, keep: set[str]) -> set[str]:
    return {f["id"] for f in record["facts"] if condition in f["conditions"] and f["id"] in keep}


def bootstrap_mean(values: list[float], seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(BOOTSTRAP))
    return statistics.fmean(values), means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def score_a() -> dict:
    records = load_facts(0)
    answers: dict[str, str] = json.loads(REVIEW_FILE.read_text())
    reviewable = {f["id"] for r in records for f in to_review(r)}
    missing = reviewable - set(answers)
    if missing:
        raise SystemExit(f"{len(missing)} of {len(reviewable)} facts still to review")
    true_ids = {i for i, v in answers.items() if v == "true"}
    false_ids = {i for i, v in answers.items() if v == "false"}

    def missed(record: dict, test: str, ref: str) -> set[str]:
        """True facts the reference states and the test does not."""
        return stated(record, ref, true_ids) - stated(record, test, true_ids)

    per_window: dict[str, list[float]] = defaultdict(list)
    missed_kinds: dict[str, dict[str, float]] = {w: defaultdict(float) for w in (720, 360)}
    kind_of = {f["id"]: f["kind"] for r in records for f in r["facts"]}
    for record in records:
        noise = statistics.fmean([len(missed(record, "w1600_s1", "w1600_s0")),
                                  len(missed(record, "w1600_s0", "w1600_s1"))])
        per_window["noise"].append(noise)
        for width in (720, 360):
            losses = [missed(record, f"w{width}_s{s}", f"w1600_s{s}") for s in SEEDS]
            loss = statistics.fmean(len(x) for x in losses)
            per_window[f"loss{width}"].append(loss)
            per_window[f"effect{width}"].append(loss - noise)
            for x in losses:
                for fact_id in x:
                    missed_kinds[width][kind_of[fact_id]] += 1 / len(SEEDS)
        for width in NEON_WIDTHS:
            per_window[f"false{width}"].append(statistics.fmean(
                len(stated(record, f"w{width}_s{s}", false_ids)) for s in SEEDS))
            per_window[f"true{width}"].append(statistics.fmean(
                len(stated(record, f"w{width}_s{s}", true_ids)) for s in SEEDS))

    summary = {"windows": len(records), "reviewed": len(reviewable),
               "true": len(true_ids & reviewable), "false": len(false_ids & reviewable),
               "unclear": sum(1 for i in reviewable if answers[i] == "unclear")}
    for key, values in per_window.items():
        mean, low, high = bootstrap_mean(values)
        summary[key] = {"mean": round(mean, 2), "ci95": [round(low, 2), round(high, 2)]}
    summary["missed_by_kind_per_window"] = {
        str(w): {k: round(v / len(records), 2) for k, v in sorted(kinds.items())}
        for w, kinds in missed_kinds.items()}
    effect = summary["effect720"]
    summary["reading"] = (
        "720 loses at most 1 true fact per window beyond seed noise" if effect["ci95"][1] < 1
        else "720 loses true facts beyond seed noise" if effect["ci95"][0] > 0
        else "not settled: the interval spans both readings")
    summary["knob_check_360_ge_720"] = summary["loss360"]["mean"] >= summary["loss720"]["mean"]
    return summary


def extraction_check() -> dict:
    """The same differences counted with two extraction orders, without the review:
    facts a 1600 description states and the 720 (or the other 1600 seed) does not."""
    result = {}
    for perm in (0, 1):
        counts: dict[str, list[int]] = defaultdict(list)
        for record in load_facts(perm):
            keep = {f["id"] for f in record["facts"] if f["kind"] != "absence"}
            for s in SEEDS:
                ref = stated(record, f"w1600_s{s}", keep)
                for width in (720, 360):
                    counts[f"w{width}"].append(len(ref - stated(record, f"w{width}_s{s}", keep)))
            counts["seed"].append(len(stated(record, "w1600_s0", keep) - stated(record, "w1600_s1", keep)))
        result[f"p{perm}"] = {k: round(statistics.fmean(v), 2) for k, v in counts.items()}
    return result


# --- Test B: scoring --------------------------------------------------------------

def load_eval(tag: str, seed: int) -> dict[str, dict]:
    path = EVAL_ROOT / "results" / f"eval_{tag}_seed{seed}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.open())}


def score_b() -> dict:
    arms = {"g31b": "g31b", "g31b_half": "g31b_half", "no-memory": "no-memory"}
    runs = {(arm, s): load_eval(tag, s) for arm, tag in arms.items() for s in SEEDS}
    ids = set(runs[("g31b", 0)])
    for key, rows in runs.items():
        if set(rows) != ids:
            raise SystemExit(f"{key} scored other questions than g31b seed 0")
    for s in SEEDS:
        for qid in ids:
            a, b = runs[("g31b", s)][qid], runs[("g31b_half", s)][qid]
            if (a["options"], a["correct"]) != (b["options"], b["correct"]):
                raise SystemExit(f"question {qid}, seed {s}: options differ between arms")

    def hit(arm: str, qid: str) -> float:
        return statistics.fmean(float(runs[(arm, s)][qid]["hit"]) for s in SEEDS)

    clip_of = {qid: runs[("g31b", 0)][qid]["scenario"] for qid in ids}
    by_clip: dict[str, list[str]] = defaultdict(list)
    for qid, clip in clip_of.items():
        by_clip[clip].append(qid)
    clips = sorted(by_clip)

    def acc(arm: str, qids: list[str]) -> float:
        return statistics.fmean(hit(arm, q) for q in qids)

    all_q = sorted(ids)
    diff = acc("g31b_half", all_q) - acc("g31b", all_q)
    rng = random.Random(0)
    boots = []
    for _ in range(BOOTSTRAP):
        sample = [q for c in rng.choices(clips, k=len(clips)) for q in by_clip[c]]
        boots.append(acc("g31b_half", sample) - acc("g31b", sample))
    boots.sort()
    low, high = boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP) - 1]
    no_mem = acc("no-memory", all_q)
    summary = {
        "questions": len(all_q), "clips": len(clips),
        "accuracy": {arm: round(acc(arm, all_q), 4) for arm in arms},
        "diff_half_minus_full": round(diff, 4), "ci95": [round(low, 4), round(high, 4)],
        "memory_gain_kept": round((acc("g31b_half", all_q) - no_mem) / (acc("g31b", all_q) - no_mem), 3),
        "unparsed": {f"{arm}_s{s}": sum(r["choice"] is None for r in runs[(arm, s)].values())
                     for arm in arms for s in SEEDS},
    }
    by_category: dict[str, list[str]] = defaultdict(list)
    for qid in all_q:
        by_category[runs[("g31b", 0)][qid]["category"]].append(qid)
    summary["by_category"] = {c: {"n": len(q), "g31b": round(acc("g31b", q), 3),
                                  "g31b_half": round(acc("g31b_half", q), 3)}
                              for c, q in sorted(by_category.items())}
    summary["reading"] = ("halving costs less than 3 points" if low > -0.03
                          else "the smaller frames lose accuracy" if high < 0
                          else "not settled: the interval spans both readings")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_facts = sub.add_parser("facts")
    p_facts.add_argument("--perm-seed", type=int, default=0)
    p_review = sub.add_parser("review")
    p_review.add_argument("--bind", default="127.0.0.1")
    p_review.add_argument("--port", type=int, default=8095)
    sub.add_parser("score-a")
    sub.add_parser("check-extraction")
    sub.add_parser("score-b")
    args = parser.parse_args()

    if args.command == "facts":
        extract_facts(args.perm_seed)
    elif args.command == "review":
        serve_review(args.bind, args.port)
    elif args.command == "score-a":
        print(json.dumps(score_a(), indent=1))
    elif args.command == "check-extraction":
        print(json.dumps(extraction_check(), indent=1))
    else:
        print(json.dumps(score_b(), indent=1))


if __name__ == "__main__":
    main()
