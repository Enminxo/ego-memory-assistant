# Describer evaluation: does the large describer invent fewer objects, and how much does the diary add?

Pre-registered on 2026-11-02, before any diary of this experiment was written.

Two questions share the same diaries, so they run together.

- **A0 (gate for post-training).** The plan is to distil the large describer
  (gemma4:31b) into the small one (gemma4:e4b) with SFT, and later to train it with RL
  rewarded by the reader's accuracy. Distillation only makes sense if the large describer
  invents fewer objects than the small one. A0 checks that first.
- **B (evaluation of the assistant).** The 59 questions of the four demo scenes cannot tell
  whether the diary helps (every 95% interval of the gain reached zero). B scores the
  reader on the 500 questions of QAEgo4D test_close, with a no-memory control and with a
  ceiling diary written from Ego4D's human narrations, which separates perception errors
  (the describer) from memory and reading errors (the reader).

## Data

- QAEgo4D `test_close` (GroundVQA's `annotations.QaEgo4D_test_close.json`): 500 four-option
  questions over 148 Ego4D NLQ clips (about 8 min each). All 148 full videos are on disk
  (`~/data/raw/ego4d/tasks/nlq/v2/video_540ss`). Each scenario is the clip as a segment of
  its full video (`egomem-scene --qaego4d-split test_close`), no re-encoding.
- Ego4D narrations (`narration.json`, `narration_pass_1`), in video time.
- Ego4D noun taxonomy (`narration_noun_taxonomy.csv`, 582 labels with their words).

## Diaries (same settings as configs/demo.yaml)

15 s windows, 8 frames each, max 60 words, temperature 0.2, seed 0.

| tag | written by |
|---|---|
| `e4b` | gemma4:e4b |
| `g31b` | gemma4:31b |
| `narration` | Ego4D narrations of the window (pass 1), `#C C` turned into `I`; empty windows read "(nothing narrated)" |

## A0: object precision against the narrations

For each window present in both `e4b` and `g31b`, objects are the taxonomy labels whose
words appear in the text (whole words, optional plural `s`/`es`, multi-word phrases
first). A label the describer mentions is **supported** if the narrations between
`t0 - 2 s` and `t1 + 2 s` mention the same label.

- **Primary endpoint:** pooled precision P = supported / mentioned, `g31b` minus `e4b`,
  paired on the same windows, 95% interval from 1,000 bootstrap resamples of clips
  (seed 0).
- **Stop words (removed before matching, primary only):** hand(s), finger(s), palm,
  thumb, left hand, right hand, person, people, man, men, woman, women, lady, they,
  himself, herself, shoulder, floor, ground, surface, land, compound. The describer names
  hands and floors in nearly every entry and the taxonomy maps "hand" to two labels, so
  these add noise without telling the models apart. Secondary: the same with no stop words.
- **Secondary:** recall (narrated labels the describer mentions), labels mentioned per
  window, latency per window.
- **Prediction:** P(g31b) > P(e4b).
- **Gate:** if the interval of P(g31b) - P(e4b) includes 0 or lies below it, the large
  describer does not invent fewer objects by this measure and distillation is not started.
- **Known bias:** narrations describe actions and do not name every visible object, so
  absolute precision is underestimated for both models; only the difference is read.
  Check by eye: 50 unsupported mentions of each model on their frames before trusting the
  number. Taxonomy groups are noisy (e.g. "stand" maps to table) and verbs can match nouns;
  this hits both models alike.

## B: accuracy of the reader

Reader gemma4:31b, constrained to a letter, option order fixed by seed 0
(`egomem-eval`), each diary read whole.

- **Primary endpoint:** accuracy with `g31b` minus accuracy with no memory, paired on the
  same questions, 95% interval from 1,000 bootstrap resamples of clips.
- **Secondary:** `narration` minus `g31b` (perception gap), `g31b` minus `e4b`, accuracy
  by question category.
- **Prediction:** narration > g31b > e4b > no memory.
- **Kill:** if the primary interval includes 0 on these 500 questions, "the diary helps"
  is not established on QAEgo4D test_close, and that is the result to report.

## Files

- `config.yaml`: paths and settings (data under `~/data/ego-memory-assistant/describer_eval`).
- `run_diaries.sh`: writes the `e4b` and `g31b` diaries clip by clip (resumable).
- `narration_diary.py`: writes the `narration` diaries.
- `object_precision.py`: A0.
