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

## Results (3 Nov 2026, one run, seed 0)

All 4,811 windows of the 148 clips were written by both describers. Median latency per
window: 1.8 s (e4b) and 9.6 s (g31b), both under the 15 s window.

**A0: the gate is passed.** Pooled precision 13.5% (e4b) and 29.5% (g31b); difference
+16.0 points [14.3, 17.7]. Secondary (no stop words): +15.6 [14.0, 17.2]. Recall: +18.1
[16.5, 19.7]. Both name about as many labels per window (4.5 and 4.2), so the gap is not
verbosity. Robustness to the narration margin (not pre-registered): +15.1 [13.5, 16.8]
with 0 s and +16.8 [15.1, 18.6] with 5 s.

By-eye check of the 50 sampled unsupported mentions per model, on the 8 frames of each
window (`eye_check_a0.json`, one rater, frames at 160x120):

| | invented | real, not narrated | negation | word is not that object | unclear |
|---|---|---|---|---|---|
| e4b | 20 | 15 | 7 | 3 | 5 |
| g31b | 0 | 40 | 1 | 6 | 3 |

So the absolute precision underestimates both describers, most of all g31b, and the
order holds. 0 of 50 is compatible with up to about 6% invented; 20 of 50 with 27-54%.

Most of e4b's inventions echo the prompt's list of example objects ("I picked up my keys
and placed them on the counter near the sink"). Over all windows, "keys" appears in 1,568
e4b entries (32.6%), 13 g31b entries and 1 narration window; any of keys, wallet, phone,
glasses or bags in 63.2% (e4b), 16.7% (g31b) and 6.4% (narration). Negations ("I do not
interact with any doors") appear in 11.7% (e4b) and 9.4% (g31b) of the windows, and the
taxonomy counts their objects as mentions.

**B: the diary helps.** Accuracy on the 500 questions: no memory 31.2%, e4b 43.6%,
g31b 53.4%, narration 59.4%. Primary g31b minus no memory: +22.2 points [17.7, 26.6].
Narration minus g31b +6.0 [0.7, 11.1]; g31b minus e4b +9.8 [5.5, 14.1]; e4b minus no
memory +12.4 [7.5, 16.9]. The predicted order holds. By category, g31b is at or above the
narration diary on "other" (50.7% vs 48.8%) and "people" (76.2% vs 66.7%, n = 21), so the
narration diary is a soft ceiling, not an upper bound.

**Not established / not run.** One option-shuffling seed; the reader is the same model as
the g31b describer, so g31b minus e4b may include self-preference (a reader from another
family would test it); the cheap alternative to distillation, e4b with a prompt without
the example objects, has not been run.

## Files

- `config.yaml`: paths and settings (data under `~/data/ego-memory-assistant/describer_eval`).
- `run_diaries.sh`: writes the `e4b` and `g31b` diaries clip by clip (resumable).
- `narration_diary.py`: writes the `narration` diaries.
- `object_precision.py`: A0.
