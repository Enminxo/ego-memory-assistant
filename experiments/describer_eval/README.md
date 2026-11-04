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

## Phase 1a: three checks before any training

Pre-registered on 2026-11-04, before any diary or evaluation of this phase was run.
Same 148 clips, 500 questions, bootstrap over clips (1,000 resamples, seed 0).

**1a-1. A prompt without example objects (the test that could kill distillation).**
New diary `e4b_noex`: gemma4:e4b with `--prompt-variant no_examples`
(`egomem.prompts.DESCRIBER_TASK_NO_EXAMPLES`): the same priorities without "(keys, wallet,
phone, glasses, bags)", plus the rule "Mention only objects and actions that appear in these
frames: do not list things I did not do or did not see". Written once, not tuned on these
results. g31b keeps the default prompt.

- **Primary:** B accuracy g31b minus e4b_noex (Gemma reader, seed 0).
- **Rule:** if its interval includes 0 or lies below 0, the prompt alone closes the gap and
  the SFT is not started as planned. Otherwise the SFT goes ahead, and its baseline is the
  better of e4b and e4b_noex on B.
- **Secondary:** B e4b_noex minus e4b; A0 precision of e4b_noex against e4b and against g31b
  (same rule and stop words as A0); share of windows naming keys, wallet, phone, glasses or
  bags, and share with a negation; by-eye check of 50 unsupported e4b_noex mentions, coded
  as in `eye_check_a0.json`.
- **Prediction:** the prompt removes most of the echo (keys well under 32.6% of windows)
  and helps B, but g31b stays ahead.

**1a-2. Option order.** All five diaries again with option-shuffling seed 1 (Gemma reader).
A comparison of 1a-1 or of B whose interval excluded 0 replicates if it keeps its sign with
seed 1; if the sign flips, it is reported as not established.

**1a-3. A reader from another family.** All five diaries with qwen3.8:27b as reader
(`config_reader_qwen27.yaml`, seed 0; thinking off, letter constrained as for Gemma). Checks
whether the g31b advantage is the Gemma reader preferring Gemma's own text. Key comparisons:
g31b minus no memory and g31b minus e4b. If g31b minus e4b includes 0 with this reader, the
g31b advantage is not established beyond the Gemma reader, and the SFT must report both
readers.

### Phase 1a results (4 Nov 2026)

All 4,811 e4b_noex windows written; 0 unparsed answers in the 15 reader runs. Ollama does
not reproduce a diary exactly with the same seed (the first window of a smoke run and of the
real run differ), so each diary is one draw.

**1a-1: the prompt does not close the gap, so the SFT goes ahead.** Primary g31b minus
e4b_noex: +7.4 points [3.5, 11.4] (Gemma reader, seed 0). It holds with seed 1 (+9.6
[5.4, 14.1]) and with the Qwen reader (+6.0 [1.9, 10.0]). The SFT baseline is e4b_noex
(46.0% against 43.6% for e4b).

| | e4b | e4b_noex | g31b | narration |
|---|---|---|---|---|
| windows naming keys | 32.6% | 0.1% | 0.3% | 0.0% |
| windows naming any example object | 63.2% | 7.6% | 16.7% | 6.4% |
| windows with a negation | 11.7% | 11.7% | 9.4% | 0.0% |
| A0 precision | 13.5% | 16.0% | 29.5% | |
| labels per window | 4.46 | 3.89 | 4.22 | |

The prompt removes the echo (keys from a third of the windows to 4 of 4,811) but not the
negations, despite the new rule. A0 precision rises by +2.5 [1.6, 3.4] and stays far below
g31b (g31b minus e4b_noex +13.5 [12.0, 15.1]). Its effect on B is positive with all three
readings but not established: +2.4 [-1.5, 6.5] (seed 0), +0.4 [-3.7, 4.5] (seed 1), +4.4
[1.1, 7.9] (Qwen reader).

By-eye check of 50 unsupported e4b_noex mentions (`eye_check_e4b_noex.json`): 0 invented
objects, 33 real objects the narration does not name, 8 negations, 6 words that are not
objects, 3 unclear (e4b with the default prompt: 20 invented). So without the echo, the
remaining A0 gap to g31b is mostly which objects get mentioned (g31b names the ones the
person handles, which the narration names too), plus negations and actions that do not
happen (seen in several sentences; A0 does not count actions).

**1a-2: option order.** Every comparison keeps its sign with seed 1. Narration minus g31b
falls to +4.8 [-0.2, 9.8]: the ceiling gap is small and not robust.

**1a-3: reader from another family.** With qwen3.8:27b: no memory 33.8%, e4b 45.0%,
e4b_noex 49.4%, g31b 55.4%, narration 59.8%. g31b minus e4b +10.4 [6.2, 14.6], the same as
with the Gemma reader (+9.8), so the g31b advantage is not the Gemma reader preferring
Gemma's text. Narration minus g31b +4.4 [-0.4, 8.8].

**Reading.** Distillation has about 6 to 10 points of B to close between e4b_noex and g31b.
That the gap is in what gets told rather than what gets seen fits the by-eye check (0 invented
objects) but is not established: e4b_noex names far fewer of the handled objects (recall -17.4
points against g31b), which could be either; phase 1b-0 tests it. Beyond g31b the measurable
room on QAEgo4D is small (narration about 5 points above, interval touching 0 in two of three
readings), which limits what phase 2 (RL) can show on this benchmark.

## Phase 1b: SFT of the small describer

Pre-registered on 2026-11-04, before any step of this phase was run.

**1b-0. Does e4b see what it does not tell? (minutes, before any training).** For 300
windows of test_close drawn at random (seed 0) among those where g31b names an object that
the narration also names and e4b_noex does not, ask a model with the same 8 frames "Is there
a {object} in these frames?", the answer constrained to yes or no. The object is the phrase
g31b wrote. Control, in the same window: an object from the same pool of phrases whose label
is not narrated anywhere in the clip and that neither diary names. Asked to gemma4:e4b (the
student) and gemma4:31b (reference).

- **Validity:** the probe is valid only if g31b says yes to at least 70% of the present
  objects and to at most 30% of the controls.
- **Primary:** e4b yes-rate on present objects minus yes-rate on controls, paired by window,
  95% interval over clips.
- **Rule:** lower bound at 30 points or more: e4b mostly sees what it leaves out, and the
  SFT freezes the vision tower. Interval including 0: e4b does not see them, so a
  frozen-vision SFT would teach guessing; the SFT then adds LoRA on the vision tower as a
  second arm. In between: freeze, and run the vision arm as a planned ablation.
- **Bias:** a control is "absent" because nobody named it, so some are visible; this
  inflates both models' false alarms alike.

**1b-1. Teacher data.** Scenarios for the 997 QAEgo4D train clips (all in NLQ train, no video
shared with test_close). 3 full 15 s windows per clip at random (seed 0), 2,991 windows. The
frames are saved as the exact JPEG bytes sent to Ollama (quality 90). Teacher: gemma4:31b
with the `no_examples` prompt, the same prompt the student gets, so the student learns the
teacher's answer to its own input. 10% of the clips (seed 0) are held out for validation
loss.

**1b-2. Training.** google/gemma-4-E4B-it in bf16 with Hugging Face transformers and peft:
LoRA rank 16, alpha 32, dropout 0.05, on the attention and MLP projections of the language
model only; vision tower and projector frozen (unless 1b-0 says otherwise). Loss on the
answer tokens only. At most 3 epochs, learning rate 1e-4 with cosine decay; the checkpoint
with the lowest validation loss is kept. A smoke test (one batch forward and backward,
memory, seconds per step) comes first.

**1b-3. Evaluation.** Base and student both run in Hugging Face bf16 with the same settings
(no_examples prompt, same 8 frames as JPEG, sampling at temperature 0.2, seed 0, at most 160
new tokens), so the comparison does not mix in Ollama's 4-bit quantization. Diaries
`e4b_hf` and `e4b_sft` on the 148 test_close clips.

- **Primary:** B accuracy e4b_sft minus e4b_hf (Gemma reader, option seed 0), paired, 95%
  interval over clips.
- **Secondary:** the same with the Qwen reader and option seed 1; A0 precision and recall;
  echo and negation shares; by-eye check of 50 unsupported e4b_sft mentions.
- **Kill:** the primary interval includes 0, or B rises while A0 precision falls (the
  student learned to guess); either way the SFT result is reported as such.
- **Prediction:** e4b_sft lands between e4b_hf and g31b on B, closing part of the 6-10 point
  gap.

## Files

- `config.yaml`: paths and settings (data under `~/data/ego-memory-assistant/describer_eval`).
- `run_diaries.sh`: writes the `e4b` and `g31b` diaries clip by clip (resumable).
- `narration_diary.py`: writes the `narration` diaries.
- `object_precision.py`: A0.
- `run_phase1a.sh`: phase 1a in one chain; `prompt_echo.py`: echo and negation counts;
  `config_reader_qwen27.yaml`: the same settings with qwen3.8:27b as reader.
