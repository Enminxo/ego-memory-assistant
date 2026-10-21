# Episodic memory assistant (prototype)

Prototype of an assistant that records what a person does with a wearable camera and answers questions about their recent past: "where did I leave my keys?", "did I turn off the stove?". It is aimed at mild cognitive impairment and early Alzheimer's disease.

```
video ─▶ 15 s clips ─▶ [describer · VLM] ─▶ timestamped text diary
                                                     │
                  question ─▶ [reader · LLM] ◀───────┘─▶ answer + cited moment
```

1. **Describer.** A vision-language model looks at 8 frames of each clip and writes one first-person diary entry.
2. **Memory.** The diary is stored as timestamped text (`memory.jsonl`). It is all the reader needs.
3. **Reader.** A language model answers from the diary alone and cites the moment it relied on.

The architecture follows Lando, Forte and Furnari, *Exploring Multimodal LMMs for Online Episodic Memory Question Answering on the Edge* (arXiv:2602.22455).

## Installation

```bash
uv sync
```

Models are served by any server with an OpenAI-compatible API. On the DGX Spark we use [Ollama](https://ollama.com):

```bash
ollama pull gemma4:31b      # describer and reader
```

Server and models are set in `configs/demo.yaml`. To use llama-server or vLLM, change `base_url` and `model`.

## Demo scenes

Four 8-minute clips from [Ego4D](https://ego4d-data.org) with QAEgo4D questions, one scene per need: stove, keys, medication and wallet (`scripts/demo_scenes.yaml`). **Downloading the videos requires signing the Ego4D license.**

```bash
./scripts/make_demo_scenes.sh           # cuts the videos and writes scenarios/*.json
```

## Usage

```bash
uv run egomem-build gas keys meds wallet           # video -> diary (data/memories/<scene>/default/)
uv run egomem-eval gas keys meds wallet            # accuracy on four-option questions
uv run egomem-eval gas keys meds wallet --no-memory   # control: the same reader without a diary
uv run egomem-app                                  # web demo at http://localhost:8090
uv run pytest                                      # tests
```

The `--no-memory` control is essential. QAEgo4D's wrong options were generated automatically and are often absurd ("on the ceiling", "who cares"), so a model scores well above the 25% chance level without seeing anything. What shows that the memory works is the gap to that control.

In the evaluation the reader must always pick a letter (decoding constrained to A-D). Left free, with the diary in front of it, it declines to choose when the diary does not mention what is asked, and each refusal counts as a miss only for the arm that has a diary.

## Adding your own recording

A scene is a JSON file in `scenarios/` with the video and its questions (the format is described in `src/egomem/scenarios.py`). For your own recording, write it by hand:

```json
{
  "id": "pill", "title": "Did I take my pill?", "category": "medication",
  "video": "data/scenes/pill.mp4",
  "questions": [{"id": "p1", "question": "Did I take my pill?", "answer": "yes",
                 "wrong_answers": ["no", "I took two", "I only opened the box"],
                 "t0": 95.0, "t1": 102.0, "category": "medication"}],
  "demo_questions": [{"text": "Did I take my pill?", "expected": "yes"}]
}
```

## Layout

| File | What it does |
|---|---|
| `src/egomem/sources.py` | Reads frames from a video at given times |
| `src/egomem/clips.py` | Cuts the video into clips and picks which frames the describer sees |
| `src/egomem/describer.py` | Clip → diary entry (the hook for other models) |
| `src/egomem/prompts.py` | Prompts of the describer and the reader |
| `src/egomem/memory.py` | The diary and the evidence frames |
| `src/egomem/reader.py` | Diary + question → answer with citations |
| `src/egomem/scenarios.py` | Scene format and building scenes from Ego4D |
| `src/egomem/build_memory.py`, `evaluate.py`, `app.py` | The three commands |
