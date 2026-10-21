# Asistente de memoria episódica (prototipo)

Prototipo de un asistente que registra lo que hace una persona con una cámara llevable y responde preguntas sobre su pasado reciente: «¿dónde he dejado las llaves?», «¿he apagado el gas?». Está pensado para deterioro cognitivo leve y fases iniciales de Alzheimer.

```
vídeo ─▶ clips de 15 s ─▶ [descriptor · VLM] ─▶ diario de texto con hora
                                                        │
                     pregunta ─▶ [lector · LLM] ◀───────┘─▶ respuesta + momento citado
```

1. **Descriptor.** Un modelo de visión y lenguaje mira 8 fotogramas de cada clip y escribe una línea de diario en primera persona.
2. **Memoria.** El diario se guarda como texto con marcas de tiempo (`memory.jsonl`). Es lo único que necesita el lector.
3. **Lector.** Un modelo de lenguaje responde leyendo solo el diario y cita el momento en que se basa.

La arquitectura sigue a Lando, Forte y Furnari, *Exploring Multimodal LMMs for Online Episodic Memory Question Answering on the Edge* (arXiv:2602.22455).

## Instalación

```bash
uv sync
```

Los modelos se sirven con cualquier servidor compatible con la API de OpenAI. En el DGX se usa [Ollama](https://ollama.com):

```bash
ollama pull gemma4:31b      # descriptor y lector
```

El servidor y los modelos se configuran en `configs/demo.yaml`. Para usar llama-server o vLLM basta con cambiar `base_url` y `model`.

## Escenas de demostración

Son 4 clips de 8 minutos de [Ego4D](https://ego4d-data.org) con preguntas de QAEgo4D, una escena por necesidad: gas, llaves, medicación y cartera (`scripts/demo_scenes.yaml`). **Hace falta firmar la licencia de Ego4D** para descargar los vídeos.

```bash
./scripts/make_demo_scenes.sh           # recorta los vídeos y crea scenarios/*.json
```

## Uso

```bash
uv run egomem-build gas keys meds wallet           # vídeo -> diario (data/memories/<escena>/default/)
uv run egomem-eval gas keys meds wallet            # acierto en preguntas de 4 opciones
uv run egomem-eval gas keys meds wallet --no-memory   # control: el mismo lector sin diario
uv run egomem-app                                  # demo web en http://localhost:8090
uv run pytest                                      # tests
```

El control `--no-memory` es imprescindible. Las opciones incorrectas de QAEgo4D se generaron automáticamente y a menudo son absurdas («en el techo», «a quién le importa»), así que un modelo acierta bastante más que el 25 % del azar sin ver nada. Lo que demuestra que la memoria funciona es la diferencia con ese control.

En la evaluación el lector está obligado a elegir una letra (decodificación restringida a A-D). Si se le deja, con el diario delante se niega a elegir cuando el diario no menciona lo que se pregunta, y cada negativa cuenta como fallo solo para el brazo con diario.

## Añadir una grabación propia

Una escena es un JSON en `scenarios/` con el vídeo y sus preguntas (el formato está descrito en `src/egomem/scenarios.py`). Para una grabación propia basta con escribirlo a mano:

```json
{
  "id": "pastilla", "title": "¿Me he tomado la pastilla?", "category": "medication",
  "video": "data/scenes/pastilla.mp4",
  "questions": [{"id": "p1", "question": "Did I take my pill?", "answer": "yes",
                 "wrong_answers": ["no", "I took two", "I only opened the box"],
                 "t0": 95.0, "t1": 102.0, "category": "medication"}],
  "demo_questions": [{"text": "¿Me he tomado la pastilla?", "expected": "sí"}]
}
```

## Estructura

| Fichero | Qué hace |
|---|---|
| `src/egomem/sources.py` | Lee fotogramas de un vídeo en instantes concretos |
| `src/egomem/clips.py` | Corta el vídeo en clips y elige qué fotogramas mira el descriptor |
| `src/egomem/describer.py` | Clip → línea de diario (punto de enganche para otros modelos) |
| `src/egomem/prompts.py` | Prompts del descriptor y del lector |
| `src/egomem/memory.py` | El diario y los fotogramas de evidencia |
| `src/egomem/reader.py` | Diario + pregunta → respuesta con citas |
| `src/egomem/scenarios.py` | Formato de escena y creación desde Ego4D |
| `src/egomem/build_memory.py`, `evaluate.py`, `app.py` | Los tres comandos |
