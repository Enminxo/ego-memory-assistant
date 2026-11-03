"""Prompts for the two roles of the pipeline.

The structure follows Lando, Forte and Furnari (arXiv:2602.22455, Figure 3):
the describer prompt has a task description, detailed instructions and a
question template; the reader gets the whole diary followed by the question.

What changes with respect to the paper is what the describer is told to
prioritise. The target users (mild cognitive impairment, early Alzheimer's)
mostly forget where they left personal objects, whether they took their
medication, and whether they switched off an appliance or closed a door, so
those come first. Changing these priorities is the cheapest experiment in the
whole project and the first thing worth ablating.
"""

from __future__ import annotations

# --- Describer (sees frames) -------------------------------------------------

DESCRIBER_TASK = (
    "These {n_frames} frames span {start}-{end} of an egocentric video recorded "
    "by a camera wearer ('I'), in chronological order. Write a short "
    "first-person diary entry of this clip.\n"
    "Prioritise, in this order:\n"
    "1. Personal objects I pick up or put down (keys, wallet, phone, glasses, "
    "bags) and exactly where I leave them.\n"
    "2. Medication, food and drink: what I take, open, eat or drink.\n"
    "3. Appliances, taps, doors and windows I switch on or off, open or close, "
    "and their state when I leave them.\n"
    "4. Other people present and what we do together.\n"
    "5. Other actions, in the order they happen, and where I am.\n"
)

DESCRIBER_QUESTIONS = (
    "Record the details needed to later answer questions like: 'Where did I "
    "leave X?', 'Did I take my pill?', 'Did I turn off the stove?', 'Did I "
    "close the door?', 'Who did I talk to?', 'How many X were there?'\n"
)

DESCRIBER_RULES = (
    "At most {max_words} words. State only what is visible; if something is "
    "unclear, say so instead of guessing. Do not describe the video itself."
)


# Variant without the list of example objects (experiments/describer_eval, phase 1a):
# on QAEgo4D the small describer echoed "keys, wallet, phone" into a third of its
# entries, and wrote negations ("I do not interact with any doors"). Same priorities,
# no example objects, and an explicit rule against listing what did not happen.
DESCRIBER_TASK_NO_EXAMPLES = DESCRIBER_TASK.replace(
    "1. Personal objects I pick up or put down (keys, wallet, phone, glasses, "
    "bags) and exactly where I leave them.\n",
    "1. Objects I pick up or put down, and exactly where I leave them.\n")
assert DESCRIBER_TASK_NO_EXAMPLES != DESCRIBER_TASK, "the example list was not found in DESCRIBER_TASK"

DESCRIBER_RULES_NO_EXAMPLES = (
    "At most {max_words} words. State only what is visible; if something is "
    "unclear, say so instead of guessing. Mention only objects and actions that "
    "appear in these frames: do not list things I did not do or did not see. "
    "Do not describe the video itself."
)

PROMPT_VARIANTS: dict[str, tuple[str, str]] = {
    "default": (DESCRIBER_TASK, DESCRIBER_RULES),
    "no_examples": (DESCRIBER_TASK_NO_EXAMPLES, DESCRIBER_RULES_NO_EXAMPLES),
}


def describer_prompt(start: str, end: str, n_frames: int, max_words: int = 60,
                     variant: str = "default") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; known: {sorted(PROMPT_VARIANTS)}")
    task, rules = PROMPT_VARIANTS[variant]
    return (task.format(n_frames=n_frames, start=start, end=end)
            + DESCRIBER_QUESTIONS
            + rules.format(max_words=max_words))


# --- Reader (sees only text) -------------------------------------------------

READER_SYSTEM = (
    "You help a person remember what they did, using only a timestamped diary "
    "written automatically from their wearable camera. The diary is your only "
    "evidence: never invent details it does not contain. Answer in one or two "
    "short sentences, in the same language as the question, and cite the "
    "moment you relied on as [mm:ss]. If the diary does not contain the "
    "answer, say so plainly."
)

READER_PROMPT = (
    "Diary:\n{memory}\n\n"
    "Question: {question}\n"
)

# Closed protocol (QAEgo4D-Closed): the reader must always pick an option. With
# READER_SYSTEM it is allowed to abstain, which is right in the app but turns
# the multiple choice into a measure of how often it abstains. Even with this
# prompt it still declines when the diary does not mention the object, so the
# letter is also enforced by constrained decoding (see Reader.choose).
READER_CHOICE_SYSTEM = (
    "You answer multiple-choice questions about what a camera wearer ('I') "
    "did, using a timestamped diary written automatically from their wearable "
    "camera. Choose the option best supported by the diary. If the diary does "
    "not settle it, choose the most likely option anyway. Reply with the "
    "letter only."
)

EMPTY_MEMORY = "(the diary is empty)"

# The closed protocol of QAEgo4D-Closed, used for evaluation.
MULTIPLE_CHOICE_SUFFIX = (
    "Options:\n{options}\n"
    "Answer with the letter of the correct option only."
)

LETTERS = "ABCD"


def reader_prompt(memory: str, question: str, options: list[str] | None = None) -> str:
    """Diary first, question last.

    The diary only grows at the end, so an unchanged prefix lets servers with
    prompt caching (llama.cpp, vLLM) skip re-reading it for every question.
    """
    prompt = READER_PROMPT.format(memory=memory or EMPTY_MEMORY, question=question)
    if options is not None:
        listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        prompt += MULTIPLE_CHOICE_SUFFIX.format(options=listed)
    return prompt
