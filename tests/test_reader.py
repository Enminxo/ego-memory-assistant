from egomem.describer import trim_incomplete_sentence
from egomem.memory import MemoryEntry
from egomem.prompts import describer_prompt, reader_prompt
from egomem.reader import Reader, parse_choice, parse_citations


def test_parse_citations():
    assert parse_citations("Las dejaste en la mesa [02:15].") == [135]
    assert parse_citations("Entre [02:15-02:30] y 07:45") == [135, 465]
    assert parse_citations("No aparece en el diario.") == []


def test_parse_choice_ignores_lowercase_article():
    assert parse_choice("B") == 1
    assert parse_choice("D. in the drawer") == 3
    assert parse_choice("It is on a table (C)") == 2
    assert parse_choice("no idea") is None


def test_trim_incomplete_sentence():
    assert trim_incomplete_sentence("I open the door. I pick up the") == "I open the door."
    assert trim_incomplete_sentence("I pick up the") == "I pick up the"


def test_prompts_contain_their_inputs():
    assert "00:15-00:30" in describer_prompt("00:15", "00:30", 8)
    p = reader_prompt("[00:00-00:15] I cook.", "Did I cook?", ["yes", "no", "maybe", "never"])
    assert p.index("I cook.") < p.index("Did I cook?") < p.index("A. yes")
    assert "(the diary is empty)" in reader_prompt("", "Where?")


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages = None

    def chat(self, messages, max_tokens=256, temperature=0.0, seed=None, response_format=None):
        from egomem.llm import ChatResult
        self.messages = messages
        self.response_format = response_format
        return ChatResult(self.reply, 0.1, None, None)


def test_reader_answer_and_choose():
    entries = [MemoryEntry(0, 0, 15, "I put the keys on the table.", 1.0, 8)]
    client = FakeClient("On the table [00:00].")
    answer = Reader(client).answer(entries, "Where are my keys?")
    assert answer.citations == [0] and answer.n_entries == 1
    assert "keys on the table" in client.messages[1]["content"]
    client.reply = '{"answer": "A"}'
    assert Reader(client).choose(entries, "Where?", ["table", "fridge", "sofa", "roof"])[0] == 0
    enum = client.response_format["json_schema"]["schema"]["properties"]["answer"]["enum"]
    assert enum == ["A", "B", "C", "D"]
    # The closed protocol forbids abstaining; the open answer allows it.
    assert "choose the most likely option anyway" in client.messages[0]["content"]
    Reader(client).answer(entries, "Where?")
    assert "say so plainly" in client.messages[0]["content"]

