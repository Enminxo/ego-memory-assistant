import json

from fastapi.testclient import TestClient

from egomem.app import create_app
from egomem.memory import MemoryEntry, TextualMemory
from egomem.reader import Answer


class FakeReader:
    def __init__(self):
        self.seen = None

    def answer(self, entries, question):
        self.seen = entries
        return Answer(text="On the table [00:20].", citations=[20], latency_s=0.1,
                      n_entries=len(entries))


def make_app(tmp_path):
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "data/scenes").mkdir(parents=True)
    (tmp_path / "data/scenes/s.mp4").write_bytes(b"0123456789" * 100)
    (tmp_path / "scenarios/s.json").write_text(json.dumps({
        "id": "s", "title": "Scene", "category": "objects", "video": "data/scenes/s.mp4",
        "questions": [], "demo_questions": [{"text": "¿Dónde?", "expected": "table"}]}))
    memory = TextualMemory(tmp_path / "data/memories/s/default/memory.jsonl")
    for i in range(3):
        memory.append(MemoryEntry(i, 15 * i, 15 * (i + 1), f"clip {i}", 1.0, 8))
    config = {"scenarios_dir": "scenarios", "data_dir": "data"}
    reader = FakeReader()
    return TestClient(create_app(config, "default", reader=reader, root=tmp_path)), reader


def test_endpoints(tmp_path):
    client, reader = make_app(tmp_path)
    scenarios = client.get("/api/scenarios").json()
    assert scenarios[0]["id"] == "s" and scenarios[0]["n_entries"] == 3
    assert len(client.get("/api/scenarios/s/memory").json()["entries"]) == 3
    assert client.get("/api/scenarios/nope/memory").status_code == 404

    answer = client.post("/api/ask", json={"scenario": "s", "question": "Where?", "t": 31}).json()
    assert [e.index for e in reader.seen] == [0, 1]  # only clips finished by t=31
    assert answer["citations"][0]["index"] == 1
    client.post("/api/ask", json={"scenario": "s", "question": "Where?"})
    assert len(reader.seen) == 3  # no t: whole diary


def test_video_supports_range_requests(tmp_path):
    client, _ = make_app(tmp_path)
    response = client.get("/api/scenarios/s/video", headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.content == b"0123456789"
