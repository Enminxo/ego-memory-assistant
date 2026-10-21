import numpy as np

from egomem.memory import EvidenceStore, MemoryEntry, TextualMemory, fmt_time


def entry(i: int, text: str = "x") -> MemoryEntry:
    return MemoryEntry(index=i, t0=15 * i, t1=15 * (i + 1), text=text, latency_s=1.0, n_frames=8)


def test_fmt_time():
    assert fmt_time(0) == "00:00"
    assert fmt_time(75.4) == "01:15"
    assert fmt_time(479.9) == "08:00"


def test_jsonl_roundtrip_keeps_order(tmp_path):
    path = tmp_path / "m" / "memory.jsonl"
    memory = TextualMemory(path)
    for i in (1, 0, 2):  # appended out of order, as after a resumed run
        memory.append(entry(i, f"clip {i}"))
    loaded = TextualMemory.load(path)
    assert [e.index for e in loaded.entries()] == [0, 1, 2]
    assert loaded.entries()[1].text == "clip 1"


def test_until_only_returns_finished_clips():
    memory = TextualMemory()
    for i in range(3):
        memory.append(entry(i))
    assert [e.index for e in memory.until(29.9)] == [0]
    assert [e.index for e in memory.until(30)] == [0, 1]
    assert memory.until(5) == []


def test_render_uses_mm_ss():
    assert TextualMemory.render([entry(4, "I open the fridge.")]) == "[01:00-01:15] I open the fridge."


def test_evidence_store(tmp_path):
    store = EvidenceStore(tmp_path / "frames")
    store.save(3, [np.zeros((540, 720, 3), dtype=np.uint8)] * 2, width=360)
    assert store.count(3) == 2
    assert store.path(3, 1).exists()
