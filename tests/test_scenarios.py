from egomem.evaluate import shuffled_options
from egomem.scenarios import Question, Scenario, categorize, load_scenario


def test_categorize():
    assert categorize("How many drug containers did i open?") == "medication"
    assert categorize("Did I turn OFF the gas cooker?") == "appliances"
    assert categorize("Did I leave the back door open?") == "appliances"
    assert categorize("whom did i talk to in the house?") == "people"
    assert categorize("where did I put my keys?") == "objects"
    assert categorize("what did l take on the floor?") == "objects"
    assert categorize("What word was written on the poster?") == "other"


def test_scenario_roundtrip(tmp_path):
    scenario = Scenario(id="s", title="t", category="objects", video="data/scenes/s.mp4",
                        questions=[Question(id="q1", question="Where?", answer="table",
                                            wrong_answers=["a", "b", "c"], t0=1.0, t1=2.0)],
                        demo_questions=[{"text": "Where?", "expected": "table"}])
    scenario.save(tmp_path / "s.json")
    loaded = load_scenario(tmp_path / "s.json")
    assert loaded == scenario


def test_live_scenario_roundtrip_without_video(tmp_path):
    scenario = Scenario(id="live-1", title="Live", category="other", live=True,
                        source={"neon": "192.168.43.1"})
    scenario.save(tmp_path / "live-1.json")
    loaded = load_scenario(tmp_path / "live-1.json")
    assert loaded == scenario and loaded.video is None and loaded.video_path(tmp_path) is None


def test_shuffled_options_deterministic_and_track_answer():
    options, correct = shuffled_options("q1", "table", ["a", "b", "c"], seed=0)
    assert options[correct] == "table" and sorted(options) == ["a", "b", "c", "table"]
    assert shuffled_options("q1", "table", ["a", "b", "c"], seed=0) == (options, correct)
    positions = {shuffled_options(f"q{i}", "table", ["a", "b", "c"], 0)[1] for i in range(40)}
    assert positions == {0, 1, 2, 3}  # the answer is not always first
