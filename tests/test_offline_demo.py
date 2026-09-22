from pathlib import Path
import run_offline_demo as demo


def test_regions_nonempty():
    assert len(demo.load_regions()) >= 1


def test_metrics_helpers():
    assert demo.jaccard({"a"}, {"a", "b"}) == 0.5
    assert demo.bleu1(["a", "b"], ["a", "c"]) == 0.5


def test_config_optional():
    # demo must run even if config shape varies
    assert Path(demo.ROOT).exists()
