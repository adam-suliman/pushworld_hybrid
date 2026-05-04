from pushworld.rl.tracking import (
    CometTrackingConfig,
    create_comet_tracker,
    parse_tags,
)


def test_noop_tracker_accepts_parameters_metrics_and_assets(tmp_path):
    tracker = create_comet_tracker(CometTrackingConfig(enabled=False))
    asset = tmp_path / "asset.txt"
    asset.write_text("hello")

    tracker.log_parameters({"a": 1, "nested": {"b": 2}})
    tracker.log_metrics({"x": 1.0, "empty": "", "none": None}, step=3, prefix="train")
    tracker.log_asset(str(asset), name="asset.txt")
    tracker.end()


def test_parse_tags():
    assert parse_tags("ppo, level0,,base ") == ("ppo", "level0", "base")
    assert parse_tags("") == ()
