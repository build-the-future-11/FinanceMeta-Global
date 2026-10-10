import json

import pytest

from market_data_integrity.core import validate_csv


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,10,12,9,11,100\n", encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"expected_interval_seconds": 60, "provenance": {"source": "synthetic"}}))
    return source, config


@pytest.mark.parametrize("raw", [
    "timestamp,open,high,low,close,close,volume\n2026-01-01,10,12,9,9999,11,100\n",
    "timestamp,open,high,low,close,volume\n2026-01-01,10,12,9,11,100,extra\n",
    "timestamp,open,high,low,close,volume,note\n2026-01-01,10,12,9,11,100\n",
    "timestamp,open,high,low,close,volume,\n2026-01-01,10,12,9,11,100,note\n",
])
def test_ambiguous_or_misaligned_csv_cannot_pass(inputs, raw):
    source, config = inputs
    source.write_text(raw, encoding="utf-8")
    assert validate_csv(source, config)["overall"] == "FAIL"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, False, 0, -1])
def test_invalid_interval_cannot_be_silently_coerced(inputs, value):
    source, config = inputs
    payload = json.loads(config.read_text())
    payload["expected_interval_seconds"] = value
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite and positive"):
        validate_csv(source, config)


@pytest.mark.parametrize("value", [None, False, 0, {}, [], " "])
def test_provenance_requires_nonblank_source_text(inputs, value):
    source, config = inputs
    payload = json.loads(config.read_text())
    payload["provenance"]["source"] = value
    config.write_text(json.dumps(payload))
    assert validate_csv(source, config)["overall"] == "FAIL"


@pytest.mark.parametrize("mapping", [{"open": "close"}, {"open": ""}, {"open": None}, []])
def test_required_fields_need_distinct_usable_column_names(inputs, mapping):
    source, config = inputs
    payload = json.loads(config.read_text())
    payload["columns"] = mapping
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="columns must"):
        validate_csv(source, config)


def test_fractional_intervals_do_not_lose_precision(inputs):
    source, config = inputs
    payload = json.loads(config.read_text())
    payload["expected_interval_seconds"] = 0.5
    config.write_text(json.dumps(payload))
    source.write_text(source.read_text() + "2026-01-01T00:00:00.5Z,10,12,9,11,100\n")
    assert validate_csv(source, config)["overall"] == "PASS"


def test_fractional_gap_is_not_rounded_away(inputs):
    source, config = inputs
    source.write_text(source.read_text() + "2026-01-01T00:01:00.5Z,10,12,9,11,100\n")
    report = validate_csv(source, config)
    assert report["overall"] == "FAIL"
    assert any(item["check"] == "expected_interval_gaps" and not item["passed"] for item in report["findings"])
