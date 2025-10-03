import json
import pandas as pd
import pytest

from ntr_converter import(
    numeric_or_nan,
    extract_number_pairs_from_label,
    infer_bands_from_price_columns,
    build_regions_from_tokens,
    validate_output,
    Region,
)

@pytest.mark.parametrize("input_val, expected", [
    ("12,5", 12.5),
    ("12.50", 12.5),
    ("12.000,5", 12000.5), 
    ("n/a", None),
    ("-", None),
    ("", None),
    ("abc", None),
    (42, 42.0),
    (None, None),
])
def test_numeric_or_nan(input_val, expected):
    assert numeric_or_nan(input_val) == expected

# to extract number pairs from label
@pytest.mark.parametrize("label, expected", [
    ("0-30kg", (0, 30)),
    ("31 to 70", (31, 70)),
    ("up to 100", (None, 100)),
    ("70kg", (None, 70)),
    ("Zone A", None),
    (None, None),
])
def test_extract_number_pairs_from_label(label, expected):
    assert extract_number_pairs_from_label(label) == expected

# to infer bands from price columns
def test_infer_bands_simple():
    cols = ["0-30", "31-70", "71+"]
    bands = infer_bands_from_price_columns(cols, unit_guess="kg")
    assert bands == [
        "0_up_to_30[kg][kg]",
        "31_up_to_70[kg][kg]",
        "71_up_to_9999999[kg][kg]",
    ]


def test_infer_bands_missing_low():
    cols = ["30", "60"]
    bands = infer_bands_from_price_columns(cols, unit_guess="kg")
    assert bands[0].startswith("0_up_to_30")
    assert bands[1].startswith("30_up_to_60")

# to build regions from tokens
def test_build_regions_new_tokens():
    tokens = ["Switzerland", "Zone A", "Zone B"]
    regions_df, id_map = build_regions_from_tokens(tokens)

    assert not regions_df.empty
    assert set(id_map.keys()) == {"Switzerland", "Zone A", "Zone B"}
    assert len(id_map.values()) == 3


def test_build_regions_extend_existing():
    existing = pd.DataFrame([
        Region(identifier_string="Switzerland").as_row(1)
    ])
    tokens = ["Switzerland", "Zone C"]
    regions_df, id_map = build_regions_from_tokens(tokens, existing)

    assert "Zone C" in id_map
    assert id_map["Switzerland"] == 1
    assert regions_df.shape[0] == 2

# to validate_output
def test_validate_output_ok():
    regions = pd.DataFrame([
        {"id": 1, "identifierstring": "Switzerland"},
        {"id": 2, "identifierstring": "Zone A"},
    ])
    tariffs = pd.DataFrame([
        {"route": json.dumps([1, 2])}
    ])
    assert validate_output(regions, tariffs) is True


def test_validate_output_fail_missing_region():
    regions = pd.DataFrame([
        {"id": 1, "identifierstring": "Switzerland"},
    ])
    tariffs = pd.DataFrame([
        {"route": json.dumps([1, 99])}
    ])
    assert validate_output(regions, tariffs) is False