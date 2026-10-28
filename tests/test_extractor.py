"""Unit tests for URL feature extraction (REQ-006, REQ-007, REQ-008)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Implementation.url_feature_extractor import (  # noqa: E402
    URL_FEATURE_NAMES, extract_url_features)


def test_returns_97_features_in_order():                      # TC-01
    f = extract_url_features("https://example.com/login")
    assert len(f) == 97 and list(f) == URL_FEATURE_NAMES


def test_absent_components_use_minus_one():                   # TC-02
    f = extract_url_features("example.com")
    for k in ("directory_length", "file_length", "params_length",
              "qty_dot_directory", "qty_params"):
        assert f[k] == -1


def test_path_and_query_are_split():                          # TC-03
    f = extract_url_features("http://a.com/x/y/z.php?id=1&u=2")
    assert f["directory_length"] == 10 and f["file_length"] == 5
    assert f["qty_params"] == 2 and f["qty_slash_url"] == 3


def test_scheme_not_counted():                                # TC-04
    assert extract_url_features("https://a.com")["qty_slash_url"] == 0


def test_ip_host_detected():                                  # TC-05
    assert extract_url_features("http://192.168.1.1/a")["domain_in_ip"] == 1


@pytest.mark.parametrize("bad", ["", "   ", "http://", "a b.com"])
def test_invalid_input_rejected(bad):                         # TC-06..09
    with pytest.raises(ValueError):
        extract_url_features(bad)
