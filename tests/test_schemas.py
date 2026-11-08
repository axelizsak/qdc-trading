"""Every shared format: its mock is valid, and typical mistakes are rejected."""
import numpy as np
import pandas as pd
import pytest

from data import schemas
from data.mock import (
    MOCKS,
    mock_agent_weights,
    mock_evidence,
    mock_features,
    mock_holdings,
    mock_macro_views,
    mock_source_registry,
)


@pytest.mark.parametrize("name", list(MOCKS))
def test_mock_matches_its_format(name):
    df = MOCKS[name]()
    columns = getattr(schemas, name.upper())
    assert list(df.columns) == columns
    getattr(schemas, f"validate_{name}")(df)


def test_mocks_survive_a_csv_round_trip(tmp_path):
    for name, make in MOCKS.items():
        path = tmp_path / f"{name}.csv"
        make().to_csv(path, index=False)
        getattr(schemas, f"validate_{name}")(pd.read_csv(path))


def test_split_list():
    assert schemas.split_list("a; b;") == ["a", "b"]
    assert schemas.split_list(np.nan) == []


def test_holdings_rejects_bad_rows():
    df = mock_holdings()
    with pytest.raises(ValueError, match="units"):
        schemas.validate_holdings(df.assign(units=0))
    with pytest.raises(ValueError, match="duplicate"):
        schemas.validate_holdings(pd.concat([df, df.head(1)]))


def test_features_rejects_bad_rows():
    df = mock_features()
    with pytest.raises(ValueError, match="finite"):
        schemas.validate_features(df.assign(value=np.nan))
    with pytest.raises(ValueError, match="duplicate"):
        schemas.validate_features(pd.concat([df, df.head(1)]))


def test_agent_weights_rejects_bad_rows():
    df = mock_agent_weights()
    with pytest.raises(ValueError, match="sum to 1"):
        schemas.validate_agent_weights(df.assign(weight=0.1))
    with pytest.raises(ValueError, match="unknown agent"):
        schemas.validate_agent_weights(df.assign(agent="technical"))
    with pytest.raises(ValueError, match="hit_rate"):
        schemas.validate_agent_weights(df.assign(hit_rate=1.5))


def test_evidence_rejects_bad_rows():
    df = mock_evidence()
    with pytest.raises(ValueError, match="unknown kind"):
        schemas.validate_evidence(df.assign(kind="rumour"))
    with pytest.raises(ValueError, match="unknown factor"):
        schemas.validate_evidence(df.assign(factor="weather"))
    with pytest.raises(ValueError, match="numeric value"):
        schemas.validate_evidence(df.assign(value=np.nan))
    with pytest.raises(ValueError, match="summary"):
        schemas.validate_evidence(df.assign(summary=""))


def test_macro_views_rejects_bad_rows():
    df = mock_macro_views()
    with pytest.raises(ValueError, match="direction"):
        schemas.validate_macro_views(df.assign(direction=2.0))
    with pytest.raises(ValueError, match="source"):
        schemas.validate_macro_views(df.assign(sources=""))
    with pytest.raises(ValueError, match="duplicate"):
        schemas.validate_macro_views(pd.concat([df, df.head(1)]))


def test_source_registry_rejects_bad_rows():
    df = mock_source_registry()
    with pytest.raises(ValueError, match="unknown status"):
        schemas.validate_source_registry(df.assign(status="paused"))
    with pytest.raises(ValueError, match="known factors"):
        schemas.validate_source_registry(df.assign(factors="oil;weather"))
    with pytest.raises(ValueError, match="n_uses"):
        schemas.validate_source_registry(df.assign(n_uses=-1))
    with pytest.raises(ValueError, match="duplicate"):
        schemas.validate_source_registry(pd.concat([df, df.head(1)]))
