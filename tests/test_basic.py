import pytest
from signal_engine.score_and_suggest import score_stock

def test_score_stock():
    result = score_stock("AAPL")
    assert isinstance(result, dict)
    assert "score" in result
    assert 0 <= result["score"] <= 1 