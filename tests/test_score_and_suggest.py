import pytest
from signal_engine.score_and_suggest import score_stock

def test_score_stock_basic(mock_env_vars):
    result = score_stock("600519.SH", use_ai_model=False)
    assert "score" in result
    assert 0 <= result["score"] <= 1
    assert "suggest" in result
    assert result["suggest"] in ["✅ BUY", "HOLD"]

def test_score_stock_with_ai(mock_env_vars):
    result = score_stock("600519.SH", use_ai_model=True)
    assert "score" in result
    assert "reason" in result 