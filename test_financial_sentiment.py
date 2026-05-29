"""
test_financial_sentiment.py
────────────────────────────
pytest test suite for the redesigned Financial Sentiment Backtesting System.

Run
───
    pytest tests/test_financial_sentiment.py -v
    pytest tests/test_financial_sentiment.py -v -s
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Shared helpers ────────────────────────────────────────────

def _make_price_df(
    start: str = "2025-01-01",
    periods: int = 90,
    price: float = 100.0,
    trend: float = 0.0,
) -> pd.DataFrame:
    idx    = pd.bdate_range(start=start, periods=periods)
    closes = [price + i * trend + np.random.normal(0, 0.1) for i in range(periods)]
    return pd.DataFrame({"Close": closes, "Volume": [1_000_000] * periods}, index=idx)


def _make_pre_analyzed(n: int = 5) -> list[dict]:
    sentiments = ["positive", "slightly_positive", "neutral",
                  "slightly_negative", "negative"]
    records = []
    for i in range(n):
        records.append({
            "symbol":          "2330.TW",
            "name":            "台積電",
            "date":            f"2025-01-{i + 1:02d}",
            "title":           f"Test headline {i}",
            "source":          "test",
            "url":             "#",
            "sentiment_type":  sentiments[i % 5],
            "sentiment_label": "正向",
            "reason":          "test reason",
        })
    return records


# ════════════════════════════════════════════════════════════════
# 1. Dataset loading
# ════════════════════════════════════════════════════════════════

class TestDatasetLoading:

    def test_load_returns_list(self, tmp_path):
        from tests.backtest_engine import load_dataset
        data = _make_pre_analyzed(3)
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_dataset(str(p))
        assert isinstance(loaded, list)
        assert len(loaded) == 3

    def test_empty_dataset_raises(self, tmp_path):
        from tests.backtest_engine import load_dataset
        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        with pytest.raises(RuntimeError, match="build_dataset"):
            load_dataset(str(p))

    def test_schema_has_required_keys(self, tmp_path):
        from tests.backtest_engine import load_dataset
        data = _make_pre_analyzed(2)
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_dataset(str(p))
        for rec in loaded:
            assert "symbol" in rec
            assert "date" in rec
            assert "sentiment_type" in rec

    def test_date_format_yyyy_mm_dd(self, tmp_path):
        import re
        from tests.backtest_engine import load_dataset
        data = _make_pre_analyzed(3)
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        for rec in load_dataset(str(p)):
            assert re.match(r"\d{4}-\d{2}-\d{2}", rec["date"])


# ════════════════════════════════════════════════════════════════
# 2. Direction / sentiment mappings
# ════════════════════════════════════════════════════════════════

class TestDirectionMapping:

    def test_return_to_direction_all_buckets(self):
        from tests.utils import return_to_direction
        assert return_to_direction(3.0)   == "bullish"
        assert return_to_direction(2.0)   == "bullish"
        assert return_to_direction(1.0)   == "weak_bullish"
        assert return_to_direction(0.5)   == "weak_bullish"
        assert return_to_direction(0.0)   == "neutral"
        assert return_to_direction(-0.4)  == "neutral"
        assert return_to_direction(-1.0)  == "weak_bearish"
        assert return_to_direction(-2.0)  == "weak_bearish"
        assert return_to_direction(-3.0)  == "bearish"

    def test_sentiment_to_direction_all_five(self):
        from tests.utils import SENTIMENT_TO_DIRECTION
        assert SENTIMENT_TO_DIRECTION["positive"]          == "bullish"
        assert SENTIMENT_TO_DIRECTION["slightly_positive"] == "weak_bullish"
        assert SENTIMENT_TO_DIRECTION["neutral"]           == "neutral"
        assert SENTIMENT_TO_DIRECTION["slightly_negative"] == "weak_bearish"
        assert SENTIMENT_TO_DIRECTION["negative"]          == "bearish"

    def test_predicted_is_bullish(self):
        from tests.utils import predicted_is_bullish
        assert predicted_is_bullish("bullish")      is True
        assert predicted_is_bullish("weak_bullish") is True
        assert predicted_is_bullish("bearish")      is False
        assert predicted_is_bullish("weak_bearish") is False
        assert predicted_is_bullish("neutral")      is None

    def test_binary_correct(self):
        from tests.utils import binary_correct
        assert binary_correct("bullish",      "bullish")      is True
        assert binary_correct("weak_bullish", "bullish")      is True
        assert binary_correct("bearish",      "bullish")      is False
        assert binary_correct("bullish",      "neutral")      is None
        assert binary_correct("neutral",      "bullish")      is None

    def test_future_days_constant(self):
        from tests.utils import FUTURE_DAYS
        assert FUTURE_DAYS == [1, 3, 5, 10, 20, 60]


# ════════════════════════════════════════════════════════════════
# 3. Wilson confidence interval
# ════════════════════════════════════════════════════════════════

class TestWilsonCI:

    def test_perfect_accuracy(self):
        from tests.utils import wilson_ci
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.95

    def test_zero_accuracy(self):
        from tests.utils import wilson_ci
        lo, hi = wilson_ci(0, 100)
        assert hi < 0.05

    def test_empty_returns_zeros(self):
        from tests.utils import wilson_ci
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_50_pct_ci_contains_half(self):
        from tests.utils import wilson_ci
        lo, hi = wilson_ci(50, 100)
        assert lo < 0.5 < hi

    def test_bounds_in_unit_interval(self):
        from tests.utils import wilson_ci
        for n_s, n_t in [(0, 10), (5, 10), (10, 10), (1, 1000), (999, 1000)]:
            lo, hi = wilson_ci(n_s, n_t)
            assert 0.0 <= lo <= hi <= 1.0


# ════════════════════════════════════════════════════════════════
# 4. Future returns – all horizons
# ════════════════════════════════════════════════════════════════

class TestFutureReturns:

    @patch("tests.backtest_engine._get_cached_history")
    def test_all_horizons_populated(self, mock_hist):
        from tests.backtest_engine import get_future_returns, FUTURE_DAYS
        # 61 trading days so all horizons (max=60) have a next day
        df = _make_price_df("2025-01-02", periods=65, price=100.0)
        df["Close"] = [100.0 + i for i in range(65)]
        mock_hist.return_value = df
        result = get_future_returns("2330.TW", "2025-01-02")
        for h in FUTURE_DAYS:
            assert result[f"future_return_{h}d"] is not None, f"{h}d missing"

    @patch("tests.backtest_engine._get_cached_history")
    def test_60d_none_when_insufficient_data(self, mock_hist):
        from tests.backtest_engine import get_future_returns
        df = _make_price_df("2025-01-02", periods=10, price=100.0)
        mock_hist.return_value = df
        result = get_future_returns("2330.TW", "2025-01-02")
        assert result["future_return_60d"] is None

    @patch("tests.backtest_engine._get_cached_history")
    def test_1d_return_formula(self, mock_hist):
        from tests.backtest_engine import get_future_returns
        df = _make_price_df("2025-01-02", periods=65, price=100.0)
        df["Close"] = [float(100 + i) for i in range(65)]
        mock_hist.return_value = df
        result = get_future_returns("2330.TW", "2025-01-02")
        # t0 = 100.0, t1 = 101.0 → +1.0%
        assert abs(result["future_return_1d"] - 1.0) < 0.01

    @patch("tests.backtest_engine._get_cached_history")
    def test_empty_history_all_none(self, mock_hist):
        from tests.backtest_engine import get_future_returns, FUTURE_DAYS
        mock_hist.return_value = pd.DataFrame()
        result = get_future_returns("2330.TW", "2025-01-02")
        for h in FUTURE_DAYS:
            assert result[f"future_return_{h}d"] is None

    @patch("tests.backtest_engine._get_cached_history")
    def test_uptrend_positive_short_horizons(self, mock_hist):
        from tests.backtest_engine import get_future_returns
        df = _make_price_df("2025-01-02", periods=65, price=100.0)
        df["Close"] = [100.0 + i * 2 for i in range(65)]  # +2/day
        mock_hist.return_value = df
        result = get_future_returns("2330.TW", "2025-01-02")
        assert result["future_return_1d"]  > 0
        assert result["future_return_3d"]  > 0
        assert result["future_return_5d"]  > 0
        assert result["future_return_10d"] > 0
        assert result["future_return_20d"] > 0
        assert result["future_return_60d"] > 0


# ════════════════════════════════════════════════════════════════
# 5. run_backtest – reads sentiment from dataset
# ════════════════════════════════════════════════════════════════

class TestRunBacktest:

    @patch("tests.backtest_engine._get_cached_history")
    def test_sentiment_read_from_dataset(self, mock_hist):
        from tests.backtest_engine import run_backtest
        df = _make_price_df("2025-01-01", periods=90, price=100.0)
        df["Close"] = [100.0 + i * 0.5 for i in range(90)]
        mock_hist.return_value = df
        dataset = _make_pre_analyzed(3)
        result  = run_backtest(dataset, primary_horizon=3)
        assert len(result) == 3
        assert "ai_sentiment" in result.columns

    @patch("tests.backtest_engine._get_cached_history")
    def test_all_horizon_columns_present(self, mock_hist):
        from tests.backtest_engine import run_backtest, FUTURE_DAYS
        df = _make_price_df("2025-01-01", periods=90)
        mock_hist.return_value = df
        dataset = _make_pre_analyzed(2)
        result  = run_backtest(dataset)
        for h in FUTURE_DAYS:
            assert f"future_return_{h}d" in result.columns

    @patch("tests.backtest_engine._get_cached_history")
    def test_records_without_sentiment_skipped(self, mock_hist):
        from tests.backtest_engine import run_backtest
        df = _make_price_df("2025-01-01", periods=90)
        mock_hist.return_value = df
        dataset = [{"symbol": "2330.TW", "date": "2025-01-02",
                    "title": "test", "name": "test"}]  # no sentiment_type
        result = run_backtest(dataset)
        assert len(result) == 0

    @patch("tests.backtest_engine._get_cached_history")
    def test_predicted_direction_mapped_correctly(self, mock_hist):
        from tests.backtest_engine import run_backtest
        df = _make_price_df("2025-01-01", periods=90)
        mock_hist.return_value = df
        dataset = [{
            "symbol": "2330.TW", "name": "台積電", "date": "2025-01-02",
            "title": "positive news", "source": "test", "url": "#",
            "sentiment_type": "positive",
        }]
        result = run_backtest(dataset)
        assert result.iloc[0]["predicted_direction"] == "bullish"


# ════════════════════════════════════════════════════════════════
# 6. compute_metrics – multi-horizon
# ════════════════════════════════════════════════════════════════

class TestComputeMetrics:

    def _base_df(self, n: int = 30) -> pd.DataFrame:
        """Build synthetic result df with all horizon columns."""
        from tests.utils import FUTURE_DAYS
        rng = np.random.default_rng(42)
        rows = []
        sents = ["positive", "negative", "neutral", "slightly_positive", "slightly_negative"]
        for i in range(n):
            row = {
                "symbol":              "2330.TW",
                "name":                "台積電",
                "date":                f"2025-01-{i % 28 + 1:02d}",
                "ai_sentiment":        sents[i % 5],
                "predicted_direction": "bullish",
                "actual_direction":    "bullish",
                "is_correct":          bool(rng.integers(0, 2)),
            }
            for h in FUTURE_DAYS:
                row[f"future_return_{h}d"] = float(rng.normal(0.5, 2.0))
            rows.append(row)
        return pd.DataFrame(rows)

    def test_horizon_metrics_all_present(self):
        from tests.backtest_engine import compute_metrics
        df = self._base_df(30)
        metrics = compute_metrics(df, primary_horizon=3)
        hm = metrics.get("horizon_metrics", {})
        from tests.utils import FUTURE_DAYS
        for h in FUTURE_DAYS:
            assert f"{h}d" in hm, f"{h}d missing from horizon_metrics"

    def test_each_horizon_has_accuracy(self):
        from tests.backtest_engine import compute_metrics
        df = self._base_df(30)
        metrics = compute_metrics(df)
        for key, stats in metrics["horizon_metrics"].items():
            assert "binary_accuracy" in stats, f"{key} missing binary_accuracy"
            assert 0 <= stats["binary_accuracy"] <= 1

    def test_per_stock_present(self):
        from tests.backtest_engine import compute_metrics
        df = self._base_df(30)
        metrics = compute_metrics(df)
        assert "2330.TW" in metrics.get("per_stock", {})

    def test_wilson_ci_in_per_stock(self):
        from tests.backtest_engine import compute_metrics
        df = self._base_df(50)
        metrics = compute_metrics(df)
        sym_stats = metrics["per_stock"]["2330.TW"]
        for h_label, h_data in sym_stats["horizons"].items():
            assert "ci_lower" in h_data
            assert "ci_upper" in h_data
            assert h_data["ci_lower"] <= h_data["ci_upper"]

    def test_sentiment_returns_present(self):
        from tests.backtest_engine import compute_metrics
        df = self._base_df(30)
        metrics = compute_metrics(df, primary_horizon=3)
        assert "sentiment_returns" in metrics

    def test_perfect_accuracy_horizon(self):
        from tests.backtest_engine import compute_metrics, FUTURE_DAYS
        rows = []
        for i in range(20):
            row = {"symbol": "2330.TW", "name": "台積電",
                   "date": "2025-01-01",
                   "ai_sentiment": "positive",
                   "predicted_direction": "bullish",
                   "actual_direction": "bullish",
                   "is_correct": True}
            for h in FUTURE_DAYS:
                row[f"future_return_{h}d"] = 5.0   # strongly positive
            rows.append(row)
        df = pd.DataFrame(rows)
        metrics = compute_metrics(df, primary_horizon=3)
        h3 = metrics["horizon_metrics"]["3d"]
        assert h3["binary_accuracy"] == 1.0

    def test_evaluated_backward_compat(self):
        """'evaluated' key must still exist for backward compat."""
        from tests.backtest_engine import compute_metrics
        df = self._base_df(30)
        metrics = compute_metrics(df)
        assert "evaluated" in metrics


# ════════════════════════════════════════════════════════════════
# 7. Export
# ════════════════════════════════════════════════════════════════

class TestExport:

    def test_csv_and_json_created(self, tmp_path):
        from tests.backtest_engine import export_results
        from tests.utils import FUTURE_DAYS
        rows = []
        for i in range(3):
            row = {"symbol": "2330.TW", "name": "台積電",
                   "date": "2025-01-01",
                   "ai_sentiment": "positive",
                   "predicted_direction": "bullish",
                   "actual_direction": "bullish",
                   "is_correct": True}
            for h in FUTURE_DAYS:
                row[f"future_return_{h}d"] = 2.0
            rows.append(row)
        df = pd.DataFrame(rows)
        metrics = {"binary_accuracy": 0.8, "per_stock": {}}

        with patch("tests.backtest_engine.results_path",
                   side_effect=lambda f: str(tmp_path / f)):
            export_results(df, metrics)

        assert (tmp_path / "prediction_results.csv").exists()
        assert (tmp_path / "prediction_metrics.json").exists()

    def test_stock_metrics_csv_created(self, tmp_path):
        """stock_metrics.csv should be created when per_stock is populated."""
        from tests.backtest_engine import export_results
        from tests.utils import FUTURE_DAYS
        rows = [{"symbol": "2330.TW", "name": "台積電",
                 "date": "2025-01-01", "ai_sentiment": "positive",
                 "predicted_direction": "bullish",
                 "actual_direction": "bullish", "is_correct": True,
                 **{f"future_return_{h}d": 2.0 for h in FUTURE_DAYS}}]
        df = pd.DataFrame(rows)
        metrics = {
            "binary_accuracy": 0.8,
            "per_stock": {
                "2330.TW": {
                    "count": 1,
                    "name":  "台積電",
                    "horizons": {
                        "3d": {"n": 1, "accuracy": 0.8,
                               "ci_lower": 0.2, "ci_upper": 0.99,
                               "win_rate": 0.8, "mean_return": 2.0}
                    },
                }
            },
        }
        with patch("tests.backtest_engine.results_path",
                   side_effect=lambda f: str(tmp_path / f)):
            export_results(df, metrics)
        assert (tmp_path / "stock_metrics.csv").exists()


# ════════════════════════════════════════════════════════════════
# 8. Visualiser smoke tests
# ════════════════════════════════════════════════════════════════

class TestVisualiser:

    def _sample_df(self) -> pd.DataFrame:
        from tests.utils import FUTURE_DAYS
        sents = ["positive", "slightly_positive", "neutral",
                 "slightly_negative", "negative"]
        rows = []
        for i in range(30):
            s = sents[i % 5]
            row = {
                "symbol":              "2330.TW",
                "date":                f"2025-01-{i % 28 + 1:02d}",
                "ai_sentiment":        s,
                "predicted_direction": "bullish",
                "actual_direction":    "bearish",
                "is_correct":          bool(i % 2),
            }
            for h in FUTURE_DAYS:
                row[f"future_return_{h}d"] = float(np.random.normal(0.5, 2.0))
            rows.append(row)
        return pd.DataFrame(rows)

    @pytest.fixture(autouse=True)
    def _redirect_outputs(self, tmp_path, monkeypatch):
        import tests.visualize_prediction as viz
        import tests.utils as utils_mod
        monkeypatch.setattr(viz, "results_path", lambda f: str(tmp_path / f))
        monkeypatch.setattr(utils_mod, "RESULTS_DIR", str(tmp_path))

    def test_confusion_matrix_smoke(self):
        from tests.visualize_prediction import plot_confusion_matrix
        df = self._sample_df()
        plot_confusion_matrix(df, ret_col="future_return_3d")

    def test_sentiment_distribution_smoke(self):
        from tests.visualize_prediction import plot_sentiment_distribution
        plot_sentiment_distribution(self._sample_df())

    def test_sentiment_vs_return_smoke(self):
        from tests.visualize_prediction import plot_sentiment_vs_return
        plot_sentiment_vs_return(self._sample_df(), ret_col="future_return_3d")

    def test_cumulative_return_smoke(self):
        from tests.visualize_prediction import plot_cumulative_return
        plot_cumulative_return(self._sample_df(), ret_col="future_return_3d")

    def test_accuracy_bar_smoke(self):
        from tests.visualize_prediction import plot_accuracy_bar
        plot_accuracy_bar(self._sample_df(), ret_col="future_return_3d")

    def test_return_distribution_smoke(self):
        from tests.visualize_prediction import plot_return_distribution
        plot_return_distribution(self._sample_df(), ret_col="future_return_3d")

    def test_accuracy_by_horizon_smoke(self):
        from tests.visualize_prediction import plot_accuracy_by_horizon
        plot_accuracy_by_horizon(self._sample_df())

    def test_f1_by_horizon_smoke(self):
        from tests.visualize_prediction import plot_f1_by_horizon
        plot_f1_by_horizon(self._sample_df())

    def test_winrate_by_horizon_smoke(self):
        from tests.visualize_prediction import plot_winrate_by_horizon
        plot_winrate_by_horizon(self._sample_df())


# ════════════════════════════════════════════════════════════════
# 9. get_stock_news_range (fetcher)
# ════════════════════════════════════════════════════════════════

class TestGetStockNewsRange:

    def _make_finmind_response(self, title: str, date_str: str) -> bytes:
        payload = {
            "status": 200,
            "data": [{
                "date":     f"{date_str} 09:00:00",
                "stock_id": "2330",
                "title":    title,
                "source":   "FinMind",
                "link":     f"https://example.com/{date_str}",
            }],
        }
        return json.dumps(payload).encode("utf-8")

    @patch("analysis.fetcher.urllib.request.urlopen")
    def test_iterates_day_by_day(self, mock_urlopen):
        from analysis.fetcher import get_stock_news_range

        call_count = 0
        def _side_effect(req, timeout=10):
            nonlocal call_count
            call_count += 1
            date_in_url = req.full_url.split("start_date=")[1][:10]
            class _Resp:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self_):
                    return self._make_finmind_response(
                        f"News on {date_in_url}", date_in_url
                    )
            return _Resp()

        mock_urlopen.side_effect = _side_effect

        articles = get_stock_news_range(
            "2330.TW",
            start_date="2025-01-01",
            end_date="2025-01-03",
            request_delay=0.0,
        )
        # 3 days → 3 requests
        assert call_count == 3
        assert len(articles) == 3

    @patch("analysis.fetcher.urllib.request.urlopen")
    def test_skips_dates_in_skip_set(self, mock_urlopen):
        from analysis.fetcher import get_stock_news_range

        queried_dates = []
        def _side_effect(req, timeout=10):
            date_in_url = req.full_url.split("start_date=")[1][:10]
            queried_dates.append(date_in_url)
            class _Resp:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self_):
                    return json.dumps({"status": 200, "data": []}).encode()
            return _Resp()

        mock_urlopen.side_effect = _side_effect

        get_stock_news_range(
            "2330.TW",
            start_date="2025-01-01",
            end_date="2025-01-05",
            request_delay=0.0,
            skip_dates={"2025-01-03", "2025-01-04"},
        )
        assert "2025-01-03" not in queried_dates
        assert "2025-01-04" not in queried_dates
        assert "2025-01-01" in queried_dates
        assert "2025-01-02" in queried_dates

    @patch("analysis.fetcher.urllib.request.urlopen")
    def test_deduplicates_by_url(self, mock_urlopen):
        from analysis.fetcher import get_stock_news_range

        same_url_payload = {
            "status": 200,
            "data": [{"date": "2025-01-01 09:00:00", "stock_id": "2330",
                      "title": "TSMC news", "source": "FinMind",
                      "link": "https://example.com/same-article"}],
        }

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(same_url_payload).encode()

        mock_urlopen.return_value = _Resp()

        articles = get_stock_news_range(
            "2330.TW",
            start_date="2025-01-01",
            end_date="2025-01-03",
            request_delay=0.0,
        )
        # Same URL across 3 days → only 1 unique article
        assert len(articles) == 1

    @patch("analysis.fetcher.urllib.request.urlopen")
    def test_progress_callback_called(self, mock_urlopen):
        from analysis.fetcher import get_stock_news_range

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"status": 200, "data": []}).encode()

        mock_urlopen.return_value = _Resp()

        calls = []
        def _cb(date_str, day_idx, total, collected):
            calls.append((date_str, day_idx, total, collected))

        get_stock_news_range(
            "2330.TW",
            start_date="2025-01-01",
            end_date="2025-01-05",
            request_delay=0.0,
            progress_callback=_cb,
        )
        assert len(calls) == 5   # one per day
        assert calls[0][2] == 5  # total_days = 5

    def test_start_after_end_raises(self):
        from analysis.fetcher import get_stock_news_range
        with pytest.raises(ValueError, match="start_date"):
            get_stock_news_range("2330.TW",
                                 start_date="2025-06-01",
                                 end_date="2025-01-01",
                                 request_delay=0.0)


# ════════════════════════════════════════════════════════════════
# 10. End-to-end pipeline (fully mocked)
# ════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:

    @patch("tests.backtest_engine._get_cached_history")
    def test_full_pipeline_all_horizons(self, mock_hist):
        from tests.backtest_engine import compute_metrics, export_results, run_backtest
        from tests.utils import FUTURE_DAYS

        df = _make_price_df("2025-01-01", periods=90, price=100.0)
        df["Close"] = [100.0 + i * 0.3 for i in range(90)]
        mock_hist.return_value = df

        dataset = _make_pre_analyzed(5)
        result  = run_backtest(dataset, primary_horizon=3)

        assert len(result) == 5
        for h in FUTURE_DAYS:
            assert f"future_return_{h}d" in result.columns

        metrics = compute_metrics(result, primary_horizon=3)
        for h in FUTURE_DAYS:
            assert f"{h}d" in metrics["horizon_metrics"]

        assert "per_stock" in metrics
        assert "2330.TW" in metrics["per_stock"]

    @patch("tests.backtest_engine._get_cached_history")
    def test_per_stock_ci_bounds_valid(self, mock_hist):
        from tests.backtest_engine import compute_metrics, run_backtest

        df = _make_price_df("2025-01-01", periods=90, price=100.0)
        df["Close"] = [100.0 + i * 0.5 for i in range(90)]
        mock_hist.return_value = df

        dataset = _make_pre_analyzed(20)
        result  = run_backtest(dataset, primary_horizon=3)
        metrics = compute_metrics(result, primary_horizon=3)

        for sym, sym_stats in metrics["per_stock"].items():
            for h_label, h_data in sym_stats["horizons"].items():
                lo = h_data.get("ci_lower", 0)
                hi = h_data.get("ci_upper", 1)
                assert 0.0 <= lo <= hi <= 1.0, (
                    f"{sym} {h_label}: CI [{lo}, {hi}] is invalid"
                )