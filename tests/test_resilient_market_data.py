import json
import urllib.parse
import unittest
from unittest.mock import patch

import pandas as pd

from analysis import fetcher
from app import app


class ResilientMarketDataTests(unittest.TestCase):
    @patch("analysis.fetcher.urllib.request.urlopen")
    @patch("analysis.fetcher.get_price_history")
    @patch("analysis.fetcher.get_tw_stock_name", return_value="台積電")
    @patch("analysis.fetcher.yf.Ticker")
    def test_stock_info_falls_back_when_yfinance_info_fails(
        self, mock_ticker, mock_stock_name, mock_price_history, mock_urlopen
    ):
        class BrokenTicker:
            @property
            def info(self):
                raise Exception("ssl failed")

        class FinMindResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def finmind_response(request, timeout=10):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
            dataset = query["dataset"][0]
            if dataset == "TaiwanStockPER":
                return FinMindResponse({
                    "status": 200,
                    "data": [
                        {"date": "2026-05-22", "stock_id": "2330", "PER": 30.32, "PBR": 9.93}
                    ],
                })
            if dataset == "TaiwanStockFinancialStatements":
                return FinMindResponse({
                    "status": 200,
                    "data": [
                        {"date": "2025-06-30", "stock_id": "2330", "type": "IncomeAfterTaxes", "value": 100.0},
                        {"date": "2025-09-30", "stock_id": "2330", "type": "IncomeAfterTaxes", "value": 200.0},
                        {"date": "2025-12-31", "stock_id": "2330", "type": "IncomeAfterTaxes", "value": 300.0},
                        {"date": "2026-03-31", "stock_id": "2330", "type": "IncomeAfterTaxes", "value": 400.0},
                        {"date": "2026-03-31", "stock_id": "2330", "type": "EPS", "value": 22.08},
                    ],
                })
            if dataset == "TaiwanStockBalanceSheet":
                return FinMindResponse({
                    "status": 200,
                    "data": [
                        {
                            "date": "2026-03-31",
                            "stock_id": "2330",
                            "type": "EquityAttributableToOwnersOfParent",
                            "value": 2000.0,
                        }
                    ],
                })
            raise AssertionError(f"Unexpected dataset: {dataset}")

        mock_ticker.return_value = BrokenTicker()
        mock_price_history.return_value = pd.DataFrame(
            {"Close": [888.0]},
            index=pd.to_datetime(["2026-05-25"]),
        )
        mock_urlopen.side_effect = finmind_response

        result = fetcher.get_stock_info("2330.TW")

        self.assertEqual(result["symbol"], "2330.TW")
        self.assertEqual(result["name"], "台積電")
        self.assertEqual(result["current_price"], 888.0)
        self.assertEqual(result["pe_ratio"], 30.32)
        self.assertEqual(result["eps"], 22.08)
        self.assertEqual(result["roe"], 0.5)

    @patch("analysis.fetcher.urllib.request.urlopen")
    @patch("analysis.fetcher.yf.Ticker")
    def test_price_history_falls_back_to_finmind_when_yfinance_is_empty(
        self, mock_ticker, mock_urlopen
    ):
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        class FinMindResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "status": 200,
                    "data": [
                        {
                            "date": "2026-05-25",
                            "open": 880,
                            "max": 895,
                            "min": 875,
                            "close": 888,
                            "Trading_Volume": 12345,
                        }
                    ],
                }).encode("utf-8")

        mock_urlopen.return_value = FinMindResponse()

        history = fetcher.get_price_history("2330.TW", period="1y")

        self.assertFalse(history.empty)
        self.assertEqual(float(history.iloc[-1]["Close"]), 888.0)
        self.assertIn("start_date=", mock_urlopen.call_args.args[0].full_url)
        self.assertIn("end_date=", mock_urlopen.call_args.args[0].full_url)

    @patch("analysis.fetcher.get_price_history")
    @patch("analysis.fetcher.yf.Ticker")
    def test_historical_pe_uses_safe_eps_fallback_when_yfinance_fundamentals_fail(
        self, mock_ticker, mock_price_history
    ):
        class BrokenTicker:
            @property
            def earnings_dates(self):
                raise Exception("ssl failed")

            @property
            def info(self):
                raise Exception("ssl failed")

        mock_ticker.return_value = BrokenTicker()
        mock_price_history.return_value = pd.DataFrame(
            {"Close": [888.0]},
            index=pd.to_datetime(["2026-05-25"]),
        )

        pe_df = fetcher.get_historical_pe("2330.TW", period="3y")

        self.assertFalse(pe_df.empty)
        self.assertEqual(float(pe_df.iloc[-1]["EPS"]), 1.0)
        self.assertEqual(float(pe_df.iloc[-1]["PE"]), 888.0)

    @patch("app.get_price_history", return_value=pd.DataFrame())
    def test_price_chart_endpoint_returns_placeholder_instead_of_404(self, mock_history):
        client = app.test_client()

        response = client.get("/api/stock/2330.TW/price_chart?period=1y")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["image"].endswith("_price.png"))

    @patch("app.get_stock_info", side_effect=Exception("unexpected provider failure"))
    def test_stock_info_endpoint_returns_degraded_payload_instead_of_500(self, mock_info):
        client = app.test_client()

        response = client.get("/api/stock/2330.TW/info")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["symbol"], "2330.TW")
        self.assertIsNone(response.get_json()["current_price"])

    @patch("app.get_price_history", side_effect=Exception("unexpected provider failure"))
    def test_price_chart_endpoint_returns_placeholder_when_fetch_raises(self, mock_history):
        client = app.test_client()

        response = client.get("/api/stock/2330.TW/price_chart?period=1y")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["image"].endswith("_price.png"))

    @patch("app.get_historical_pe", side_effect=Exception("unexpected provider failure"))
    def test_pe_river_endpoint_returns_placeholder_when_fetch_raises(self, mock_pe):
        client = app.test_client()

        response = client.get("/api/stock/2330.TW/pe_river?period=3y")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["image"].endswith("_pe_river.png"))


if __name__ == "__main__":
    unittest.main()
