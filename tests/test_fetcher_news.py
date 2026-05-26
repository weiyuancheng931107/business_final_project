import unittest
from unittest.mock import patch

from analysis import fetcher


class NewsFetcherTests(unittest.TestCase):
    def test_normalizes_yfinance_content_news_shape(self):
        normalizer = getattr(fetcher, "_normalize_yfinance_news_item", None)
        self.assertIsNotNone(normalizer, "yfinance news normalizer should exist")

        item = {
            "content": {
                "title": "台積電法說會釋出展望",
                "provider": {"displayName": "Yahoo Finance"},
                "pubDate": "2026-05-25T03:04:05Z",
                "canonicalUrl": {"url": "https://example.com/tsmc-news"},
            }
        }

        self.assertEqual(
            normalizer(item),
            {
                "title": "台積電法說會釋出展望",
                "source": "Yahoo Finance",
                "date": "2026-05-25",
                "url": "https://example.com/tsmc-news",
            },
        )

    @patch("analysis.fetcher.urllib.request.urlopen")
    @patch("analysis.fetcher._fetch_finmind_news", create=True)
    @patch("analysis.fetcher._fetch_yfinance_news", create=True)
    def test_get_stock_news_combines_yfinance_and_finmind_without_duplicates(
        self, mock_yfinance_news, mock_finmind_news, mock_urlopen
    ):
        class EmptyFinMindResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"status": 200, "data": []}'

        mock_urlopen.return_value = EmptyFinMindResponse()
        mock_yfinance_news.return_value = [
            {
                "title": "台積電法說會釋出展望",
                "source": "Yahoo Finance",
                "date": "2026-05-25",
                "url": "https://example.com/tsmc-news",
            }
        ]
        mock_finmind_news.return_value = {
            "news": [
                {
                    "title": "台積電法說會釋出展望",
                    "source": "FinMind",
                    "date": "2026-05-25",
                    "url": "https://example.com/tsmc-news",
                },
                {
                    "title": "台股盤中電子權值股走強",
                    "source": "FinMind",
                    "date": "2026-05-24",
                    "url": "https://example.com/taiwan-market",
                },
            ],
            "next_start_date": "2026-05-17",
            "has_more": True,
        }

        result = fetcher.get_stock_news("2330.TW")

        mock_yfinance_news.assert_called_once_with("2330.TW", limit=fetcher.NEWS_LIMIT)
        mock_finmind_news.assert_called_once_with("2330.TW", None, limit=fetcher.NEWS_LIMIT)
        self.assertEqual(
            [item["title"] for item in result["news"]],
            ["台積電法說會釋出展望", "台股盤中電子權值股走強"],
        )
        self.assertEqual(result["next_start_date"], "2026-05-17")
        self.assertTrue(result["has_more"])

    @patch("analysis.fetcher.urllib.request.urlopen")
    def test_finmind_news_uses_single_day_request_required_by_api(self, mock_urlopen):
        class FinMindResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"status": 200, "data": ['
                    b'{"date": "2026-05-25 09:00:00", "stock_id": "2330", '
                    b'"title": "TSMC news", "source": "FinMind", '
                    b'"link": "https://example.com/finmind"}'
                    b"]}"
                )

        mock_urlopen.return_value = FinMindResponse()

        result = fetcher._fetch_finmind_news("2330.TW", "2026-05-25", limit=1)
        requested_url = mock_urlopen.call_args.args[0].full_url

        self.assertIn("dataset=TaiwanStockNews", requested_url)
        self.assertIn("data_id=2330", requested_url)
        self.assertIn("start_date=2026-05-25", requested_url)
        self.assertNotIn("end_date=", requested_url)
        self.assertEqual(result["news"][0]["title"], "TSMC news")

    @patch("analysis.fetcher.urllib.request.urlopen")
    def test_fetch_article_content_without_limits(self, mock_urlopen):
        class MockResponse:
            def __init__(self, text):
                self.text = text

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return self.text.encode("utf-8")

        # 1. 測試中文網頁 (不限字數，使用 \\ 區隔段落)
        p1 = "這是一段非常長而且大於二十五個字元的中文測試段落一，用來避免被過濾。"
        p2 = "這是一段非常長而且大於二十五個字元的中文測試段落二，用來避免被過濾。"
        mock_urlopen.return_value = MockResponse(f"<html><body><article><p>{p1}</p><p>{p2}</p></article></body></html>")
        
        result_zh = fetcher.fetch_article_content("https://example.com/zh")
        self.assertEqual(result_zh, f"{p1} \\\\ {p2}")
        self.assertTrue(any('\u4e00' <= char <= '\u9fff' for char in result_zh))

        # 2. 測試英文網頁
        p_en1 = "This is a very long English paragraph one that is longer than twenty five characters."
        p_en2 = "This is a very long English paragraph two that is longer than twenty five characters."
        mock_urlopen.return_value = MockResponse(f"<html><body><article><p>{p_en1}</p><p>{p_en2}</p></article></body></html>")
        
        result_en = fetcher.fetch_article_content("https://example.com/en")
        self.assertEqual(result_en, f"{p_en1} \\\\ {p_en2}")
        self.assertFalse(any('\u4e00' <= char <= '\u9fff' for char in result_en))


if __name__ == "__main__":
    unittest.main()
