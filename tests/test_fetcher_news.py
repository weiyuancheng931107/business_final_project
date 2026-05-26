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
    def test_fetch_article_content_limits_by_language(self, mock_urlopen):
        class MockResponse:
            def __init__(self, text):
                self.text = text

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return self.text.encode("utf-8")

        # 1. 測試中文網頁 (限制 800 字)
        chinese_p = "這是一段很長很長的中文測試段落，用來測試中文網頁的字數限制是否正確運作。" * 50
        mock_urlopen.return_value = MockResponse(f"<html><body><article><p>{chinese_p}</p><p>{chinese_p}</p></article></body></html>")
        
        result_zh = fetcher.fetch_article_content("https://example.com/zh")
        self.assertEqual(len(result_zh), 800)
        self.assertTrue(any('\u4e00' <= char <= '\u9fff' for char in result_zh))

        # 2. 測試英文網頁 (限制 2500 字)
        english_p = "This is a very long English paragraph used to test if the character limit works correctly for non-CJK text." * 50
        mock_urlopen.return_value = MockResponse(f"<html><body><article><p>{english_p}</p><p>{english_p}</p></article></body></html>")
        
        result_en = fetcher.fetch_article_content("https://example.com/en")
        self.assertEqual(len(result_en), 2500)
        self.assertFalse(any('\u4e00' <= char <= '\u9fff' for char in result_en))


if __name__ == "__main__":
    unittest.main()
