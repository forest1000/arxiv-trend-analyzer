from datetime import datetime
from app.fetcher import search_iter




def test_search_query_build(monkeypatch):
    class Dummy:
        def __init__(self):
            self._called = False


        def results(self):
            self._called = True
            
            class R:
                def get_short_id(self):
                    return "a1"
                title = "t"
                summary = "s"
                published = datetime(2025,1,1)
                updated = None
                authors = []
                categories = []
                pdf_url = None
            yield R()


    # arxiv.Search を置き換え
    import app.fetcher as f
    monkeypatch.setattr(f.arxiv, "Search", lambda **kwargs: Dummy())


    items = list(search_iter("test", start=datetime(2024,1,1), end=datetime(2024,12,31), sleep_sec=0))
    assert len(items) == 1
    assert items[0].id == "a1"