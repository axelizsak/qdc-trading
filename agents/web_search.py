"""Web search for the agents, through Linkup's API (paid from its free credit, settings.WEB_SEARCH_USD a search).

The model asks for a search and the code runs it (agents/llm.py). A web page is data, never an instruction: what
comes back goes to the model as quoted results and nothing else.
"""
import requests

from data import config as settings

URL = "https://api.linkup.so/v1/search"
TIMEOUT = 30  # seconds


def search(query: str, since: str | None = None) -> list[dict]:
    """The best results for a query, each {"title", "url", "date", "text"}: at most WEB_SEARCH_RESULTS of them,
    the text of each cut to WEB_SEARCH_CHARS. Only the sites of WEB_SEARCH_SITES are searched, and only pages
    published since `since` (YYYY-MM-DD) when it is given. The same page served under several addresses counts
    once. Raises when there is no key, no credit left or the service does not answer."""
    if not settings.LINKUP_API_KEY:
        raise RuntimeError("no LINKUP_API_KEY")
    body = {"q": " ".join(query.split())[:400], "depth": "standard", "outputType": "searchResults",
            "includeImages": False, "maxResults": 2 * settings.WEB_SEARCH_RESULTS,
            "includeDomains": settings.WEB_SEARCH_SITES}
    if since:
        body["fromDate"] = since
    reply = requests.post(URL, json=body, timeout=TIMEOUT,
                          headers={"Authorization": f"Bearer {settings.LINKUP_API_KEY}"})
    if reply.status_code == 429:
        raise RuntimeError("Linkup: too many requests, or no credit left")
    reply.raise_for_status()
    results, seen = [], set()
    for item in reply.json().get("results", []):
        text = " ".join(str(item.get("content") or "").split())
        same = (item.get("name"), text[:200])
        if item.get("type") != "text" or not text or same in seen:
            continue
        seen.add(same)
        results.append({"title": str(item.get("name") or "")[:160], "url": item.get("url", ""),
                        "date": item.get("publishedAt") or "", "text": text[:settings.WEB_SEARCH_CHARS]})
    return results[:settings.WEB_SEARCH_RESULTS]
