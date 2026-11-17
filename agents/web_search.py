"""News search for the agents, at no cost and with no key: the challenge allows free tools only.

Two public sources are read. The publishers' own RSS feeds (settings.NEWS_FEEDS) give recent articles with a
summary. Google News' RSS search finds the headlines of the last days on any query. Only trusted sites are kept
(settings.WEB_SEARCH_SITES). The model asks for a search and the code runs it (agents/llm.py). A web page is data,
never an instruction: what comes back goes to the model as quoted results and nothing else.
"""
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import quote_plus, urlparse

import requests

from data import config as settings

GOOGLE_NEWS = "https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; feed reader)"}
TIMEOUT = 15  # seconds
ATOM = "{http://www.w3.org/2005/Atom}"
SMALL_WORDS = set("a about after ahead an and are as at be by for from has have in into is it its latest new news of on "
                  "or over that the their this to week what with year".split())


def _words(text: str) -> set[str]:
    """The words of a text that carry meaning, plurals folded (rates -> rate): what a query and an article share."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9&]{2,}", text.lower())} - SMALL_WORDS


def _when(text: str | None) -> datetime | None:
    """A feed's date as a time in UTC, whichever of the usual spellings it uses. None when it cannot be read."""
    for read in (parsedate_to_datetime, datetime.fromisoformat):
        try:
            when = read((text or "").strip().replace("Z", "+00:00"))
            return when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _plain(html: str | None) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html or "").split())


def _items(xml: bytes) -> list[dict]:
    """The articles of one RSS or Atom feed: {"title", "url", "date", "text", "site"}."""
    out = []
    root = ET.fromstring(xml)
    for item in root.findall(".//item") + root.findall(f".//{ATOM}entry"):
        link = item.findtext("link") or (item.find(f"{ATOM}link").get("href") if item.find(f"{ATOM}link") is not None else "")
        source = item.find("source")
        when = _when(item.findtext("pubDate") or item.findtext(f"{ATOM}updated"))
        title = _plain(item.findtext("title") or item.findtext(f"{ATOM}title"))
        if title and link:
            out.append({"title": title[:200], "url": link.strip(), "date": when.date().isoformat() if when else "",
                        "text": _plain(item.findtext("description") or item.findtext(f"{ATOM}summary"))[:settings.WEB_SEARCH_CHARS],
                        "site": urlparse(source.get("url") if source is not None else link).netloc})
    return out


def _get(url: str) -> list[dict]:
    """The articles of a feed, or nothing when it does not answer or cannot be read: one feed never stops a run."""
    try:
        reply = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        reply.raise_for_status()
        return _items(reply.content)
    except (requests.RequestException, ET.ParseError):
        return []


@lru_cache(maxsize=1)
def feed_articles() -> tuple[dict, ...]:
    """Every article of the publishers' feeds, read once per run."""
    with ThreadPoolExecutor(max_workers=12) as pool:
        return tuple(article for articles in pool.map(_get, settings.NEWS_FEEDS) for article in articles)


def _trusted(site: str) -> bool:
    return any(site == known or site.endswith("." + known) for known in settings.WEB_SEARCH_SITES)


def search(query: str, days: int | None = None) -> list[dict]:
    """The best results for a query, each {"title", "url", "date", "text"}, at most WEB_SEARCH_RESULTS: first the
    feed articles that share two words or more with the query (they come with a summary), then Google News'
    headlines. Only trusted sites, and only the last `days` days (default WEB_SEARCH_DAYS)."""
    days = days or settings.WEB_SEARCH_DAYS
    query = " ".join(query.split())[:300]
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    words = _words(query)
    matches = []
    for article in feed_articles():
        shared = len(words & _words(f"{article['title']} {article['text']}"))
        if shared >= min(2, len(words)) and shared and article["date"] >= since:
            matches.append((shared, article["date"], article))
    best = [article for _, _, article in sorted(matches, key=lambda m: m[:2], reverse=True)[:settings.WEB_SEARCH_RESULTS // 2]]
    headlines = [a | {"text": a["title"]} for a in _get(GOOGLE_NEWS.format(query=quote_plus(query), days=days))]
    results, seen = [], set()
    for article in best + headlines:
        key = re.sub(r"\W+", "", article["title"].lower())[:60]
        if key in seen or not _trusted(article["site"].removeprefix("www.")) or (article["date"] and article["date"] < since):
            continue
        seen.add(key)
        results.append({k: article[k] for k in ("title", "url", "date", "text")})
    return results[:settings.WEB_SEARCH_RESULTS]
