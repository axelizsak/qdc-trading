"""News search for the agents, at no cost and with no key: the challenge allows free tools only.

Three public sources are read, and only trusted sites are kept (settings.WEB_SEARCH_SITES):

- the publishers' own RSS feeds (settings.NEWS_FEEDS): recent articles with a summary;
- Bing News' RSS search: for any query, articles with their own link and a short summary. It is asked twice, once
  among the main press only (settings.WEB_SEARCH_PRESS) and once openly;
- Google News' RSS search, only when the two above find little: it gives a headline without a summary, and a link
  that goes through Google instead of the publisher's own.

The model asks for a search and the code runs it (agents/llm.py). A web page is data, never an instruction: what
comes back goes to the model as quoted results and nothing else.
"""
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from itertools import zip_longest
from urllib.parse import parse_qs, quote_plus, urlparse

import requests

from data import config as settings

BING_NEWS = "https://www.bing.com/news/search?q={query}&format=rss&setmkt=en-US"
GOOGLE_NEWS = "https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
FEW = 3  # results under which Google News' headlines fill in
MATCH = 0.4  # share of a query's words, rare words counting for more, that a feed article must hold to answer it
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


def _own_link(link: str) -> str:
    """The article's own address. Bing's feed gives a link of its own that carries it in its `url` part."""
    inside = parse_qs(urlparse(link).query).get("url") if urlparse(link).netloc.endswith("bing.com") else None
    return inside[0] if inside else link.strip()


def _items(xml: bytes) -> list[dict]:
    """The articles of one RSS or Atom feed: {"title", "url", "date", "text", "site"}."""
    out = []
    root = ET.fromstring(xml)
    for item in root.findall(".//item") + root.findall(f".//{ATOM}entry"):
        link = _own_link(item.findtext("link") or (item.find(f"{ATOM}link").get("href") if item.find(f"{ATOM}link") is not None else ""))
        source = item.find("source")
        when = _when(item.findtext("pubDate") or item.findtext(f"{ATOM}updated"))
        title = _plain(item.findtext("title") or item.findtext(f"{ATOM}title"))
        if title and link:
            out.append({"title": title[:200], "url": link, "date": when.date().isoformat() if when else "",
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
def feeds() -> tuple[tuple[dict, ...], Counter]:
    """Every article of the publishers' feeds with its words, and the number of articles each word is in. Read once
    per run."""
    with ThreadPoolExecutor(max_workers=12) as pool:
        articles = tuple(article | {"words": _words(f"{article['title']} {article['text']}")}
                         for articles in pool.map(_get, settings.NEWS_FEEDS) for article in articles)
    return articles, Counter(word for article in articles for word in article["words"])


def _trusted(site: str) -> bool:
    return any(site == known or site.endswith("." + known) for known in settings.WEB_SEARCH_SITES)


def _kept(articles: list[dict], since: str) -> list[dict]:
    """The articles from trusted sites and of the last days, each once, as the model gets them."""
    results, seen = [], set()
    for article in articles:
        key = re.sub(r"\W+", "", article["title"].lower())[:60]
        if key in seen or not _trusted(article["site"].removeprefix("www.")) or (article["date"] and article["date"] < since):
            continue
        seen.add(key)
        results.append({k: article[k] for k in ("title", "url", "date", "text")})
    return results


def search(query: str, days: int | None = None) -> list[dict]:
    """The best results for a query, each {"title", "url", "date", "text"}, at most WEB_SEARCH_RESULTS. Three lists
    take turns, so that no one fills the answer alone: Bing News among the main press, the feed articles that hold
    enough of the query's words (MATCH), and Bing News openly. When they find little, Google News' headlines fill
    in. Only trusted sites, and only the last `days` days (default WEB_SEARCH_DAYS)."""
    days = days or settings.WEB_SEARCH_DAYS
    query = " ".join(query.split())[:300]
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    articles, count = feeds()
    # A word few articles use (Bridgewater) says more about a topic than one many use (market): it counts for more.
    weight = {word: math.log((len(articles) + 1) / (count[word] + 1)) for word in _words(query)}
    matches = []
    for article in articles:
        held = sum(weight[word] for word in weight.keys() & article["words"]) / (sum(weight.values()) or 1)
        if held >= MATCH and article["date"] >= since:
            matches.append((held, article["date"], article))
    matching = [article for _, _, article in sorted(matches, key=lambda m: m[:2], reverse=True)[:settings.WEB_SEARCH_RESULTS // 2]]
    press = " OR ".join(f"site:{site}" for site in settings.WEB_SEARCH_PRESS)
    lists = [_get(BING_NEWS.format(query=quote_plus(f"{query} ({press})"))), matching, _get(BING_NEWS.format(query=quote_plus(query)))]
    found = [article for turn in zip_longest(*lists) for article in turn if article]
    results = _kept(found, since)
    if len(results) < FEW:
        headlines = [a | {"text": a["title"]} for a in _get(GOOGLE_NEWS.format(query=quote_plus(query), days=days))]
        results = _kept(found + headlines, since)
    return results[:settings.WEB_SEARCH_RESULTS]
