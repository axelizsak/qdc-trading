"""The door to the model (agents/llm.py) and the news search behind it, with a fake network."""
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from agents import llm, web_search
from data import config as settings

FAST, SCARCE = "fast-model", "scarce-model"


def reply(status=200, **body):
    return SimpleNamespace(status_code=status, headers={"content-type": "application/json"}, json=lambda: body)


def answer(text="done", reason="STOP", calls=(), tokens=(100, 20)):
    parts = [{"functionCall": {"name": "web_search", "args": {"queries": list(calls)}, "id": "call_0"},
              "thoughtSignature": "sig"}] if calls else [{"text": text}]
    return reply(candidates=[{"content": {"role": "model", "parts": parts}, "finishReason": reason}],
                 usageMetadata={"promptTokenCount": tokens[0], "candidatesTokenCount": tokens[1]})


def error(status, name, message="", **detail):
    return reply(status, error={"status": name, "message": message, "details": [detail] if detail else []})


@pytest.fixture
def network(monkeypatch):
    """Replies to send back in order, and the requests received: (model, body)."""
    replies, sent, waits = [], [], []

    def post(url, json, timeout, headers):
        assert headers == {"x-goog-api-key": "key"}
        sent.append((url.split("/models/")[1].split(":")[0], json))
        return replies.pop(0)

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm, "_resting", {})
    monkeypatch.setattr(llm, "_next_start", {})
    monkeypatch.setattr(settings, "LLM_REASONING", SCARCE)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_FAST", FAST)
    return SimpleNamespace(replies=replies, sent=sent, waits=waits)


def ask(model=FAST, **extra):
    return llm.client().messages.create(model=model, system="Be brief.", max_tokens=500, **extra, messages=[
        {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Go on"}])


def test_no_key_no_client(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.client()


def test_a_request_is_put_in_the_models_words(network):
    network.replies.append(answer('{"a": 1}', tokens=(120, 30)))
    schema = {"type": "object", "properties": {"a": {"type": "number"}}}
    out = ask(output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}})
    model, body = network.sent[0]
    assert model == FAST and body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"] and body["contents"][2]["parts"] == [{"text": "Go on"}]
    assert body["generationConfig"] == {"maxOutputTokens": 500, "thinkingConfig": {"thinkingLevel": "medium"},
                                        "responseMimeType": "application/json", "responseJsonSchema": schema}
    assert "tools" not in body
    assert out.content[0].text == '{"a": 1}' and out.content[0].type == "text" and out.stop_reason == "end_turn"
    assert (out.usage.input_tokens, out.usage.output_tokens, out.usage.server_tool_use.web_search_requests) == (120, 30, 0)


@pytest.mark.parametrize("given, expected", [(answer(reason="MAX_TOKENS"), "max_tokens"), (answer(reason="SAFETY"), "refusal"),
                                             (reply(promptFeedback={"blockReason": "OTHER"}), "refusal")])
def test_a_cut_or_refused_answer_is_told_apart(network, given, expected):
    network.replies.append(given)
    assert ask().stop_reason == expected


def test_the_models_searches_are_run_by_the_code(network, monkeypatch):
    searched = []
    monkeypatch.setattr(web_search, "search", lambda query: searched.append(query) or [{"url": f"https://a.example/{query[:3]}"}])
    network.replies += [answer(calls=["ecb rate", " ", "fed rate", "boj rate"]), answer("Rates are on hold.")]
    out = ask(tools=[{"name": "web_search", "max_uses": 2}])
    assert searched == ["ecb rate", "fed rate"]  # the empty one is skipped, the last is over the limit
    first, second = network.sent[0][1], network.sent[1][1]
    assert first["tools"] == [llm.SEARCH_TOOL] and "at most 2 searches" in first["systemInstruction"]["parts"][0]["text"]
    assert second["contents"][3]["parts"][0]["thoughtSignature"] == "sig"  # the model's turn goes back as it came
    reply_part = second["contents"][4]["parts"][0]["functionResponse"]
    done = reply_part["response"]["searches"]
    assert reply_part["id"] == "call_0" and done[0] == {"query": "ecb rate", "results": [{"url": "https://a.example/ecb"}]}
    assert done[1]["query"] == "fed rate" and "no search left" in done[2]["error"]
    assert second["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}  # it now has to answer
    assert out.content[0].text == "Rates are on hold." and out.usage.server_tool_use.web_search_requests == 2
    assert out.search_urls == ["https://a.example/ecb", "https://a.example/fed"]  # what a source may be checked against
    assert out.usage.input_tokens == 200 and out.usage.requests == 2  # both requests are counted


def test_a_failed_search_does_not_stop_the_answer(network, monkeypatch):
    def broken(query):
        raise RuntimeError("no credit")
    monkeypatch.setattr(web_search, "search", broken)
    network.replies += [answer(calls=["ecb"]), answer("No fresh news.")]
    out = ask(tools=[{"name": "web_search", "max_uses": 3}])
    done = network.sent[1][1]["contents"][4]["parts"][0]["functionResponse"]["response"]["searches"]
    assert "the search failed" in done[0]["error"] and out.content[0].text == "No fresh news."
    assert out.search_urls == [] and out.usage.server_tool_use.web_search_requests == 0  # nothing found, nothing to cite


def test_busy_and_the_minutes_quota_are_waited_out(network):
    minute = error(429, "RESOURCE_EXHAUSTED", **{"@type": "RetryInfo", "retryDelay": "17s"})
    network.replies += [error(503, "UNAVAILABLE", "high demand"), minute, answer("ok")]
    assert ask().content[0].text == "ok" and len(network.sent) == 3
    assert llm.BUSY_WAIT in network.waits and 17.0 in network.waits


def test_the_fast_model_answers_when_the_scarce_one_cannot(network, capsys):
    day = error(429, "RESOURCE_EXHAUSTED", quotaId="GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    network.replies += [day, answer("from the fast model"), answer("again")]
    out = ask(SCARCE)
    assert [m for m, _ in network.sent] == [SCARCE, FAST] and out.model == FAST and out.content[0].text == "from the fast model"
    assert "answers instead" in capsys.readouterr().out
    assert ask(SCARCE).model == FAST and len(network.sent) == 3  # the scarce model is left alone for a while


def test_each_model_is_paced_on_its_own(network, monkeypatch):
    clock = iter([100.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
    for model in (FAST, FAST, "other-model", FAST):
        llm._wait_for_a_turn(model)
    step = 60 / settings.LLM_REQUESTS_PER_MINUTE
    assert network.waits == pytest.approx([0, step, 0, 2 * step])  # the other model does not wait for the first


def test_another_model_than_the_scarce_one_gets_every_try(network):
    network.replies += [error(503, "UNAVAILABLE"), error(503, "UNAVAILABLE"), answer("ok")]
    assert ask("search-model").model == "search-model" and len(network.sent) == 3


def test_when_nothing_answers_the_caller_is_told(network):
    network.replies += [error(503, "UNAVAILABLE")] * llm.TRIES
    with pytest.raises(llm.Unavailable, match="503"):
        ask()
    network.replies.append(error(400, "INVALID_ARGUMENT", "bad schema"))
    with pytest.raises(RuntimeError, match="bad schema"):
        ask()
    assert len(network.sent) == llm.TRIES + 1  # a request the service rejects is not sent again


FEED = """<rss><channel>
<item><title>ECB holds rates as inflation cools</title><link>https://www.cnbc.com/ecb-holds</link><pubDate>{now}</pubDate>
  <description>&lt;p&gt;The ECB kept its  deposit rate at 2%.&lt;/p&gt;</description></item>
<item><title>ECB rates: what happened</title><link>https://www.cnbc.com/old</link><pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
  <description>A story on ECB rates from long ago.</description></item>
<item><title>Football results</title><link>https://www.cnbc.com/football</link><pubDate>{now}</pubDate><description>Not markets.</description></item>
</channel></rss>"""
GOOGLE = """<rss><channel>
<item><title>ECB rate decision due Thursday - Reuters</title><link>https://news.google.com/rss/articles/abc</link>
  <pubDate>{now}</pubDate><source url="https://www.reuters.com">Reuters</source></item>
<item><title>ECB will cut rates, says a blog - Some Blog</title><link>https://news.google.com/rss/articles/def</link>
  <pubDate>{now}</pubDate><source url="https://someblog.example">Some Blog</source></item>
</channel></rss>"""


def test_a_search_reads_free_feeds_and_keeps_trusted_recent_articles(monkeypatch):
    now = format_datetime(datetime.now(timezone.utc))
    asked = []

    def get(url, headers, timeout):
        asked.append(url)
        if "broken" in url:
            raise web_search.requests.ConnectionError("down")
        xml = GOOGLE if url.startswith("https://news.google.com") else FEED
        return SimpleNamespace(content=xml.format(now=now).encode(), raise_for_status=lambda: None)

    monkeypatch.setattr(web_search.requests, "get", get)
    monkeypatch.setattr(settings, "NEWS_FEEDS", ["https://feed.example/markets", "https://feed.example/broken"])
    web_search.feed_articles.cache_clear()
    found = web_search.search("  ECB   rates decision ")
    web_search.feed_articles.cache_clear()
    assert [r["url"] for r in found] == ["https://www.cnbc.com/ecb-holds", "https://news.google.com/rss/articles/abc"]
    assert found[0]["text"] == "The ECB kept its deposit rate at 2%." and found[0]["date"] == datetime.now(timezone.utc).date().isoformat()
    assert found[1]["text"] == found[1]["title"] == "ECB rate decision due Thursday - Reuters"  # a headline has no summary
    assert set(found[0]) == {"title", "url", "date", "text"}
    google = next(url for url in asked if url.startswith("https://news.google.com"))
    assert "q=ECB+rates+decision+when:14d" in google  # the old story, the football and the unknown blog are left out
    assert len(settings.WEB_SEARCH_SITES) == len(set(settings.WEB_SEARCH_SITES)) and len(settings.NEWS_FEEDS) > 0
