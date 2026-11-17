"""The door to the model (agents/llm.py) and the web search behind it, with a fake network."""
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
    monkeypatch.setattr(web_search, "search", lambda query, since=None: searched.append((query, since)) or [{"url": f"https://a.example/{query[:3]}"}])
    network.replies += [answer(calls=["ecb rate", " ", "fed rate", "boj rate"]), answer("Rates are on hold.")]
    out = ask(tools=[{"name": "web_search", "max_uses": 2, "since": "2026-11-23"}])
    assert searched == [("ecb rate", "2026-11-23"), ("fed rate", "2026-11-23")]  # the empty one is skipped, the last is over the limit
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
    def broken(query, since=None):
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


def test_a_search_returns_a_few_clean_results(monkeypatch):
    sent = {}
    page = {"type": "text", "name": "ECB holds", "url": "https://a.example/ecb", "content": "The  ECB\nheld rates. " * 200,
            "publishedAt": "2026-12-04"}
    results = [page, {**page, "url": "https://b.example/ecb"}, {"type": "image", "name": "chart", "url": "u"},
               {"type": "text", "name": "Other", "url": "https://c.example", "content": "Other news."}]

    def post(url, json, timeout, headers):
        sent.update(url=url, body=json, headers=headers)
        return SimpleNamespace(status_code=200, json=lambda: {"results": results}, raise_for_status=lambda: None)

    monkeypatch.setattr(web_search.requests, "post", post)
    with pytest.raises(RuntimeError, match="LINKUP_API_KEY"):
        web_search.search("ecb")
    monkeypatch.setattr(settings, "LINKUP_API_KEY", "key")
    found = web_search.search("  ecb   rate decision ", "2026-11-23")
    assert sent["headers"] == {"Authorization": "Bearer key"} and sent["body"]["q"] == "ecb rate decision"
    assert sent["body"]["includeDomains"] == settings.WEB_SEARCH_SITES and sent["body"]["outputType"] == "searchResults"
    assert sent["body"]["fromDate"] == "2026-11-23"
    assert [r["url"] for r in found] == ["https://a.example/ecb", "https://c.example"]  # the copy and the image are out
    assert len(found[0]["text"]) == settings.WEB_SEARCH_CHARS and "  " not in found[0]["text"] and found[0]["date"] == "2026-12-04"
    web_search.search("ecb")
    assert "fromDate" not in sent["body"]
    assert len(settings.WEB_SEARCH_SITES) == len(set(settings.WEB_SEARCH_SITES)) <= 100
    monkeypatch.setattr(web_search.requests, "post", lambda *a, **k: SimpleNamespace(status_code=429))
    with pytest.raises(RuntimeError, match="no credit left"):
        web_search.search("ecb")
