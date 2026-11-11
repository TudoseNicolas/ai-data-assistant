import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ai_agent as agent  # noqa: E402


class FakeLLM:
    """Returns prepared replies in order, so the agent can be tested without an API key."""

    name = "Fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


@pytest.fixture
def df():
    return pd.DataFrame({"City": ["A", "B", "A", "C"], "Revenue": [10.0, 5.0, 20.0, 1.0]})


def test_extract_code_removes_fences_and_known_imports():
    reply = "```python\nimport pandas as pd\nresult = df.head()\n```"
    assert agent.extract_code(reply) == "result = df.head()"


@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "result = open('secrets.txt').read()",
    "result = df.__class__",
    "df.to_csv('out.csv')\nresult = 1",
    "result = pd.read_csv('other.csv')",
])
def test_unsafe_code_is_rejected(code, df):
    with pytest.raises(agent.UnsafeCodeError):
        agent.run_code(code, df)


def test_column_names_containing_os_are_allowed(df):
    data = df.rename(columns={"Revenue": "cost"})
    result, _ = agent.run_code("result = data_cost = df['cost'].sum()", data)
    assert result == 36.0


def test_run_code_returns_result_and_keeps_data_unchanged(df):
    before = df.copy()
    result, fig = agent.run_code("df['Revenue'] = 0\nresult = df['Revenue'].sum()", df)
    assert result == 0 and fig is None
    pd.testing.assert_frame_equal(df, before)


def test_ask_answers_with_explanation(df):
    llm = FakeLLM([
        "result = df.groupby('City')['Revenue'].sum().nlargest(2)",
        "City A has the highest revenue with 30.",
    ])
    answer = agent.ask(df, "Top 2 cities by revenue", llm)
    assert answer.error is None
    assert answer.attempts == 1
    assert list(answer.result.index) == ["A", "B"]
    assert answer.explanation.startswith("City A")


def test_ask_corrects_itself_after_an_error(df):
    llm = FakeLLM([
        "result = df['Sales'].sum()",  # wrong column -> KeyError
        "result = df['Revenue'].sum()",
        "Total revenue is 36.",
    ])
    answer = agent.ask(df, "Total revenue?", llm)
    assert answer.error is None
    assert answer.attempts == 2
    assert answer.result == 36.0
    assert "KeyError" in llm.prompts[1]  # the error was sent back to the model


def test_ask_gives_up_after_max_attempts(df):
    llm = FakeLLM(["result = df['Nope'].sum()"] * 3)
    answer = agent.ask(df, "?", llm, max_attempts=3)
    assert answer.error and answer.attempts == 3


def test_ask_never_retries_unsafe_code(df):
    llm = FakeLLM(["import os\nresult = os.listdir()"])
    answer = agent.ask(df, "list files", llm)
    assert "not allowed" in answer.error and answer.attempts == 1


class BusyLLM:
    name = "Busy"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        raise agent.ModelError("The AI model is very busy right now.")


def test_service_errors_are_not_treated_as_code_errors(df):
    llm = BusyLLM()
    answer = agent.ask(df, "Total revenue?", llm)
    assert "busy" in answer.error
    assert llm.calls == 1  # no pointless "fix your code" retries


class Busy(Exception):
    code = 503


class Missing(Exception):
    code = 404


class FakeModel:
    def __init__(self, name):
        self.name = f"models/{name}"
        self.supported_actions = ["generateContent"]


class FakeModels:
    """Simulates the Gemini API: `behaviour` maps model -> list of outcomes per call."""

    def __init__(self, behaviour, available):
        self.behaviour = {k: list(v) for k, v in behaviour.items()}
        self.available = available
        self.calls = []

    def list(self):
        return [FakeModel(n) for n in self.available]

    def generate_content(self, model, contents):
        self.calls.append(model)
        outcome = self.behaviour.get(model, [Missing("404 NOT_FOUND")]).pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("R", (), {"text": outcome})()


def make_client(behaviour, available=(), model="main-flash"):
    client = agent.GeminiClient.__new__(agent.GeminiClient)
    client.model, client.retry_delays, client._fallbacks = model, (0, 0), None
    client._client = type("C", (), {"models": FakeModels(behaviour, list(available))})()
    return client


def test_gemini_client_retries_when_busy_then_succeeds():
    client = make_client({"main-flash": [Busy("503"), Busy("503"), "result = 1"]})
    assert client.generate("hi") == "result = 1"
    assert client._client.models.calls == ["main-flash"] * 3


def test_gemini_client_switches_to_another_model_when_busy():
    client = make_client(
        {"main-flash": [Busy("503")] * 3, "gemini-9-flash": ["result = 2"]},
        available=["gemini-9-flash", "text-embedding-1", "gemini-9-flash-image", "gemini-9-pro"],
    )
    assert client.generate("hi") == "result = 2"
    assert client.model == "gemini-9-flash"  # keeps using the model that worked


def test_gemini_client_skips_models_that_no_longer_exist():
    client = make_client({"main-flash": [Missing("404 NOT_FOUND")], "gemini-9-flash": ["ok"]},
                         available=["gemini-9-flash"])
    assert client.generate("hi") == "ok"


def test_gemini_client_gives_a_clear_message_when_everything_is_busy():
    client = make_client({"main-flash": [Busy("503")] * 3}, available=[])
    with pytest.raises(agent.ModelError, match="busy"):
        client.generate("hi")


def test_gemini_client_reports_other_errors_immediately():
    class BadKey(Exception):
        code = 400

    client = make_client({"main-flash": [BadKey("API key not valid")]}, available=["gemini-9-flash"])
    with pytest.raises(agent.ModelError, match="API key not valid"):
        client.generate("hi")


def test_prompt_contains_only_a_sample_of_rows():
    big = pd.DataFrame({"x": range(1000)})
    llm = FakeLLM(["result = df['x'].sum()", "ok"])
    agent.ask(big, "sum", llm)
    assert "999" not in llm.prompts[0]


def test_suggestions_use_the_columns(df):
    questions = agent.suggest_questions(df)
    assert "Top 5 City by total Revenue" in questions


# local model (Ollama)

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


@pytest.fixture
def fake_ollama():
    """A tiny local HTTP server that answers like Ollama does."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            self._send(200, {"models": [{"name": "qwen2.5-coder:7b"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body["model"] != "qwen2.5-coder:7b":
                self._send(404, {"error": "model not found"})
            else:
                self._send(200, {"response": "```python\nresult = df['Revenue'].sum()\n```"})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ollama_client_answers_through_the_agent(fake_ollama, df):
    llm = agent.OllamaClient(model="qwen2.5-coder:7b", host=fake_ollama)
    assert llm.installed_models() == ["qwen2.5-coder:7b"]
    answer = agent.ask(df, "Total revenue?", llm)
    assert answer.error is None and answer.result == 36.0
    assert answer.model == "qwen2.5-coder:7b"


def test_ollama_client_explains_a_missing_model(fake_ollama, df):
    llm = agent.OllamaClient(model="other-model", host=fake_ollama)
    answer = agent.ask(df, "Total revenue?", llm)
    assert "ollama pull other-model" in answer.error


def test_ollama_client_explains_when_ollama_is_not_running(df):
    llm = agent.OllamaClient(host="http://127.0.0.1:9")
    assert llm.installed_models() is None
    answer = agent.ask(df, "Total revenue?", llm)
    assert "Can’t reach Ollama" in answer.error


def test_gemini_client_explains_quota_limits():
    class Quota(Exception):
        code = 429

    client = make_client({"main-flash": [Quota("429 RESOURCE_EXHAUSTED quota exceeded")] * 3}, available=[])
    with pytest.raises(agent.ModelError, match="usage limit"):
        client.generate("hi")
