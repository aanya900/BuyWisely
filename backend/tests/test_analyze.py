import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest
import requests
from fastapi.testclient import TestClient


PAYLOAD = {
    "product_data": {
        "title": "Example Earbuds",
        "price": "2499 INR",
        "battery_life": "30 hours including case",
        "water_resistance": "Not provided",
    },
    "question": "Is this suitable for gym use?",
}


@pytest.fixture
def api_module(monkeypatch):
    # Replace rag before loading main.py so tests do not
    # initialize Chroma or download an embedding model.
    fake_rag = ModuleType("rag")
    fake_rag.retrieve_context = lambda question: [
        "Water resistance is not provided."
    ]

    main_path = Path(__file__).resolve().parents[1] / "main.py"

    spec = importlib.util.spec_from_file_location(
        "buywisely_main_under_test",
        main_path,
    )

    module = importlib.util.module_from_spec(spec)

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "rag", fake_rag)
        spec.loader.exec_module(module)

    # Give this application a separate requests reference.
    # This avoids modifying requests globally.
    module.requests = SimpleNamespace(
        post=None,
        exceptions=requests.exceptions,
    )

    return module


@pytest.fixture
def client(api_module):
    with TestClient(api_module.app) as test_client:
        yield test_client


def model_response(body=None, http_error=False, invalid_json=False):
    class FakeResponse:
        def raise_for_status(self):
            if http_error:
                raise requests.exceptions.HTTPError(
                    "Simulated model HTTP error"
                )

        def json(self):
            if invalid_json:
                raise ValueError("Simulated invalid JSON")

            return body

    return FakeResponse()


def test_analyze_success(client, api_module, monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout

        return model_response({
            "response": (
                "Gym suitability cannot be confirmed because "
                "water resistance is not provided."
            )
        })

    monkeypatch.setattr(api_module.requests, "post", fake_post)

    response = client.post("/api/v1/analyze", json=PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "Gym suitability cannot be confirmed because "
        "water resistance is not provided."
    )
    assert data["sources"] == [
        "Water resistance is not provided."
    ]

    prompt = captured["payload"]["prompt"]

    assert PAYLOAD["question"] in prompt
    assert "Example Earbuds" in prompt
    assert "Water resistance is not provided." in prompt
    assert captured["payload"]["max_tokens"] == 180
    assert captured["url"] == api_module.LLM_SERVER


def test_analyze_empty_context(client, api_module, monkeypatch):
    monkeypatch.setattr(
        api_module,
        "retrieve_context",
        lambda question: [],
    )

    captured = {}

    def fake_post(url, json, timeout):
        captured["prompt"] = json["prompt"]
        return model_response({
            "response": "Water resistance is not provided."
        })

    monkeypatch.setattr(api_module.requests, "post", fake_post)

    response = client.post("/api/v1/analyze", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert (
        "No additional information was retrieved."
        in captured["prompt"]
    )


def test_analyze_retrieval_failure(client, api_module, monkeypatch):
    def broken_retrieval(question):
        raise RuntimeError("Simulated retrieval failure")

    def unexpected_model_call(*args, **kwargs):
        pytest.fail("Model should not be called after retrieval fails.")

    monkeypatch.setattr(
        api_module,
        "retrieve_context",
        broken_retrieval,
    )
    monkeypatch.setattr(
        api_module.requests,
        "post",
        unexpected_model_call,
    )

    response = client.post("/api/v1/analyze", json=PAYLOAD)

    assert response.status_code == 500
    assert "retrieval failed" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "exception_class,expected_status",
    [
        (requests.exceptions.ConnectionError, 503),
        (requests.exceptions.Timeout, 504),
    ],
)
def test_analyze_connection_errors(
    client,
    api_module,
    monkeypatch,
    exception_class,
    expected_status,
):
    def broken_post(*args, **kwargs):
        raise exception_class("Simulated connection problem")

    monkeypatch.setattr(api_module.requests, "post", broken_post)

    response = client.post("/api/v1/analyze", json=PAYLOAD)

    assert response.status_code == expected_status
    assert "detail" in response.json()


def test_analyze_model_http_error(client, api_module, monkeypatch):
    monkeypatch.setattr(
        api_module.requests,
        "post",
        lambda *args, **kwargs: model_response(http_error=True),
    )

    response = client.post("/api/v1/analyze", json=PAYLOAD)

    assert response.status_code == 502


def test_analyze_invalid_json(client, api_module, monkeypatch):
    monkeypatch.setattr(
        api_module.requests,
        "post",
        lambda *args, **kwargs: model_response(invalid_json=True),
    )

    response = client.post("/api/v1/analyze", json=PAYLOAD)

    assert response.status_code == 502
    assert "valid JSON" in response.json()["detail"]


def test_analyze_empty_answer(client, api_module, monkeypatch):
    monkeypatch.setattr(
        api_module.requests,
        "post",
        lambda *args, **kwargs: model_response({
            "response": "   "
        }),
    )

    response = client.post("/api/v1/analyze", json=PAYLOAD)

    assert response.status_code == 502


def test_analyze_blank_question(client):
    payload = {
        **PAYLOAD,
        "question": "",
    }

    response = client.post("/api/v1/analyze", json=payload)

    assert response.status_code == 422