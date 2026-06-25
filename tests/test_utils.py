import json

import pytest

from cafl.utils.utils import list_gemini_models


def test_list_gemini_models_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        list_gemini_models()


def test_list_gemini_models_filters_generate_content_models(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "models": [
                        {
                            "name": "models/gemini-3-flash-preview",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/text-embedding-004",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("cafl.utils.utils.urlopen", lambda url: FakeResponse())

    assert list_gemini_models() == ["gemini/gemini-3-flash-preview"]
