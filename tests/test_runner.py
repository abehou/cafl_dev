from pathlib import Path

from envs import runner
from cafl.tools import retrieval
from cafl.utils import config_utils


def test_resolve_environment_uses_env_root_config_and_data_dir(monkeypatch, tmp_path):
    env_root = tmp_path / "housing_qa"
    data_dir = env_root / "data"
    data_dir.mkdir(parents=True)
    (env_root / "config.json").write_text("{}", encoding="utf-8")

    resolved_data_dir, config_path = config_utils.resolve_environment("housing_qa", tmp_path)

    assert resolved_data_dir == data_dir.resolve()
    assert config_path == (env_root / "config.json").resolve()


def test_resolve_environment_accepts_data_dir_path(tmp_path):
    env_root = tmp_path / "housing_qa"
    data_dir = env_root / "data"
    data_dir.mkdir(parents=True)
    (env_root / "config.json").write_text("{}", encoding="utf-8")

    resolved_data_dir, config_path = config_utils.resolve_environment(str(data_dir), tmp_path)

    assert resolved_data_dir == data_dir.resolve()
    assert config_path == (env_root / "config.json").resolve()


def test_prepare_run_dir_defaults_to_data_runs(tmp_path):
    data_dir = tmp_path / "data"

    run_dir = runner.prepare_run_dir(None, "housing_qa", data_dir=data_dir)

    assert run_dir.parent == data_dir / "runs"
    assert run_dir.name.endswith("-housing-qa-batch")


def test_resolve_memory_dir_defaults_to_env_memory(tmp_path):
    env_root = tmp_path / "housing_qa"
    data_dir = env_root / "data"
    data_dir.mkdir(parents=True)
    config_path = env_root / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    memory_dir = config_utils.resolve_memory_dir({}, config_path)

    assert memory_dir == env_root / "memory"


def test_resolve_memory_dir_uses_configured_relative_path(tmp_path):
    env_root = tmp_path / "housing_qa"
    config_path = env_root / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")

    memory_dir = config_utils.resolve_memory_dir({"memory_dir": "notes"}, config_path)

    assert memory_dir == env_root / "notes"


def test_validate_config_requires_core_fields(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    try:
        config_utils.validate_env_config({"task_file": "questions.jsonl"}, data_dir)
    except ValueError as error:
        message = str(error)
        assert "corpus_dir" in message
        assert "task_field" in message
        assert "ground_truth_field" in message
        assert "output_schema" in message
    else:
        raise AssertionError("Expected ValueError")


def test_validate_config_checks_task_file_and_task_fields(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "questions.jsonl").write_text(
        '{"question": "Q?", "answer": "True"}\n',
        encoding="utf-8",
    )
    (data_dir / "corpus").mkdir()
    config = {
        "task_file": "questions.jsonl",
        "corpus_dir": "corpus",
        "task_field": "missing_question",
        "ground_truth_field": "answer",
        "output_schema": {"answer": "a string"},
    }

    try:
        config_utils.validate_env_config(config, data_dir)
    except ValueError as error:
        assert "missing_question" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_validate_config_accepts_valid_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "questions.jsonl").write_text(
        '{"question": "Q?", "answer": "True"}\n',
        encoding="utf-8",
    )
    (data_dir / "corpus").mkdir()
    config = {
        "task_file": "questions.jsonl",
        "corpus_dir": "corpus",
        "task_field": "question",
        "ground_truth_field": "answer",
        "output_schema": {"answer": "a string"},
    }

    assert config_utils.validate_env_config(config, data_dir) is None


def test_prepare_bm25_index_builds_configured_corpus(tmp_path):
    data_dir = tmp_path / "data"
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "docs.jsonl").write_text(
        '{"citation": "A", "state": "missouri", "text": "tenant notice"}\n',
        encoding="utf-8",
    )

    index_path = retrieval.prepare_bm25_index({"corpus_dir": "corpus"}, data_dir)

    assert index_path == data_dir / "indexes" / "bm25.sqlite"
    assert index_path.exists()


def test_prepare_bm25_index_uses_bm25_config_section(tmp_path, monkeypatch):
    calls = []
    data_dir = tmp_path / "data"
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True)

    def fake_build(corpus_path, index_path, **kwargs):
        calls.append((corpus_path, index_path, kwargs))
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("fake", encoding="utf-8")
        return {"index_path": str(index_path), "n_documents": 3, "rebuilt": True}

    monkeypatch.setattr(retrieval, "build_bm25_index", fake_build)

    index_path = retrieval.prepare_bm25_index(
        {
            "corpus_dir": "corpus",
            "bm25": {
                "index_path": "custom/search.sqlite",
                "text_fields": ["body"],
            },
        },
        data_dir,
    )

    assert index_path == data_dir / "custom" / "search.sqlite"
    assert calls == [
        (
            corpus_dir,
            data_dir / "custom" / "search.sqlite",
            {
                "config": {
                    "text_fields": ["body"],
                },
                "rebuild": False,
            },
        )
    ]


def test_prepare_bm25_index_raises_for_missing_configured_corpus(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    try:
        retrieval.prepare_bm25_index({"corpus_dir": "missing"}, data_dir)
    except FileNotFoundError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_prepare_bm25_index_reports_existing_index(tmp_path, capsys):
    data_dir = tmp_path / "data"
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "docs.jsonl").write_text(
        '{"citation": "A", "state": "missouri", "text": "tenant notice"}\n',
        encoding="utf-8",
    )

    retrieval.prepare_bm25_index({"corpus_dir": "corpus"}, data_dir)
    retrieval.prepare_bm25_index({"corpus_dir": "corpus"}, data_dir)

    captured = capsys.readouterr()
    assert "Built BM25 index" in captured.out
    assert "Using existing BM25 index" in captured.out


def test_template_vars_direct_agent_to_bm25(tmp_path):
    index_path = tmp_path / "indexes" / "bm25.sqlite"

    template_vars = config_utils.template_vars_for_env(
        {"corpus_dir": "corpus", "agent_additional_system_prompt": "Extra instructions."},
        tmp_path,
        bm25_index_path=index_path,
    )

    assert template_vars["task_environment_instructions"] == "Extra instructions."
    assert template_vars["corpus_dir"] == str(tmp_path / "corpus")
    assert "BM25 corpus search" in template_vars["bm25_instructions"]
    assert f"--index {index_path}" in template_vars["bm25_instructions"]
    assert "python -m cafl.tools.retrieval search" in template_vars["bm25_instructions"]
