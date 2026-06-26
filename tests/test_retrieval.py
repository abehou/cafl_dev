import json
import subprocess
import sys
import pytest

from cafl.memory import format_working_memory, record_tool_observation
from cafl.tools.retrieval import bm25_tool_instruction, build_bm25_index, get_bm25_doc, make_fts_query, search_bm25


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_bm25_index_and_search_jsonl_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(
        corpus_dir / "statutes.jsonl",
        [
            {
                "citation": "MO Rev Stat § 441.060",
                "state": "missouri",
                "path": "Missouri|Landlord Tenant",
                "text": "A tenancy may be terminated with one month's notice to vacate.",
            },
            {
                "citation": "AK Stat § 34.03.220",
                "state": "alaska",
                "path": "Alaska|Residential Landlord Tenant",
                "text": "A material noncompliance by the tenant may support notice to quit.",
            },
        ],
    )
    index_path = tmp_path / "indexes" / "bm25.sqlite"

    stats = build_bm25_index(corpus_dir, index_path)
    results = search_bm25(index_path, "tenant notice", limit=2)

    assert stats["n_documents"] == 2
    assert index_path.exists()
    assert [result["citation"] for result in results] == [
        "AK Stat § 34.03.220",
        "MO Rev Stat § 441.060",
    ]
    assert all("snippet" in result for result in results)
    assert all("record" not in result for result in results)


def test_search_bm25_can_include_full_record_when_requested(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"citation": "A", "text": "tenant notice rent"}])
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(corpus_dir, index_path)

    result = search_bm25(index_path, "tenant notice", include_record=True)[0]

    assert result["record"] == {"citation": "A", "text": "tenant notice rent"}


def test_get_bm25_doc_returns_full_record_by_doc_id(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(
        corpus_dir / "docs.jsonl",
        [
            {"citation": "A", "state": "missouri", "text": "tenant notice rent"},
            {"citation": "B", "state": "alaska", "text": "material breach"},
        ],
    )
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(corpus_dir, index_path)

    doc = get_bm25_doc(index_path, 1)

    assert doc == {
        "doc_id": 1,
        "source": "docs.jsonl",
        "line_no": 2,
        "metadata": {"citation": "B", "state": "alaska", "text": "material breach"},
        "record": {"citation": "B", "state": "alaska", "text": "material breach"},
    }


def test_get_bm25_doc_rejects_unknown_doc_id(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"citation": "A", "text": "tenant notice rent"}])
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(corpus_dir, index_path)

    with pytest.raises(ValueError, match="Unknown doc_id: 9"):
        get_bm25_doc(index_path, 9)


def test_bm25_tool_observation_updates_working_memory():
    memory = {}
    action = {
        "command": (
            "python -m cafl.tools.retrieval search --index /tmp/bm25.sqlite "
            '--query "tenant material breach" --filter state=alaska --limit 5'
        )
    }
    output = {
        "returncode": 0,
        "output": json.dumps(
            [
                {
                    "doc_id": 7,
                    "source": "docs.jsonl",
                    "line_no": 8,
                    "snippet": "A material breach by the tenant may support termination.",
                    "metadata": {"citation": "AK Stat § 34.03.220", "state": "alaska"},
                    "citation": "AK Stat § 34.03.220",
                    "state": "alaska",
                }
            ]
        ),
    }

    record_tool_observation(memory, action, output)
    formatted = format_working_memory(memory)

    assert memory["tool_observations"][0]["tool"] == "bm25.search"
    assert memory["tool_observations"][0]["relevance"] == "high"
    assert "AK Stat § 34.03.220" in formatted
    assert "If the memory already contains enough evidence" in formatted


def test_make_fts_query_removes_duplicate_stopword_noise():
    query = make_fts_query("What are the tenant tenant notice rules in the state?")

    assert query == '"tenant" OR "notice" OR "rules" OR "state"'


def test_bm25_index_uses_configured_text_fields_and_auto_metadata(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(
        corpus_dir / "docs.jsonl",
        [
            {
                "case_name": "Tenant v. Landlord",
                "jurisdiction": "missouri",
                "body": "The tenant must receive notice before eviction.",
                "unused": "not searchable",
            },
            {
                "case_name": "Other Case",
                "jurisdiction": "alaska",
                "body": "A material breach can matter.",
                "unused": "tenant notice",
            },
        ],
    )
    index_path = tmp_path / "bm25.sqlite"

    build_bm25_index(
        corpus_dir,
        index_path,
        config={
            "text_fields": ["body"],
        },
    )
    results = search_bm25(index_path, "tenant notice", filters={"jurisdiction": "missouri"})
    unused_results = search_bm25(index_path, "tenant notice", filters={"jurisdiction": "alaska"})

    assert len(results) == 1
    assert results[0]["case_name"] == "Tenant v. Landlord"
    assert results[0]["jurisdiction"] == "missouri"
    assert results[0]["metadata"] == {
        "body": "The tenant must receive notice before eviction.",
        "case_name": "Tenant v. Landlord",
        "jurisdiction": "missouri",
        "unused": "not searchable",
    }
    assert unused_results == []


def test_bm25_shard_field_routes_filtered_search_and_preserves_global_doc_ids(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(
        corpus_dir / "docs.jsonl",
        [
            {"citation": "MO", "jurisdiction": "missouri", "body": "tenant notice rules"},
            {"citation": "AK", "jurisdiction": "alaska", "body": "tenant notice rules"},
        ],
    )
    index_path = tmp_path / "bm25.sqlite"

    stats = build_bm25_index(
        corpus_dir,
        index_path,
        config={"text_fields": ["body"], "shard_field": "jurisdiction"},
    )
    results = search_bm25(index_path, "tenant notice", filters={"jurisdiction": "alaska"})
    doc = get_bm25_doc(index_path, results[0]["doc_id"])

    assert stats["n_shards"] == 2
    assert results[0]["doc_id"] == 1
    assert results[0]["citation"] == "AK"
    assert doc["record"]["citation"] == "AK"


def test_bm25_tool_instruction_mentions_configured_shard_field(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"citation": "A", "jurisdiction": "missouri", "body": "tenant notice"}])
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(
        corpus_dir,
        index_path,
        config={"text_fields": ["body"], "shard_field": "jurisdiction"},
    )

    instruction = bm25_tool_instruction(index_path)

    assert "Shard field: jurisdiction" in instruction
    assert "--filter jurisdiction=VALUE" in instruction


def test_bm25_auto_indexes_record_fields_when_text_fields_are_omitted(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"citation": "A", "statute_body": "tenant notice rules"}])
    index_path = tmp_path / "bm25.sqlite"

    build_bm25_index(corpus_dir, index_path, config={})
    result = search_bm25(index_path, "tenant notice")[0]

    assert result["citation"] == "A"


def test_bm25_index_accepts_whole_environment_config(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"case_name": "A", "jurisdiction": "missouri", "body": "tenant notice"}])
    index_path = tmp_path / "bm25.sqlite"

    build_bm25_index(
        corpus_dir,
        index_path,
        config={
            "task_file": "questions.jsonl",
            "bm25": {
                "text_fields": ["body"],
            },
        },
    )
    result = search_bm25(index_path, "tenant", filters={"jurisdiction": "missouri"})[0]

    assert result["jurisdiction"] == "missouri"


def test_bm25_rebuilds_when_index_config_changes(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"title": "Alpha", "body": "tenant notice"}])
    index_path = tmp_path / "bm25.sqlite"

    first = build_bm25_index(corpus_dir, index_path, config={"text_fields": ["title"]})
    changed = build_bm25_index(corpus_dir, index_path, config={"text_fields": ["body"]})

    assert first["rebuilt"] is True
    assert changed["rebuilt"] is True
    assert search_bm25(index_path, "tenant")


def test_bm25_tool_instruction_lists_auto_filter_fields(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"case_name": "A", "jurisdiction": "missouri", "body": "tenant notice"}])
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(
        corpus_dir,
        index_path,
        config={
            "text_fields": ["body"],
        },
    )

    instruction = bm25_tool_instruction(index_path)

    assert "Available compact fields: body, case_name, jurisdiction" in instruction
    assert "get-doc" in instruction
    assert "Use BM25 before grepping" not in instruction
    assert "--allow-large-limit" not in instruction
    assert "Keep --limit at 20 or lower" in instruction
    assert "Truncated output does not automatically mean the search failed" in instruction


def test_retrieval_cli_rejects_large_search_limit_without_override(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"citation": "A", "text": "tenant notice"}])
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(corpus_dir, index_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cafl.tools.retrieval",
            "search",
            "--index",
            str(index_path),
            "--query",
            "tenant notice",
            "--limit",
            "50",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--allow-large-limit" in result.stderr


def test_retrieval_cli_allows_large_search_limit_with_override(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"citation": "A", "text": "tenant notice"}])
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(corpus_dir, index_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cafl.tools.retrieval",
            "search",
            "--index",
            str(index_path),
            "--query",
            "tenant notice",
            "--limit",
            "50",
            "--allow-large-limit",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)[0]["citation"] == "A"


def test_search_bm25_supports_metadata_filters(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(
        corpus_dir / "docs.jsonl",
        [
            {"citation": "A", "state": "missouri", "text": "tenant notice rent"},
            {"citation": "B", "state": "alaska", "text": "tenant notice rent"},
        ],
    )
    index_path = tmp_path / "bm25.sqlite"
    build_bm25_index(corpus_dir, index_path)

    results = search_bm25(index_path, "tenant notice", filters={"state": "alaska"})

    assert len(results) == 1
    assert results[0]["citation"] == "B"


def test_build_bm25_index_reuses_existing_index_unless_rebuild(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_jsonl(corpus_dir / "docs.jsonl", [{"citation": "A", "text": "tenant notice"}])
    index_path = tmp_path / "bm25.sqlite"

    first = build_bm25_index(corpus_dir, index_path)
    write_jsonl(
        corpus_dir / "docs.jsonl",
        [
            {"citation": "A", "text": "tenant notice"},
            {"citation": "B", "text": "material breach"},
        ],
    )
    reused = build_bm25_index(corpus_dir, index_path)
    rebuilt = build_bm25_index(corpus_dir, index_path, rebuild=True)

    assert first["rebuilt"] is True
    assert first["n_documents"] == 1
    assert reused["rebuilt"] is False
    assert reused["n_documents"] == 1
    assert rebuilt["rebuilt"] is True
    assert rebuilt["n_documents"] == 2
