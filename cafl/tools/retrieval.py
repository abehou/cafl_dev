"""Local BM25 retrieval tools for JSONL corpora."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import shutil
import sqlite3
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cafl.memory import register_tool_summarizer
from cafl.utils.formatting import json_preview
from cafl.utils.shell import shell_arg, shell_args, split_shell_command
from cafl.utils.utils import parse_tool_output_json, safe_slug

INDEX_CONFIG_KEY = "index_config"
INDEX_COMPACT_FIELDS_KEY = "compact_fields"
INDEX_SHARDS_KEY = "shards"
MAX_COMPACT_FIELD_CHARS = 240
DEFAULT_MAX_SEARCH_LIMIT = 20
QUERY_TERM_LIMIT = 24
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
SUMMARY_TOP_RESULTS = 3
MAX_OPEN_SHARD_CONNECTIONS = 32


def corpus_files(corpus_path: Path | str) -> list[Path]:
    path = Path(corpus_path)
    if path.is_file():
        return [path] if path.suffix == ".jsonl" else []
    return sorted(path.rglob("*.jsonl"))


def build_bm25_index(
    corpus_path: Path | str,
    index_path: Path | str,
    *,
    config: dict | None = None,
    text_fields: tuple[str, ...] | list[str] | None = None,
    rebuild: bool = False,
) -> dict:
    corpus_path = Path(corpus_path)
    index_path = Path(index_path)
    index_config = normalize_index_config(
        config,
        text_fields=text_fields,
    )
    if index_path.exists() and not rebuild and index_matches_config(index_path, index_config):
        return {
            "index_path": str(index_path),
            "n_documents": count_documents(index_path),
            "n_shards": len(read_index_shards(index_path)),
            "rebuilt": False,
        }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()
    shard_root = shard_root_path(index_path)
    if shard_root.exists():
        shutil.rmtree(shard_root)

    files = corpus_files(corpus_path)
    with sqlite3.connect(index_path) as conn:
        initialize_index_schema(conn)
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            (INDEX_CONFIG_KEY, json.dumps(index_config, sort_keys=True)),
        )

        n_documents = 0
        compact_fields = set()
        shard_field = index_config.get("shard_field")
        shard_conns: OrderedDict[str, sqlite3.Connection] = OrderedDict()
        shard_paths: dict[str, str] = {}
        shard_counts: dict[str, int] = {}
        for file_path in tqdm(files, desc="Indexing corpus files", unit="file"):
            source = str(file_path.relative_to(corpus_path) if corpus_path.is_dir() else file_path.name)
            records = tqdm(
                iter_jsonl_records(file_path),
                desc=f"Indexing {source}",
                unit="doc",
                leave=False,
            )
            for line_no, record in records:
                n_documents += 1
                doc_id = n_documents - 1
                text = record_text(record, tuple(index_config["text_fields"]))
                metadata = compact_record_metadata(record)
                compact_fields.update(metadata)
                insert_index_record(conn, doc_id, source, line_no, record, metadata, text)
                if shard_field and shard_field in metadata:
                    shard_value = metadata[shard_field]
                    if shard_value in shard_conns:
                        shard_conn = shard_conns.pop(shard_value)
                        shard_conns[shard_value] = shard_conn
                    else:
                        shard_path = shard_index_path(index_path, shard_field, shard_value)
                        is_new_shard = shard_value not in shard_paths
                        if is_new_shard:
                            shard_path.parent.mkdir(parents=True, exist_ok=True)
                        shard_conn = sqlite3.connect(shard_path)
                        if is_new_shard:
                            initialize_index_schema(shard_conn)
                            shard_conn.execute(
                                "INSERT INTO index_meta (key, value) VALUES (?, ?)",
                                (INDEX_CONFIG_KEY, json.dumps(index_config, sort_keys=True)),
                            )
                            shard_paths[shard_value] = str(shard_path)
                            shard_counts[shard_value] = 0
                        shard_conns[shard_value] = shard_conn
                        if len(shard_conns) > MAX_OPEN_SHARD_CONNECTIONS:
                            _old_value, old_conn = shard_conns.popitem(last=False)
                            old_conn.commit()
                            old_conn.close()
                    insert_index_record(shard_conn, doc_id, source, line_no, record, metadata, text)
                    shard_counts[shard_value] += 1
        finalize_index(conn, compact_fields)
        for shard_conn in shard_conns.values():
            shard_conn.commit()
            shard_conn.close()
        for shard_path in shard_paths.values():
            with sqlite3.connect(shard_path) as shard_conn:
                finalize_index(shard_conn, compact_fields)
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            (
                INDEX_SHARDS_KEY,
                json.dumps(
                    {
                        "field": shard_field,
                        "paths": shard_paths,
                        "counts": shard_counts,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        conn.commit()

    clear_index_metadata_cache()
    return {"index_path": str(index_path), "n_documents": n_documents, "n_shards": len(shard_paths), "rebuilt": True}


def initialize_index_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE document_metadata (
            doc_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (doc_id, key)
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            doc_id UNINDEXED,
            text,
            tokenize='unicode61'
        )
        """
    )


def insert_index_record(
    conn: sqlite3.Connection,
    doc_id: int,
    source: str,
    line_no: int,
    record: dict,
    metadata: dict[str, str],
    text: str,
) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, source, line_no, record_json, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            source,
            line_no,
            json.dumps(record, ensure_ascii=False),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )
    conn.execute("INSERT INTO documents_fts (doc_id, text) VALUES (?, ?)", (doc_id, text))
    conn.executemany(
        "INSERT INTO document_metadata (doc_id, key, value) VALUES (?, ?, ?)",
        [(doc_id, key, value) for key, value in metadata.items() if value is not None],
    )


def finalize_index(conn: sqlite3.Connection, compact_fields: set[str]) -> None:
    conn.execute("CREATE INDEX idx_document_metadata_key_value ON document_metadata(key, value)")
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES (?, ?)",
        (INDEX_COMPACT_FIELDS_KEY, json.dumps(sorted(compact_fields), ensure_ascii=False)),
    )
    conn.commit()


def prepare_bm25_index(
    config: dict,
    data_dir: Path,
    *,
    rebuild: bool = False,
    skip: bool = False,
) -> Path | None:
    corpus_dir = config.get("corpus_dir")
    if skip or not corpus_dir:
        return None

    corpus_path = data_dir / corpus_dir
    if not corpus_path.exists():
        raise FileNotFoundError(f"Configured corpus_dir does not exist: {corpus_path}")

    bm25_config = dict(config.get("bm25", {}))
    index_path = data_dir / bm25_config.pop("index_path", "indexes/bm25.sqlite")
    stats = build_bm25_index(
        corpus_path,
        index_path,
        config=bm25_config,
        rebuild=rebuild,
    )
    action = "Built" if stats["rebuilt"] else "Using existing"
    shard_text = f", {stats.get('n_shards', 0)} shard(s)" if stats.get("n_shards") else ""
    print(f"{action} BM25 index: {stats['index_path']} ({stats['n_documents']} documents{shard_text})")
    return index_path


def search_bm25(
    index_path: Path | str,
    query: str,
    *,
    limit: int = 10,
    filters: dict[str, str] | None = None,
    include_record: bool = False,
) -> list[dict]:
    index_path = Path(index_path)
    index_path, filters = routed_search_index(index_path, filters or {})
    match_query = make_fts_query(query)
    if not match_query:
        return []
    compact_fields = read_compact_fields(index_path)

    where = ["documents_fts MATCH ?"]
    params: list[Any] = [match_query]
    for key, value in (filters or {}).items():
        if key not in compact_fields:
            raise ValueError(f"Unsupported filter field: {key}")
        where.append(
            """
            EXISTS (
                SELECT 1 FROM document_metadata
                WHERE document_metadata.doc_id = documents.doc_id
                AND document_metadata.key = ?
                AND document_metadata.value = ?
            )
            """
        )
        params.extend([key, value])
    params.append(limit)

    select_columns = [
        "documents.doc_id",
        "documents.source",
        "documents.line_no",
        "documents.metadata_json",
        "bm25(documents_fts) AS score",
        "snippet(documents_fts, 1, '', '', ' ... ', 32) AS snippet",
    ]
    if include_record:
        select_columns.insert(4, "documents.record_json")
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM documents_fts
        JOIN documents ON documents.doc_id = documents_fts.doc_id
        WHERE {' AND '.join(where)}
        ORDER BY score
        LIMIT ?
    """
    with sqlite3.connect(index_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    return [row_to_result(row) for row in rows]


def get_bm25_doc(index_path: Path | str, doc_id: int) -> dict:
    index_path = Path(index_path)
    with sqlite3.connect(index_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT doc_id, source, line_no, record_json, metadata_json
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()

    if row is None:
        raise ValueError(f"Unknown doc_id: {doc_id}")

    metadata = json.loads(row["metadata_json"])
    return {
        "doc_id": row["doc_id"],
        "source": row["source"],
        "line_no": row["line_no"],
        "metadata": metadata,
        "record": json.loads(row["record_json"]),
    }


def count_documents(index_path: Path | str) -> int:
    with sqlite3.connect(index_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def normalize_index_config(
    config: dict | None = None,
    *,
    text_fields: tuple[str, ...] | list[str] | None = None,
) -> dict:
    config = dict(config or {})
    if "bm25" in config:
        config = dict(config.get("bm25") or {})
    return {
        "text_fields": list(text_fields if text_fields is not None else config.get("text_fields") or []),
        "shard_field": config.get("shard_field"),
    }


def read_index_config(index_path: Path | str) -> dict:
    return dict(_read_index_config_cached(str(Path(index_path))))


@lru_cache(maxsize=128)
def _read_index_config_cached(index_path: str) -> dict:
    with sqlite3.connect(index_path) as conn:
        value = conn.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            (INDEX_CONFIG_KEY,),
        ).fetchone()[0]
    return json.loads(value)


def read_compact_fields(index_path: Path | str) -> list[str]:
    return list(_read_compact_fields_cached(str(Path(index_path))))


@lru_cache(maxsize=128)
def _read_compact_fields_cached(index_path: str) -> tuple[str, ...]:
    with sqlite3.connect(index_path) as conn:
        row = conn.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            (INDEX_COMPACT_FIELDS_KEY,),
        ).fetchone()
    if row is not None:
        return tuple(json.loads(row[0]))
    return ()


def read_index_shards(index_path: Path | str) -> dict:
    return dict(_read_index_shards_cached(str(Path(index_path))))


@lru_cache(maxsize=128)
def _read_index_shards_cached(index_path: str) -> dict:
    with sqlite3.connect(index_path) as conn:
        row = conn.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            (INDEX_SHARDS_KEY,),
        ).fetchone()
    return json.loads(row[0]) if row is not None else {}


def clear_index_metadata_cache() -> None:
    _read_index_config_cached.cache_clear()
    _read_compact_fields_cached.cache_clear()
    _read_index_shards_cached.cache_clear()


def index_matches_config(index_path: Path | str, index_config: dict) -> bool:
    with contextlib.suppress(Exception):
        if read_index_config(index_path) != index_config:
            return False
        shards = read_index_shards(index_path)
        return all(Path(path).exists() for path in shards.get("paths", {}).values())
    return False


def shard_root_path(index_path: Path | str) -> Path:
    index_path = Path(index_path)
    return index_path.with_suffix(index_path.suffix + ".shards")


def shard_index_path(index_path: Path | str, field: str, value: str) -> Path:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    slug = safe_slug(value, max_length=48)
    return shard_root_path(index_path) / field / f"{slug}-{digest}.sqlite"


def routed_search_index(index_path: Path, filters: dict[str, str]) -> tuple[Path, dict[str, str]]:
    shards = read_index_shards(index_path)
    shard_field = shards.get("field")
    if not shard_field or shard_field not in filters:
        return index_path, filters

    shard_path = shards.get("paths", {}).get(filters[shard_field])
    if not shard_path or not Path(shard_path).exists():
        return index_path, filters

    remaining_filters = dict(filters)
    remaining_filters.pop(shard_field, None)
    return Path(shard_path), remaining_filters


def iter_jsonl_records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def record_text(record: dict, fields: tuple[str, ...]) -> str:
    # If text_fields is configured, only those fields control search. This prevents
    # accidental matches from IDs, labels, or other short metadata fields.
    if fields:
        parts = [stringify_metadata(record.get(field)) for field in fields if record.get(field) is not None]
        return "\n".join(part for part in parts if part)

    # If text_fields is omitted, index the corpus record automatically. This keeps
    # env config small while still supporting arbitrary JSONL schemas.
    return "\n".join(auto_text_values(record))


def auto_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(auto_text_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(auto_text_values(item))
        return values
    return [str(value)]


def compact_record_metadata(record: dict) -> dict[str, str]:
    # Compact fields are safe to show in every result and cheap to filter by.
    # Long statute/body text still lives in BM25 snippets or include_record output.
    metadata = {}
    for field, value in record.items():
        text = stringify_metadata(value)
        if text is not None and len(text) <= MAX_COMPACT_FIELD_CHARS:
            metadata[field] = text
    return metadata


def stringify_metadata(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def make_fts_query(query: str) -> str:
    terms = re.findall(r"[\w§.-]+", query)
    kept_terms = []
    seen_terms = set()
    for term in terms:
        key = term.casefold()
        if key in QUERY_STOPWORDS or key in seen_terms:
            continue
        seen_terms.add(key)
        kept_terms.append(term)
        if len(kept_terms) >= QUERY_TERM_LIMIT:
            break
    return " OR ".join(f'"{term}"' for term in kept_terms)


def row_to_result(row: sqlite3.Row) -> dict:
    metadata = json.loads(row["metadata_json"])
    result = {
        "doc_id": row["doc_id"],
        "source": row["source"],
        "line_no": row["line_no"],
        "score": row["score"],
        "snippet": row["snippet"],
        "metadata": metadata,
    }
    result.update(metadata)
    if "record_json" in row.keys():
        result["record"] = json.loads(row["record_json"])
    return result


def bm25_tool_instruction(index_path: Path | str) -> str:
    config_text = ""
    with contextlib.suppress(Exception):
        config = read_index_config(index_path)
        compact_fields = read_compact_fields(index_path)
        shards = read_index_shards(index_path)
        text_fields = ", ".join(config["text_fields"]) if config["text_fields"] else "auto-detected from corpus records"
        shard_text = ""
        if shards.get("field"):
            shard_text = (
                f"Shard field: {shards['field']} ({len(shards.get('paths', {}))} shard indexes). "
                f"Use --filter {shards['field']}=VALUE whenever the task specifies that jurisdiction/category.\n"
            )
        config_text = (
            "\n"
            f"Searchable text fields: {text_fields}\n"
            f"Available compact fields: {', '.join(compact_fields)}\n"
            f"{shard_text}"
        )
    return f"""

## BM25 Corpus Search
Use the BM25 corpus search tool for local corpus retrieval.
{config_text}

Search command:
python -m cafl.tools.retrieval search --index {index_path} --query "your search terms" --limit 10

You may add metadata filters when useful, for example:
python -m cafl.tools.retrieval search --index {index_path} --query "notice to quit tenant" --filter state=missouri --limit 5

If a jurisdiction-like shard field is listed above, always include that filter when the task gives the value.

If a result looks relevant, inspect the full record by doc_id:
python -m cafl.tools.retrieval get-doc --index {index_path} --doc-id DOC_ID

Keep searches precise. Prefer more focused searches over broader result dumps. Keep --limit at {DEFAULT_MAX_SEARCH_LIMIT} or lower.

Truncated output does not automatically mean the search failed. If the visible snippet or a get-doc result is enough evidence, stop searching and answer.

Search returns compact snippets plus doc_id, source, line_no, and metadata. Use get-doc when you need the full record.
"""


@register_tool_summarizer
def summarize_bm25_tool_observation(action: dict, output: Any) -> dict | None:
    command = action.get("command")
    if not isinstance(command, str) or "cafl.tools.retrieval" not in command:
        return None

    parsed = _parse_retrieval_command(command)
    if parsed is None:
        return None

    output_data = parse_tool_output_json(output)
    if parsed["subcommand"] == "search":
        return _summarize_bm25_search(parsed, output_data)
    if parsed["subcommand"] == "get-doc":
        return _summarize_bm25_doc(parsed, output_data)
    return None


def _parse_retrieval_command(command: str) -> dict | None:
    parts = split_shell_command(command)
    if parts is None:
        return None

    with contextlib.suppress(ValueError):
        module_index = parts.index("cafl.tools.retrieval")
        if module_index < 1 or parts[module_index - 1] != "-m":
            return None
        subcommand = parts[module_index + 1] if len(parts) > module_index + 1 else ""
        return {
            "subcommand": subcommand,
            "query": shell_arg(parts, "--query"),
            "filters": shell_args(parts, "--filter"),
            "doc_id": shell_arg(parts, "--doc-id"),
        }
    return None


def _summarize_bm25_search(parsed: dict, output_data: Any) -> dict:
    query = parsed.get("query") or ""
    if not isinstance(output_data, list):
        return {
            "tool": "bm25.search",
            "query": query,
            "filters": parsed.get("filters", []),
            "summary": json_preview(output_data),
        }

    top_results = [
        {
            "doc_id": result.get("doc_id"),
            "citation": result.get("citation") or result.get("metadata", {}).get("citation"),
            "matched_terms": _matched_query_terms(query, result),
        }
        for result in output_data[:SUMMARY_TOP_RESULTS]
    ]
    relevance = _estimate_result_relevance(top_results)
    compact_results = _compact_top_results(top_results)
    return {
        "tool": "bm25.search",
        "query": query,
        "filters": parsed.get("filters", []),
        "top_results": compact_results,
        "relevance": relevance,
        "summary": _search_summary_text(query, len(output_data), compact_results, relevance),
    }


def _summarize_bm25_doc(parsed: dict, output_data: Any) -> dict:
    if not isinstance(output_data, dict):
        return {
            "tool": "bm25.get-doc",
            "doc_id": parsed.get("doc_id"),
            "summary": json_preview(output_data),
        }

    metadata = output_data.get("metadata") or {}
    record = output_data.get("record") or {}
    citation = metadata.get("citation") or record.get("citation")
    state = metadata.get("state") or record.get("state")
    preview = _record_text_preview(record)
    return {
        "tool": "bm25.get-doc",
        "doc_id": output_data.get("doc_id") or parsed.get("doc_id"),
        "summary": " ".join(part for part in [citation, state, preview] if part),
    }


def _matched_query_terms(query: str, result: dict) -> list[str]:
    terms = [term.casefold() for term in re.findall(r"[\w§.-]+", query) if term.casefold() not in QUERY_STOPWORDS]
    haystack = json.dumps(result, ensure_ascii=False).casefold()
    matched = []
    for term in terms:
        if term in haystack and term not in matched:
            matched.append(term)
    return matched


def _estimate_result_relevance(top_results: list[dict]) -> str:
    if not top_results:
        return "none"
    max_matches = max(len(result.get("matched_terms", [])) for result in top_results)
    if max_matches >= 3:
        return "high"
    if max_matches >= 1:
        return "medium"
    return "low"


def _compact_top_results(top_results: list[dict]) -> list[str]:
    results = []
    for result in top_results:
        label = str(result.get("doc_id"))
        citation = result.get("citation")
        if citation:
            label += f" {citation}"
        matched = result.get("matched_terms") or []
        if matched:
            label += f" matches={','.join(matched[:6])}"
        results.append(label)
    return results


def _search_summary_text(query: str, n_results: int, top_results: list[str], relevance: str) -> str:
    if n_results == 0:
        return f'No BM25 results for "{query}".'
    return f'BM25 search for "{query}" returned {n_results} result(s); relevance={relevance}; top={top_results}.'


def _record_text_preview(record: dict) -> str:
    for key in ("text", "body", "content", "snippet"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value[:300]
    return json_preview(record, max_chars=300)


def parse_filters(values: list[str] | None) -> dict[str, str]:
    filters = {}
    for value in values or []:
        key, sep, item = value.partition("=")
        if not sep:
            raise ValueError(f"Filter must be in key=value form: {value}")
        filters[key] = item
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query local BM25 indexes for JSONL corpora.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--corpus", required=True)
    build_parser.add_argument("--index", required=True)
    build_parser.add_argument("--config", help="Optional JSON config file. Uses its bm25 section when present.")
    build_parser.add_argument("--rebuild", action="store_true")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--index", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--filter", action="append", default=[])
    search_parser.add_argument("--include-record", action="store_true")
    search_parser.add_argument("--allow-large-limit", action="store_true")

    doc_parser = subparsers.add_parser("get-doc")
    doc_parser.add_argument("--index", required=True)
    doc_parser.add_argument("--doc-id", type=int, required=True)

    args = parser.parse_args()
    if args.command == "build":
        config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
        print(json.dumps(build_bm25_index(args.corpus, args.index, config=config, rebuild=args.rebuild), ensure_ascii=False))
    elif args.command == "search":
        if args.limit > DEFAULT_MAX_SEARCH_LIMIT and not args.allow_large_limit:
            parser.error(
                f"search --limit may not exceed {DEFAULT_MAX_SEARCH_LIMIT} unless --allow-large-limit is set. "
                "Prefer precise queries, metadata filters, or get-doc on a promising doc_id."
            )
        results = search_bm25(
            args.index,
            args.query,
            limit=args.limit,
            filters=parse_filters(args.filter),
            include_record=args.include_record,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif args.command == "get-doc":
        print(json.dumps(get_bm25_doc(args.index, args.doc_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
