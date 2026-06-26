from cafl.memory import load_environment_memory


def test_load_environment_memory_returns_empty_when_missing(tmp_path):
    assert load_environment_memory(tmp_path / "memory", max_chars=100) == ""


def test_load_environment_memory_reads_memory_md_with_cap(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("abcdef", encoding="utf-8")

    assert load_environment_memory(memory_dir, max_chars=3) == "abc"
