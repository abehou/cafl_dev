# CAFL: Coding Agent For Law

CAFL is a lightweight framework for turning legal datasets into reproducible agent workflows. It is built for researchers and practitioners who need to deploy language model agents on real legal tasks. Typical workflows involve retrieving sources, reasoning over jurisdiction-specific rules, producing structured answers, and leaving an auditable trace of what they did.

The project is motivated by the observation that **legal datasets are heterogeneous but share similar core workflows of retrieving, reasoning, and validating answers**. Instead of having to rebuild similar pipelines for every dataset, we want to leave the flexibility to coding agents to figuring out the dataset specificity and reuse a set of robust and efficient framework.

## Motivating Example
A typical workflow might be:
```
- We have a legal dataset of multiple task items
- We build a pipeline to ingest data and make LLMs to solve the tasks, which involves retrieving some relevant legal texts and calling LLMs to reason over the tasks. We spend much time cleaning and normalizing the data so the retrieval and reasoning work. 
- We optimize LLM's prompts for the task
- We measure the task performance. 
```
We repeat building this workflow for different legal datasets, because each dataset is very heterogeneous. While the datasets may focus on different jurisdictions, statutes, or legal systems, the workflow might be similar variations of the abovementioned one.

Example: suppose we have `housing_qa`, a Question & Answering dataset about housing laws from different states. We 

Instead, CAFL builds:
```
- We show the legal dataset of multiple tasks to 
```




The main principles are:

- **Dataset-first configuration.** A task environment should describe its files, fields, output schema, retrieval corpus, and domain instructions without requiring benchmark-specific runner code.
- **Local evidence before answers.** Agents can inspect files and search local corpora, with BM25 retrieval and optional jurisdiction/category shards for legal datasets where scope matters.
- **Traceability over vibes.** Runs record events, tool calls, outputs, summaries, and evaluation artifacts so failures can be inspected rather than guessed at.
- **Structured outputs and validation.** Tasks can require JSON-shaped answers, and CAFL retries malformed final answers before scoring them.
- **Working memory, not hidden state.** During a run, compact working memory is injected back into the agent so it can notice prior tool results and avoid repeated unhelpful searches. Durable environment memory is explicit in `MEMORY.md`.
- **Parallel but debuggable.** Batch runs can execute concurrently, while preserving per-item traces and result ordering.

In short, CAFL is meant to make legal-agent evaluation less about rebuilding plumbing and more about studying the behavior that actually matters.

## Onboarding A New Task Environment

Create one folder under `envs/` for each benchmark or workflow. The runner expects the environment config at the environment root and data under `data/`:

```text
envs/my_task/
  config.json
  data/
    questions.jsonl
    corpus/
      docs.jsonl
  memory/
    MEMORY.md        # optional durable instructions
```

`questions.jsonl` should contain one JSON object per task. At minimum, include the field used as the task prompt and the field used as the ground-truth answer:

```json
{"question": "Does the tenant need notice before eviction?", "answer": "True"}
```

The corpus directory should contain JSONL records that the agent can retrieve from. The schema is flexible, but legal corpora usually work best with:

```json
{
  "citation": "AL Code § 35-9A-421 (2021)",
  "jurisdiction": "alabama",
  "text": "Full statute, case, regulation, or guidance text..."
}
```

Use `config.json` to tell CAFL how to read the task file, how to evaluate outputs, and how to build retrieval:

```json
{
  "task_file": "questions.jsonl",
  "corpus_dir": "corpus",
  "task_field": "question",
  "ground_truth_field": "answer",
  "agent_additional_system_prompt": "You are a housing specialist. Pay close attention to the jurisdiction in the question.",
  "bm25": {
    "index_path": "indexes/bm25.sqlite",
    "text_fields": ["text", "citation", "jurisdiction"],
    "shard_field": "jurisdiction"
  },
  "output_schema": {
    "answer": "a string either True or False",
    "citations": [
      {
        "citation": "a legal citation",
        "excerpt": "the relevant supporting text"
      }
    ]
  }
}
```

Required config fields are `task_file`, `corpus_dir`, `task_field`, `ground_truth_field`, and `output_schema`.

Optional but recommended BM25 fields:

- `text_fields`: fields to index for search. If omitted, CAFL indexes all textual values in each corpus record.
- `shard_field`: a short jurisdiction/category field used to build faster shard indexes. For legal datasets, use something like `state`, `jurisdiction`, `city`, `court`, `agency`, or `country` when tasks usually specify that value.
- `index_path`: where to write the main BM25 index, relative to the environment data directory.

Do not add `"memory_dir": "memory"` unless you want a custom path. CAFL already defaults to `envs/<env>/memory`. `MEMORY.md` is durable environment memory loaded at run start; per-run working memory stays in process on the run state.

Run the environment with:

```bash
python envs/runner.py --env my_task --rebuild --num_items 10 --max-concurrency 4
```

Use `--rebuild` after changing `bm25.text_fields`, `bm25.shard_field`, or the corpus. Run outputs are written under `envs/<env>/data/runs/` by default.

## Housing QA Example

Housing QA lives in `envs/housing_qa/` and is a good reference environment:

- `envs/housing_qa/download_housing_qa.py` downloads and converts the dataset.
- `envs/housing_qa/config.json` configures task fields, output schema, BM25 retrieval, and `shard_field: "state"`.
- `envs/housing_qa/memory/MEMORY.md` can hold durable retrieval or reasoning guidance.

Prepare the data:

```bash
python envs/housing_qa/download_housing_qa.py
```

Build retrieval shards and run a small batch:

```bash
python envs/runner.py --env housing_qa --rebuild --num_items 10 --max-concurrency 4
```

For Housing QA, the task questions include the state, and the corpus has a `state` field. Setting `bm25.shard_field` to `state` lets searches like `--filter state=alabama` route to a per-state BM25 index instead of scanning the full national corpus.
