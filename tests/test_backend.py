import asyncio
import json
from types import SimpleNamespace

import pytest
from minisweagent.environments.local import LocalEnvironment

from cafl.backend import Cafl, MiniEvent, RunState
from cafl.config import CaflConfig
from cafl.logging import EventLogger, DEFAULT_MAX_EVENT_CHARS


class FakeModel:
    def __init__(self):
        self.calls = 0

    def query(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "run command",
                "extra": {"actions": [{"cmd": "echo hi"}]},
            }
        return {
            "role": "exit",
            "content": "done",
            "extra": {"exit_status": "submitted", "submission": "ok"},
        }

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [
            {
                "role": "tool",
                "content": output["output"],
                "extra": {"output": output},
            }
            for output in outputs
        ]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class FakeEnv:
    def execute(self, action, cwd=""):
        return {"output": f"executed {action['cmd']}"}

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class FakeToolEnv:
    def execute(self, action, cwd=""):
        return {"output": f"executed {action['command']}", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class StatelessModel(FakeModel):
    def query(self, messages):
        assistant_or_exit_count = sum(1 for message in messages if message["role"] in {"assistant", "exit"})
        if assistant_or_exit_count == 0:
            return {
                "role": "assistant",
                "content": "run command",
                "extra": {"actions": [{"cmd": "echo hi"}]},
            }
        return {
            "role": "exit",
            "content": "done",
            "extra": {"exit_status": "submitted", "submission": "ok"},
        }


class LocalCommandModel(FakeModel):
    def query(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "run command",
                "extra": {"actions": [{"command": "printf cafl-default-env"}]},
            }
        return {
            "role": "exit",
            "content": "done",
            "extra": {"exit_status": "submitted", "submission": "ok"},
        }


class PlainAnswerModel(FakeModel):
    def query(self, messages):
        self.calls += 1
        return {
            "role": "assistant",
            "content": f"answer: {messages[-1]['content']}",
            "extra": {"cost": 0.01},
        }


class AlwaysToolModel(FakeModel):
    def query(self, messages):
        self.calls += 1
        return {
            "role": "assistant",
            "content": "keep searching",
            "extra": {"actions": [{"cmd": f"echo {self.calls}"}]},
        }


class RepeatToolThenAnswerModel(FakeModel):
    def query(self, messages):
        self.calls += 1
        if messages[-1]["role"] == "user" and "repeated the same tool call" in messages[-1]["content"]:
            return {
                "role": "assistant",
                "content": "final from available evidence",
                "extra": {"actions": []},
            }
        return {
            "role": "assistant",
            "content": "repeat search",
            "extra": {"actions": [{"cmd": "echo same", "tool_call_id": f"call-{self.calls}"}]},
        }


class WorkingMemoryAwareModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.seen_messages = []

    def query(self, messages):
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "inspect",
                "extra": {"actions": [{"cmd": "echo evidence"}]},
            }
        assert "## Working Memory" in messages[-1]["content"]
        assert "executed echo evidence" in messages[-1]["content"]
        return {
            "role": "assistant",
            "content": "answer from memory",
            "extra": {"actions": []},
        }


class InvalidThenValidJsonModel(FakeModel):
    def query(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "not json",
                "extra": {"cost": 0.01},
            }
        return {
            "role": "assistant",
            "content": json.dumps({"answer": "True"}),
            "extra": {"cost": 0.01},
        }


class FakeLiteLLMMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class FakeLiteLLMResponse:
    def __init__(self, content, tool_calls=None):
        self.choices = [type("Choice", (), {"message": FakeLiteLLMMessage(content, tool_calls)})()]

    def model_dump(self):
        return {"choices": [{"message": self.choices[0].message.model_dump()}]}


def fake_bash_tool_call(command="echo hi"):
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="bash", arguments=json.dumps({"command": command})),
    )


def test_run_returns_plain_answer_without_env():
    agent = Cafl(
        PlainAnswerModel(),
        system_template="system",
        instance_template="{{ task }}",
    )

    result = agent.run("what is CAFL?", output_root=None)

    assert result.answer == "answer: what is CAFL?"
    assert agent.model.calls == 1


def test_stream_uses_local_environment_by_default():
    agent = Cafl(LocalCommandModel(), system_template="system", instance_template="{{ task }}")

    events = list(agent.stream("what is CAFL?"))

    assert isinstance(agent.env, LocalEnvironment)
    assert [(event.role, event.content) for event in events] == [
        ("assistant", "run command"),
        ("tool_call", "{'command': 'printf cafl-default-env'}"),
        ("tool", "cafl-default-env"),
        ("exit", "done"),
    ]


def test_default_system_template_includes_tool_and_submission_protocol():
    agent = Cafl(PlainAnswerModel())

    system_message = agent.run("what is CAFL?", output_root=None).state.messages[0]["content"]

    assert "Use bash tool calls when you need" in system_message
    assert "answer directly without a tool call" in system_message
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in system_message
    assert "remaining output is the final" in system_message


def test_default_system_template_renders_environment_retrieval_vars():
    agent = Cafl(PlainAnswerModel())

    system_message = agent.run(
        "what is CAFL?",
        output_root=None,
        task_environment_instructions="Use local statutes.",
        corpus_dir="/tmp/corpus",
        bm25_instructions="## BM25 Corpus Search\npython -m cafl.tools.retrieval search",
    ).state.messages[0]["content"]

    assert "Use local statutes." in system_message
    assert "The local corpus directory is: /tmp/corpus" in system_message
    assert "## BM25 Corpus Search" in system_message
    assert "python -m cafl.tools.retrieval search" in system_message


def test_environment_memory_is_added_to_system_prompt(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("Use state filters before broad retrieval.", encoding="utf-8")
    agent = Cafl(
        PlainAnswerModel(),
        cafl_config=CaflConfig(memory_dir=str(memory_dir)),
    )

    system_message = agent.run("what is CAFL?", output_root=None).state.messages[0]["content"]

    assert "## Environment Memory" in system_message
    assert "Use state filters before broad retrieval." in system_message


def test_run_keeps_working_memory_in_state_without_writing_run_memory(tmp_path):
    memory_dir = tmp_path / "memory"
    agent = Cafl(
        FakeModel(),
        FakeEnv(),
        cafl_config=CaflConfig(system_template="{{ task }}", memory_dir=str(memory_dir)),
    )

    result = agent.run("what is CAFL?", output_root=None)

    assert result.state.memory["actions"] == [{"cmd": "echo hi"}]
    assert result.state.memory["tool_observations"][0]["summary"] == '{"output": "executed echo hi"}'
    assert not (memory_dir / "run_memory.jsonl").exists()


def test_output_schema_is_added_to_system_prompt():
    agent = Cafl(
        PlainAnswerModel(),
        cafl_config=CaflConfig(output_schema={"answer": "a string"}, output_validation_retries=0),
    )

    system_message = agent.run("what is CAFL?", output_root=None).state.messages[0]["content"]

    assert "## Output Schema" in system_message
    assert '"answer": "a string"' in system_message
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in system_message
    assert "Do not use a bash tool call, heredoc" in system_message


def test_output_schema_invalid_answer_triggers_retry():
    agent = Cafl(
        InvalidThenValidJsonModel(),
        cafl_config=CaflConfig(output_schema={"answer": "a string"}),
    )

    result = agent.run("answer with json", output_root=None)

    assert result.answer == '{"answer": "True"}'
    assert agent.model.calls == 2
    assert [(event.role, event.status) for event in result.events] == [
        ("assistant", "completed"),
        ("user", "failed"),
        ("assistant", "completed"),
    ]
    assert "did not match the required JSON output schema" in result.events[1].content
    assert result.state.output_validation_failures == 1


def test_default_step_limit_stops_unbounded_tool_loop():
    agent = Cafl(
        AlwaysToolModel(),
        FakeEnv(),
        system_template="system",
        instance_template="task",
        cafl_config=CaflConfig(step_limit=2, repeated_tool_call_limit=0),
    )

    result = agent.run("loop", output_root=None)

    assert result.answer == ""
    assert result.state.n_calls == 2
    assert result.events[-1].role == "exit"
    assert result.events[-1].content == "LimitsExceeded"


def test_repeated_tool_call_interrupts_with_calibration_message():
    env = FakeEnv()
    agent = Cafl(
        RepeatToolThenAnswerModel(),
        env,
        system_template="system",
        instance_template="task",
        cafl_config=CaflConfig(repeated_tool_call_limit=3),
    )

    result = agent.run("loop", output_root=None)

    assert result.answer == "final from available evidence"
    assert [event.role for event in result.events].count("tool_call") == 2
    assert any(event.role == "user" and event.status == "failed" for event in result.events)
    assert "repeated the same tool call" in result.state.messages[-2]["content"]
    assert "Recalibrate your retrieval strategy" in result.state.messages[-2]["content"]
    assert result.state.messages[-2]["extra"]["interrupt_type"] == "RepeatedToolCallCalibration"
    assert "action_counts" in result.state.memory


def test_working_memory_is_injected_without_persisting_to_state_messages():
    model = WorkingMemoryAwareModel()
    agent = Cafl(
        model,
        FakeEnv(),
        system_template="system",
        instance_template="task",
    )

    result = agent.run("use memory", output_root=None)

    assert result.answer == "answer from memory"
    assert model.calls == 2
    assert "## Working Memory" in model.seen_messages[1][-1]["content"]
    assert not any(message.get("extra", {}).get("memory_type") == "working_memory" for message in result.state.messages)
    assert result.state.memory["actions"] == [{"cmd": "echo evidence"}]
    assert "executed echo evidence" in result.state.memory["tool_observations"][-1]["summary"]


def test_named_model_can_answer_without_tool_call(monkeypatch):
    calls = {}

    def fake_completion(**kwargs):
        calls.update(kwargs)
        return FakeLiteLLMResponse("direct answer", None)

    monkeypatch.setattr("minisweagent.models.litellm_model.litellm.completion", fake_completion)
    agent = Cafl(model="gemini-3-flash")

    result = agent.run("what is CAFL?", output_root=None, max_tokens=32, timeout=5)

    assert result.answer == "direct answer"
    assert "temperature" not in calls
    assert calls["tools"][0]["function"]["name"] == "bash"
    assert [(event.role, event.content) for event in result.events] == [("assistant", "direct answer")]
    assert not any(message.get("extra", {}).get("interrupt_type") == "FormatError" for message in result.state.messages)


def test_run_accepts_gemini_model_name(monkeypatch):
    calls = {}

    def fake_completion(**kwargs):
        calls.update(kwargs)
        return FakeLiteLLMResponse("inspect", [fake_bash_tool_call()])

    monkeypatch.setattr("minisweagent.models.litellm_model.litellm.completion", fake_completion)
    agent = Cafl(model="gemini-3-flash", step_limit=1)

    result = agent.run("what is CAFL?", output_root=None, max_tokens=32, timeout=5)

    assert result.answer == ""
    assert calls["model"] == "gemini/gemini-3-flash-preview"
    assert "temperature" not in calls
    assert calls["max_tokens"] == 32
    assert calls["timeout"] == 5
    assert calls["tools"][0]["function"]["name"] == "bash"
    assert result.events[1].role == "tool_call"


def test_named_non_gemini_3_model_keeps_temperature(monkeypatch):
    calls = {}

    def fake_completion(**kwargs):
        calls.update(kwargs)
        return FakeLiteLLMResponse("direct answer", None)

    monkeypatch.setattr("minisweagent.models.litellm_model.litellm.completion", fake_completion)
    agent = Cafl(model="provider/fast")

    result = agent.run("what is CAFL?", output_root=None)

    assert result.answer == "direct answer"
    assert calls["model"] == "provider/fast"
    assert calls["temperature"] == 0


def test_named_model_truncates_large_tool_output(monkeypatch):
    def fake_completion(**kwargs):
        command = "python -c \"print('x' * 15000)\""
        return FakeLiteLLMResponse("inspect", [fake_bash_tool_call(command)])

    monkeypatch.setattr("minisweagent.models.litellm_model.litellm.completion", fake_completion)
    agent = Cafl(model="gemini-3-flash", step_limit=1)

    result = agent.run("inspect something noisy", output_root=None, max_tokens=32, timeout=5)

    tool_event = next(event for event in result.events if event.role == "tool")
    assert "Tool output was truncated" in tool_event.content
    assert "<output_head>" in tool_event.content
    assert "<output_tail>" in tool_event.content
    assert len(tool_event.content) < 13000


def test_run_uses_injected_cafl_config(monkeypatch):
    calls = {}

    def fake_completion(**kwargs):
        calls.update(kwargs)
        return FakeLiteLLMResponse("inspect", [fake_bash_tool_call()])

    monkeypatch.setattr("minisweagent.models.litellm_model.litellm.completion", fake_completion)
    agent = Cafl(
        model="fast",
        env=FakeToolEnv(),
        step_limit=1,
        cafl_config=CaflConfig(
            default_model="provider/default",
            model_aliases={"fast": "provider/fast"},
            system_template="system: {{ flavor }}",
            instance_template="task: {{ task }}",
        ),
    )

    result = agent.run("what is CAFL?", output_root=None, flavor="vanilla")

    assert result.answer == ""
    assert calls["model"] == "provider/fast"
    assert calls["messages"] == [
        {"role": "system", "content": "system: vanilla"},
        {"role": "user", "content": "task: what is CAFL?"},
    ]


def test_event_logger_writes_concise_file(tmp_path):
    log_path = tmp_path / "events.log"
    logger = EventLogger(log_path)

    logger(
        MiniEvent(
            run_id="run",
            task_id="task",
            item_id="item-000",
            role="tool",
            content="x" * (DEFAULT_MAX_EVENT_CHARS + 10),
            index=0,
            status="completed",
        )
    )

    logged = log_path.read_text()
    assert "... <truncated 10 chars>" in logged
    assert "[item-000 #0 tool/completed]" in logged
    assert len(logged) < DEFAULT_MAX_EVENT_CHARS + 200


@pytest.mark.asyncio
async def test_run_many_async_preserves_question_order_without_env():
    agent = Cafl(
        PlainAnswerModel(),
        system_template="system",
        instance_template="{{ task }}",
    )

    results = await agent.run_many_async(["first", "second", "third"], output_root=None)

    assert [result.answer for result in results] == ["answer: first", "answer: second", "answer: third"]


@pytest.mark.asyncio
async def test_run_many_async_can_show_progress(monkeypatch):
    progress_calls = []

    class FakeProgress:
        def __init__(self, iterable, *, total, desc, unit):
            progress_calls.append({"total": total, "desc": desc, "unit": unit})
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr("tqdm.tqdm", FakeProgress)
    agent = Cafl(
        PlainAnswerModel(),
        system_template="system",
        instance_template="{{ task }}",
    )

    results = await agent.run_many_async(
        ["first", "second"],
        output_root=None,
        show_progress=True,
        progress_desc="Testing agents",
    )

    assert [result.answer for result in results] == ["answer: first", "answer: second"]
    assert progress_calls == [{"total": 2, "desc": "Testing agents", "unit": "task"}]


@pytest.mark.asyncio
async def test_run_many_async_limits_max_concurrency(monkeypatch):
    running = 0
    max_seen = 0
    lock = asyncio.Lock()
    original_run_async = Cafl.run_async

    async def tracked_run_async(self, *args, **kwargs):
        nonlocal running, max_seen
        async with lock:
            running += 1
            max_seen = max(max_seen, running)
        await asyncio.sleep(0.01)
        try:
            return await original_run_async(self, *args, **kwargs)
        finally:
            async with lock:
                running -= 1

    monkeypatch.setattr(Cafl, "run_async", tracked_run_async)
    agent = Cafl(
        PlainAnswerModel(),
        system_template="system",
        instance_template="{{ task }}",
    )

    results = await agent.run_many_async(
        ["first", "second", "third", "fourth"],
        output_root=None,
        max_concurrency=2,
    )

    assert [result.answer for result in results] == [
        "answer: first",
        "answer: second",
        "answer: third",
        "answer: fourth",
    ]
    assert max_seen <= 2


@pytest.mark.asyncio
async def test_run_many_async_rejects_invalid_max_concurrency():
    agent = Cafl(
        PlainAnswerModel(),
        system_template="system",
        instance_template="{{ task }}",
    )

    with pytest.raises(ValueError, match="max_concurrency"):
        await agent.run_many_async(["first"], output_root=None, max_concurrency=0)


def test_run_writes_results_trace_and_summary(tmp_path):
    agent = Cafl(
        PlainAnswerModel(),
        system_template="system",
        instance_template="{{ task }}",
    )

    result = agent.run("what is CAFL?", output_root=tmp_path)

    assert result.output_dir is not None
    assert result.output_dir.parent == tmp_path
    results_records = [json.loads(line) for line in (result.output_dir / "results.jsonl").read_text().splitlines()]
    trace_records = [json.loads(line) for line in (result.output_dir / "trace.jsonl").read_text().splitlines()]
    summary = json.loads((result.output_dir / "summary.json").read_text())
    assert [record["type"] for record in results_records] == ["question", "answer"]
    assert results_records[0]["content"] == "what is CAFL?"
    assert results_records[1]["content"] == "answer: what is CAFL?"
    assert trace_records[0]["role"] == "assistant"
    assert summary["answer"] == "answer: what is CAFL?"
    assert summary["n_events"] == 1


@pytest.mark.asyncio
async def test_run_many_async_writes_batch_subfolders(tmp_path):
    agent = Cafl(
        PlainAnswerModel(),
        system_template="system",
        instance_template="{{ task }}",
    )

    results = await agent.run_many_async(["first", "second"], output_root=tmp_path)

    batch_dir = results[0].output_dir.parent
    assert [result.output_dir.name for result in results] == ["item-000", "item-001"]
    assert (batch_dir / "summary.json").exists()
    batch_summary = json.loads((batch_dir / "summary.json").read_text())
    assert batch_summary["n_items"] == 2
    assert [item["answer"] for item in batch_summary["items"]] == ["answer: first", "answer: second"]


def test_stream_yields_raw_mini_events_in_order():
    state = RunState(run_id="run-1", task_id="task-1", item_id="item-1")
    agent = Cafl(
        FakeModel(),
        FakeEnv(),
        system_template="system",
        instance_template="task",
    )

    events = list(agent.stream("do it", state=state))

    assert [(event.role, event.content, event.status) for event in events] == [
        ("assistant", "run command", "completed"),
        ("tool_call", "{'cmd': 'echo hi'}", "pending"),
        ("tool", "executed echo hi", "completed"),
        ("exit", "done", "completed"),
    ]
    assert [event.index for event in events] == [0, 1, 2, 3]
    assert [event.run_id for event in events] == ["run-1"] * 4
    assert [event.task_id for event in events] == ["task-1"] * 4
    assert [event.item_id for event in events] == ["item-1"] * 4
    assert [event.event_id for event in events] == [None, None, None, None]
    assert state.next_event_index == 4


def test_run_agentic_mode_returns_result_and_persists_trace(tmp_path):
    logged_events = []
    agent = Cafl(
        FakeModel(),
        FakeEnv(),
        system_template="system",
        instance_template="task",
        event_logger=logged_events.append,
    )

    result = agent.run("do it", output_root=tmp_path)

    assert result.answer == "ok"
    assert [(event.role, event.content, event.status) for event in result.events] == [
        ("assistant", "run command", "completed"),
        ("tool_call", "{'cmd': 'echo hi'}", "pending"),
        ("tool", "executed echo hi", "completed"),
        ("exit", "done", "completed"),
    ]
    assert logged_events == result.events
    assert result.state.events == result.events
    assert result.state.output_dir == result.output_dir
    assert [event.event_id for event in result.events] == [0, 1, 2, 3]
    trace_records = [json.loads(line) for line in (result.output_dir / "trace.jsonl").read_text().splitlines()]
    assert [record["role"] for record in trace_records] == ["assistant", "tool_call", "tool", "exit"]
    assert [record["event_id"] for record in trace_records] == [0, 1, 2, 3]
    tool_trace_records = [json.loads(line) for line in (result.output_dir / "tool_trace.jsonl").read_text().splitlines()]
    assert tool_trace_records == [
        {
            "event_index": 1,
            "tool": "bash",
            "arguments": {"cmd": "echo hi"},
        }
    ]


def test_same_agent_can_stream_independent_states():
    agent = Cafl(
        FakeModel(),
        FakeEnv(),
        system_template="system",
        instance_template="task",
    )

    first = list(
        agent.stream(
            "do first",
            state=RunState(run_id="run-1", task_id="task", item_id="item-1"),
        )
    )
    agent.model.calls = 0
    second_state = RunState(run_id="run-2", task_id="task", item_id="item-2")
    second = list(agent.stream("do second", state=second_state))

    assert [event.index for event in first] == [0, 1, 2, 3]
    assert [event.index for event in second] == [0, 1, 2, 3]
    assert [event.run_id for event in first + second] == ["run-1"] * 4 + ["run-2"] * 4
    assert second_state.messages[-1]["role"] == "exit"


def test_stream_creates_state_when_none_is_provided():
    agent = Cafl(
        FakeModel(),
        FakeEnv(),
        system_template="system",
        instance_template="task",
    )

    events = list(agent.stream("do it"))

    assert [event.index for event in events] == [0, 1, 2, 3]
    assert all(event.run_id.startswith("run-") for event in events)
    assert all(event.task_id.startswith("task-do it-") for event in events)


@pytest.mark.asyncio
async def test_stream_async_streams_independent_states():
    agent = Cafl(
        StatelessModel(),
        FakeEnv(),
        system_template="system",
        instance_template="task",
    )

    async def collect(run_id, item_id):
        state = RunState(run_id=run_id, task_id="task", item_id=item_id)
        return [event async for event in agent.stream_async("do it", state=state)]

    first, second = await asyncio.gather(
        collect("run-1", "item-1"),
        collect("run-2", "item-2"),
    )

    assert [event.index for event in first] == [0, 1, 2, 3]
    assert [event.index for event in second] == [0, 1, 2, 3]
    assert sorted(event.run_id for event in first + second) == ["run-1"] * 4 + ["run-2"] * 4


@pytest.mark.asyncio
async def test_stream_async_creates_state_when_none_is_provided():
    agent = Cafl(
        FakeModel(),
        FakeEnv(),
        system_template="system",
        instance_template="task",
    )

    events = [event async for event in agent.stream_async("do it")]

    assert [event.index for event in events] == [0, 1, 2, 3]
    assert all(event.run_id.startswith("run-") for event in events)
    assert all(event.task_id.startswith("task-do it-") for event in events)
