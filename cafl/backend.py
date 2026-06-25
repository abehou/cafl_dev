'''
Design idea of the backend:
In order to support async + parallelism, we make:
- Cafl is stateless and knows how to execute the RunState
- RunState is the mutatable record, basically the 
prompt + all the past contexts that the agent needs to be aware of; 
- Cafl.run_many_async() can run over many RunState objects in parallel
- stream() yields MiniEvent objects; run() records them for traces/logging.
'''
import os
os.environ["MSWEA_SILENT_STARTUP"] = "1"
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models.litellm_model import LitellmModel, parse_toolcall_actions
from minisweagent.utils.serialize import recursive_merge
from jinja2 import StrictUndefined, Template
from dataclasses import asdict, dataclass, field
from collections.abc import AsyncGenerator, Generator, Iterator
import asyncio
from pathlib import Path
from .config import CaflConfig
from .utils.formatting import message_content, result_record, stringify, write_batch_summary, write_summary
from .utils.utils import append_jsonl, get_path_time_signature, get_time_signature, safe_slug


'''
A single observation, can be a single agent message, or a tool call, or a tool call result, etc. 
Essentially the finegrained action unit the agent takes or the feedback received.
'''
@dataclass
class MiniEvent:
    run_id: str
    task_id: str
    item_id: str | None # the index of the item in the task, if applicable
    role: str # "tool", "agent" etc.
    content: str
    index: int
    status: str # "pending", "completed", "failed", "cancelled" etc.
    event_id: int | None = None
    error: str | None = None

@dataclass
class RunState:
    run_id: str
    task_id: str
    item_id: str | None = None
    messages: list[dict] = field(default_factory=list) # need to use default_factory to avoid mutable default argument; otherwise can point to the same list.
    extra_template_vars: dict = field(default_factory=dict)
    cost: float = 0.0
    n_calls: int = 0
    next_event_index: int = 0


@dataclass
class CaflResult:
    run_id: str
    task_id: str
    item_id: str | None
    question: str
    answer: str
    events: list[MiniEvent]
    state: RunState
    output_dir: Path | None = None
    status: str = "completed"

class ToolLitellmModel(LitellmModel):
    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)


class Cafl(DefaultAgent):
    def __init__(self, model=None, env=None, *args, **kwargs):
        cafl_config = kwargs.pop("cafl_config", None)
        self.cafl_config = cafl_config if cafl_config is not None else CaflConfig()
        self.event_logger = kwargs.pop("event_logger", None)
        kwargs.setdefault("system_template", self.cafl_config.system_template)
        kwargs.setdefault("instance_template", self.cafl_config.instance_template)
        self._model_from_name = model is None or isinstance(model, str)
        model = self._make_model(model or self.cafl_config.default_model, self.cafl_config) if self._model_from_name else model
        env = env if env is not None else LocalEnvironment()
        super().__init__(model, env, *args, **kwargs)

    @staticmethod
    def _make_model(model_name: str, cafl_config: CaflConfig) -> ToolLitellmModel:
        return ToolLitellmModel(
            model_name=cafl_config.resolve_model_name(model_name),
            model_kwargs={
                "temperature": 0,
            },
            observation_template=cafl_config.observation_template,
            cost_tracking="ignore_errors",
        )

    def run(
        self,
        task: str = "",
        *,
        state: RunState | None = None,
        output_root: Path | str | None = "runs",
        output_dir: Path | str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> CaflResult:
        question = task
        output_path = self._prepare_question_dir(question, output_root=output_root, output_dir=output_dir)
        state = self._init_run(question, state=state, **kwargs)
        if output_path is not None:
            append_jsonl(output_path / "results.jsonl", result_record("question", state, question))
        events: list[MiniEvent] = []
        trace_path = output_path / "trace.jsonl" if output_path is not None else None
        for event in self.stream(
            question,
            state=state,
            max_tokens=max_tokens,
            timeout=timeout,
            initialize=False,
            **kwargs,
        ):
            self._record_event(event, events, trace_path)
        answer, answer_message = self._extract_answer(state)
        result = CaflResult(
            run_id=state.run_id,
            task_id=state.task_id,
            item_id=state.item_id,
            question=question,
            answer=answer,
            events=events,
            state=state,
            output_dir=output_path,
        )
        if output_path is not None:
            append_jsonl(output_path / "results.jsonl", result_record("answer", state, answer, message=answer_message))
            write_summary(output_path / "summary.json", result)
        return result

    async def run_async(
        self,
        task: str,
        *,
        state: RunState | None = None,
        output_root: Path | str | None = "runs",
        output_dir: Path | str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> CaflResult:
        return await asyncio.to_thread(
            self.run,
            task,
            state=state,
            output_root=output_root,
            output_dir=output_dir,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

    def run_many(
        self,
        tasks: list[str],
        *,
        output_root: Path | str | None = "runs",
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[CaflResult]:
        return asyncio.run(
            self.run_many_async(
                tasks,
                output_root=output_root,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
        )

    async def run_many_async(
        self,
        tasks: list[str],
        *,
        output_root: Path | str | None = "runs",
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[CaflResult]:
        batch_dir = self._prepare_batch_dir(output_root) if output_root is not None else None
        results = await asyncio.gather(
            *(
                self.run_async(
                    task,
                    state=RunState(
                        run_id=batch_dir.name if batch_dir is not None else f"run-{index}",
                        task_id="run-many",
                        item_id=f"item-{index:03d}",
                    ),
                    output_root=None,
                    output_dir=(batch_dir / f"item-{index:03d}") if batch_dir is not None else None,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    **kwargs,
                )
                for index, task in enumerate(tasks)
            )
        )
        if batch_dir is not None:
            write_batch_summary(batch_dir / "summary.json", results)
        return results

    def _prepare_question_dir(
        self,
        question: str,
        *,
        output_root: Path | str | None,
        output_dir: Path | str | None,
    ) -> Path | None:
        if output_dir is not None:
            path = Path(output_dir)
        elif output_root is not None:
            path = Path(output_root) / f"{get_path_time_signature()}-{safe_slug(question)}"
        else:
            return None
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _prepare_batch_dir(self, output_root: Path | str | None) -> Path:
        path = Path(output_root) / f"{get_path_time_signature()}-batch"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _extract_answer(self, state: RunState) -> tuple[str, dict]:
        for message in reversed(state.messages):
            if message.get("role") == "exit":
                extra = message.get("extra", {})
                if "submission" in extra:
                    return stringify(extra["submission"]), message
        for message in reversed(state.messages):
            if message.get("role") == "assistant" and not message.get("extra", {}).get("actions", []):
                return message_content(message), message
        raise RuntimeError("Cannot extract answer because no final assistant or exit message was recorded.")

    def _record_event(self, event: MiniEvent, events: list[MiniEvent], trace_path: Path | None) -> None:
        event.event_id = len(events)
        events.append(event)
        if trace_path is not None:
            append_jsonl(trace_path, asdict(event))
        if self.event_logger is not None:
            self.event_logger(event)

    def _init_run(self, task: str = "", *, state: RunState | None = None, **kwargs) -> RunState:
        timestamp = get_time_signature()
        run_id = f"run-{timestamp}"
        task_id = f"task-{timestamp}" if task == "" else f"task-{task}-{timestamp}"
        state = state or RunState(run_id=run_id, task_id=task_id)
        state.extra_template_vars |= {"task": task, **kwargs}
        state.messages.clear()
        state.next_event_index = 0

        self._add_messages(
            state,
            self.model.format_message(role="system", content=self._render_template_for_state(state, self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template_for_state(state, self.config.instance_template)),
        )
        return state

    def stream(
        self,
        task: str = "",
        *,
        state: RunState | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        initialize: bool = True,
        **kwargs,
    ) -> Generator[MiniEvent, None, dict]:
        state = self._init_run(task, state=state, **kwargs) if initialize else state
        
        while True:
            try:
                yield from self._stream_step(state, max_tokens=max_tokens, timeout=timeout)
            except InterruptAgentFlow as e:
                self._add_messages(state, *e.messages)
                for message in e.messages:
                    yield self._emit_message_event(state, message)
            except Exception as e:
                for message in self._handle_uncaught_exception(state, e):
                    yield self._emit_message_event(state, message, status="failed", error=str(e))
                raise
            if self._should_stop(state):
                return state.messages[-1].get("extra", {})

    async def stream_async(
        self,
        task: str = "",
        *,
        state: RunState | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        initialize: bool = True,
        **kwargs,
    ) -> AsyncGenerator[MiniEvent, None]:
        state = self._init_run(task, state=state, **kwargs) if initialize else state

        while True:
            try:
                async for event in self._stream_step_async(state, max_tokens=max_tokens, timeout=timeout):
                    yield event
            except InterruptAgentFlow as e:
                self._add_messages(state, *e.messages)
                for message in e.messages:
                    yield self._emit_message_event(state, message)
            except Exception as e:
                for message in self._handle_uncaught_exception(state, e):
                    yield self._emit_message_event(state, message, status="failed", error=str(e))
                raise
            if self._should_stop(state):
                return

    def _stream_step(
        self,
        state: RunState,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> Iterator[MiniEvent]:
        message = self._query(state, max_tokens=max_tokens, timeout=timeout)
        yield self._emit_message_event(state, message)

        outputs = []
        actions = message.get("extra", {}).get("actions", [])
        if message.get("role") == "exit" or not actions:
            return
        for action in actions:
            yield self._make_event(state, "tool_call", repr(action), "pending")
            outputs.append(self.env.execute(action))

        observation_messages = self.model.format_observation_messages(message, outputs, self._get_template_vars(state))
        self._add_messages(state, *observation_messages)
        for observation_message in observation_messages:
            yield self._emit_message_event(state, observation_message)

    # async streaming for websocket / UI
    async def _stream_step_async(
        self,
        state: RunState,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[MiniEvent, None]:
        message = await asyncio.to_thread(self._query, state, max_tokens=max_tokens, timeout=timeout)
        yield self._emit_message_event(state, message)

        outputs = []
        actions = message.get("extra", {}).get("actions", [])
        if message.get("role") == "exit" or not actions:
            return
        for action in actions:
            yield self._make_event(state, "tool_call", repr(action), "pending")
            outputs.append(await asyncio.to_thread(self.env.execute, action))

        observation_messages = self.model.format_observation_messages(message, outputs, self._get_template_vars(state))
        self._add_messages(state, *observation_messages)
        for observation_message in observation_messages:
            yield self._emit_message_event(state, observation_message)

    def _emit_message_event(
        self,
        state: RunState,
        message: dict,
        status: str = "completed",
        error: str | None = None,
    ) -> MiniEvent:
        return self._make_event(
            state=state,
            role=str(message.get("role", "assistant")),
            content=message_content(message),
            status=status,
            error=error,
        )

    def _make_event(
        self,
        state: RunState,
        role: str,
        content: str,
        status: str,
        error: str | None = None,
    ) -> MiniEvent:
        event = MiniEvent(
            run_id=state.run_id,
            task_id=state.task_id,
            item_id=state.item_id,
            role=role,
            content=content,
            index=state.next_event_index,
            status=status,
            error=error,
        )
        state.next_event_index += 1
        return event

    @staticmethod
    def _should_stop(state: RunState) -> bool:
        if not state.messages:
            return False
        last_message = state.messages[-1]
        return last_message.get("role") == "exit" or (
            last_message.get("role") == "assistant"
            and not last_message.get("extra", {}).get("actions", [])
        )

    def _query(
        self,
        state: RunState,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        if 0 < self.config.step_limit <= state.n_calls or 0 < self.config.cost_limit <= state.cost:
            raise InterruptAgentFlow(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        state.n_calls += 1
        if self._model_from_name:
            query_kwargs = {}
            if max_tokens is not None:
                query_kwargs["max_tokens"] = max_tokens
            if timeout is not None:
                query_kwargs["timeout"] = timeout
            message = self.model.query(state.messages, **query_kwargs)
        else:
            message = self.model.query(state.messages)
        state.cost += message.get("extra", {}).get("cost", 0.0)
        self._add_messages(state, message)
        return message

    def _add_messages(self, state: RunState, *messages: dict) -> list[dict]:
        self.logger.debug(messages)
        state.messages.extend(messages)
        return list(messages)

    def _handle_uncaught_exception(self, state: RunState, e: Exception) -> list[dict]:
        return self._add_messages(
            state,
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                },
            ),
        )

    def _get_template_vars(self, state: RunState, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": state.n_calls, "model_cost": state.cost},
            state.extra_template_vars,
            kwargs,
        )

    def _render_template_for_state(self, state: RunState, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self._get_template_vars(state))
