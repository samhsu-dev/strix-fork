"""SDK run hooks used by Strix orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agents.lifecycle import RunHooks

from strix.report.state import get_global_report_state


if TYPE_CHECKING:
    from agents import RunContextWrapper
    from agents.agent import Agent
    from agents.items import ModelResponse, TResponseInputItem
    from agents.tool import Tool


logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when the accumulated LLM cost reaches the configured budget."""


class CompositeRunHooks(RunHooks[dict[str, Any]]):
    """Dispatch each run-hook callback to several child hooks in registration order."""

    def __init__(self, hooks: Sequence[RunHooks[dict[str, Any]]]) -> None:
        if not hooks:
            raise ValueError("CompositeRunHooks requires at least one child hook")
        self._hooks = tuple(hooks)

    async def _fanout(self, method: str, *args: Any) -> None:
        for hook in self._hooks:
            await getattr(hook, method)(*args)

    async def on_llm_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        await self._fanout("on_llm_start", context, agent, system_prompt, input_items)

    async def on_llm_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        response: ModelResponse,
    ) -> None:
        await self._fanout("on_llm_end", context, agent, response)

    async def on_tool_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        tool: Tool,
    ) -> None:
        await self._fanout("on_tool_start", context, agent, tool)

    async def on_tool_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        tool: Tool,
        result: str,
    ) -> None:
        await self._fanout("on_tool_end", context, agent, tool, result)

    async def on_agent_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
    ) -> None:
        await self._fanout("on_agent_start", context, agent)

    async def on_agent_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        output: Any,
    ) -> None:
        await self._fanout("on_agent_end", context, agent, output)

    async def on_handoff(
        self,
        context: RunContextWrapper[dict[str, Any]],
        from_agent: Agent[dict[str, Any]],
        to_agent: Agent[dict[str, Any]],
    ) -> None:
        await self._fanout("on_handoff", context, from_agent, to_agent)


class ReportUsageHooks(RunHooks[dict[str, Any]]):
    """Persist SDK-native usage after every model response."""

    def __init__(self, *, model: str, max_budget_usd: float | None = None) -> None:
        import math
        if max_budget_usd is not None and (not math.isfinite(max_budget_usd) or max_budget_usd <= 0):
            raise ValueError("max_budget_usd must be a finite number greater than 0")
        self._model = model
        self._max_budget_usd = max_budget_usd

    async def on_llm_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        response: ModelResponse,
    ) -> None:
        report_state = get_global_report_state()
        if report_state is None:
            return

        ctx = context.context if isinstance(context.context, dict) else {}
        agent_name = getattr(agent, "name", None)
        if not isinstance(agent_name, str):
            agent_name = None
        agent_id = ctx.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            agent_id = agent_name or "unknown"

        try:
            report_state.record_sdk_usage(
                agent_id=agent_id,
                agent_name=agent_name,
                model=self._model,
                usage=response.usage,
            )
        except Exception:
            logger.exception("failed to record SDK usage for agent %s", agent_id)

        if self._max_budget_usd is not None:
            cost = report_state.get_total_llm_cost()
            if cost >= self._max_budget_usd:
                raise BudgetExceededError(
                    f"Token budget of ${self._max_budget_usd:.2f} exceeded (spent ${cost:.4f})"
                )
