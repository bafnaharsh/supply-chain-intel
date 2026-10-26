"""One user turn through the ADK Runner, summarised for the CLI and tests."""

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from google.genai import types

from assistant.agent import K_GUARD, K_LLM, LLM_CAP_TEXT, estimate_usd, log
from assistant.result import AnswerResult
from core.config import ASSISTANT_APP_NAME, settings

__all__ = ["AnswerResult", "ask"]


async def ask(
    question: str,
    *,
    agent: LlmAgent,
    session_service: BaseSessionService,
    user_id: str,
    session_id: str,
) -> AnswerResult:
    """Run one question in an existing session; the Runner replays its earlier events."""
    runner = Runner(agent=agent, app_name=ASSISTANT_APP_NAME, session_service=session_service)
    message = types.Content(role="user", parts=[types.Part(text=question)])
    # Second belt behind the before_model guardrail, which blocks call max+1 first.
    run_config = RunConfig(
        max_llm_calls=settings.assistant_max_llm_calls,
        # The tools are synchronous HTTP calls; on the pool they no longer block the event loop,
        # so one slow tool call in the assistant API does not stall every other chat.
        tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
    )
    text, tools_called, p_tok, o_tok, guardrail = "", [], 0, 0, None
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message, run_config=run_config
        ):
            tools_called += [fc.name for fc in event.get_function_calls()]
            if event.usage_metadata:
                p_tok += event.usage_metadata.prompt_token_count or 0
                o_tok += event.usage_metadata.candidates_token_count or 0
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts)
    except LlmCallsLimitExceededError:
        text, guardrail = LLM_CAP_TEXT, "max_llm_calls"
        log.warning(
            "guardrail_triggered",
            extra={"guardrail": guardrail, "session_id": session_id, "user_id": user_id},
        )
    session = await session_service.get_session(
        app_name=ASSISTANT_APP_NAME, user_id=user_id, session_id=session_id
    )
    # The final output as the user sees it, after any guardrail replacement or cap.
    log.info(
        "final_answer",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "text": text,
            "tools_called": tools_called,
            "guardrail": guardrail or session.state.get(K_GUARD),
        },
    )
    model = agent.model if isinstance(agent.model, str) else agent.model.model
    return AnswerResult(
        text=text,
        tools_called=tools_called,
        llm_calls=min(session.state.get(K_LLM, 0), settings.assistant_max_llm_calls),
        usage={
            "prompt_tokens": p_tok,
            "output_tokens": o_tok,
            "est_usd": round(estimate_usd(model, p_tok, o_tok), 6),
        },
        guardrail=guardrail or session.state.get(K_GUARD),
    )
