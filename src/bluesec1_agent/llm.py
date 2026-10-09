import json
import re
import time
from types import TracebackType
from typing import Any, Protocol, Self, cast

from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import ValidationError

from bluesec1_agent.models import (
    NEXT_STEP_RESPONSE_FORMAT,
    FinishInvestigationRequest,
    NextStep,
)

STRUCTURED_OUTPUT_INSTRUCTION = """
Every assistant reply must be exactly one JSON object with these top-level fields:
{
  "current_state": "short evidence-focused string",
  "plan_remaining_steps_brief": ["one to five short step strings"],
  "task_completed": false,
  "function": {
    "tool": "one advertised tool name",
    "...": "that tool's arguments placed directly in this function object"
  }
}

Never use top-level `action`, `parameters`, `tool_name`, or `arguments` fields.
Never wrap tool arguments in a nested `parameters` or `arguments` object.
For example, a call with one advertised argument is:
{"tool":"advertised_tool","argument_name":"value"}
Set task_completed=true only for a function whose tool is finish_investigation.
Return no prose, markdown, XML, comments, or keys outside the required object.
""".strip()

REPAIR_INSTRUCTION = (
    "Your previous reply violated the required NextStep contract. " + STRUCTURED_OUTPUT_INSTRUCTION
)


class NextStepRequestError(RuntimeError):
    """Raised when an LLM cannot produce a valid structured step."""


class StructuredStepClient(Protocol):
    """Narrow async contract consumed by the investigation loop."""

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Generate and validate one investigation step."""
        ...


class OpenAIStructuredStepClient:
    """Portable structured-output client for OpenAI-compatible providers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_completion_tokens: int,
        max_retries: int,
        reasoning_effort: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Initialize an OpenAI-compatible structured step generator.

        Args:
            base_url: OpenAI-compatible API root.
            api_key: Provider credential.
            model: Provider model identifier.
            timeout_seconds: Timeout applied to each provider request.
            max_completion_tokens: Maximum generated tokens per attempt.
            max_retries: Maximum schema-generation attempts per step.
            client: Optional injected SDK client used by tests.
        """
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.last_usage: dict[str, int] | None = None
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=2,
        )

    async def __aenter__(self) -> Self:
        """Enter the async LLM client context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close an internally owned provider client."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the provider transport created by this wrapper."""
        if self._owns_client:
            await self._client.close()

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Return one valid `NextStep`, repairing malformed replies when possible.

        Args:
            messages: Current investigation conversation.

        Returns:
            A locally validated structured investigation step.

        Raises:
            NextStepRequestError: If every provider reply violates the schema.
        """
        request_messages = list(messages)
        if request_messages and request_messages[0].get("role") == "system":
            first_message = dict(request_messages[0])
            first_message["content"] = (
                f"{first_message.get('content', '')}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}"
            )
            request_messages[0] = first_message
        else:
            request_messages.insert(
                0,
                {"role": "system", "content": STRUCTURED_OUTPUT_INSTRUCTION},
            )
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            started_at = time.perf_counter()
            request: dict[str, Any] = {
                "model": self.model,
                "messages": cast(list[ChatCompletionMessageParam], request_messages),
                "response_format": cast(ResponseFormatJSONSchema, NEXT_STEP_RESPONSE_FORMAT),
                "max_completion_tokens": self.max_completion_tokens,
            }
            if self.reasoning_effort is not None:
                request["reasoning_effort"] = self.reasoning_effort
            response = await self._client.chat.completions.create(
                **request,
            )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            raw_content = extract_raw_content(response)
            usage = response_usage(response)
            self.last_usage = usage

            try:
                step = parse_next_step_payload(raw_content)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Invalid structured LLM response on attempt {}/{}: {}",
                    attempt,
                    self.max_retries,
                    raw_content[:2000] or str(exc),
                )
                if attempt == self.max_retries:
                    break
                request_messages.extend(
                    [
                        {"role": "assistant", "content": raw_content or "<empty>"},
                        {"role": "user", "content": REPAIR_INSTRUCTION},
                    ]
                )
                continue

            logger.info(
                "LLM selected {} in {} ms: input={} cached={} output={} reasoning={} total={}.",
                step.function.tool,
                latency_ms,
                usage["input_tokens"],
                usage["cached_tokens"],
                usage["output_tokens"],
                usage["reasoning_tokens"],
                usage["total_tokens"],
            )
            return step

        message = str(last_error or ValueError("Unable to parse model response."))
        raise NextStepRequestError(message) from last_error


def extract_message_text(message: Any) -> str:
    """Normalize string and multipart assistant content into plain text.

    Args:
        message: Provider-specific chat completion message.

    Returns:
        Concatenated textual message content.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
        elif getattr(part, "type", None) == "text":
            text_parts.append(str(getattr(part, "text", "")))
    return "".join(text_parts)


def extract_raw_content(response: Any) -> str:
    """Extract the first assistant message from a completion response.

    Args:
        response: OpenAI-compatible chat completion response.

    Returns:
        Raw assistant text.

    Raises:
        ValueError: If the provider response is missing a usable message.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Model response is missing choices.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("Model response is missing a message.")

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"Model refused structured output: {refusal}")
    return extract_message_text(message)


def extract_json_object(raw_content: str) -> str:
    """Remove common provider wrappers around a JSON object.

    Args:
        raw_content: Untrusted assistant response text.

    Returns:
        Candidate JSON object text for local validation.
    """
    stripped = raw_content.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    return match.group(0) if match is not None else stripped


def parse_next_step_payload(raw_content: str) -> NextStep:
    """Validate one raw provider reply as a consistent `NextStep`.

    Args:
        raw_content: Assistant response containing a structured step.

    Returns:
        Parsed and validated step.

    Raises:
        ValueError: If JSON, schema, or completion semantics are invalid.
    """
    if not raw_content:
        raise ValueError("Model returned empty content for NextStep response.")

    normalized_content = extract_json_object(raw_content)
    try:
        step = NextStep.model_validate_json(normalized_content)
    except ValidationError as exc:
        raise ValueError("Model returned invalid NextStep JSON.") from exc

    is_finish = isinstance(step.function, FinishInvestigationRequest)
    if step.task_completed != is_finish:
        step = step.model_copy(update={"task_completed": is_finish})
    return step


def response_total_tokens(response: Any) -> int:
    """Read total token usage when the provider supplies it.

    Args:
        response: OpenAI-compatible chat completion response.

    Returns:
        Non-negative total token count, or zero when unavailable.
    """
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0)
    return total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else 0


def response_usage(response: Any) -> dict[str, int]:
    """Normalize provider token accounting without assuming optional detail objects."""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
    output_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))
    total_tokens = getattr(usage, "total_tokens", 0)
    input_details = getattr(usage, "prompt_tokens_details", None)
    output_details = getattr(usage, "completion_tokens_details", None)
    cached_tokens = getattr(input_details, "cached_tokens", 0)
    reasoning_tokens = getattr(output_details, "reasoning_tokens", 0)

    def normalized(value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    return {
        "input_tokens": normalized(input_tokens),
        "cached_tokens": normalized(cached_tokens),
        "output_tokens": normalized(output_tokens),
        "reasoning_tokens": normalized(reasoning_tokens),
        "total_tokens": normalized(total_tokens),
    }


def render_json(value: Any) -> str:
    """Serialize participant-visible context for an LLM message.

    Args:
        value: Public JSON-compatible value.

    Returns:
        Stable readable JSON text.
    """
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
