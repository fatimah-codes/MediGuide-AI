"""LangChain pipeline: ChatOpenAI, LLMChain, streaming, message demo, cache benchmark."""
import time
import warnings

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from . import cache_manager as cm
from .prompts import (
    ASSESSMENT_CHAT_TEMPLATE,
    DEMO_FOLLOW_UP,
    DEMO_PREVIOUS_AI_REPLY,
    DEMO_SYSTEM,
    JSON_SCHEMA,
    NARRATIVE_CHAT_TEMPLATE,
)

# LLMChain still works but newer LangChain versions print deprecation warnings.
warnings.filterwarnings("ignore", message=".*LLMChain.*")

# LLMChain lives in different packages depending on the LangChain version.
try:
    from langchain.chains import LLMChain  # LangChain 0.x
except Exception:  # noqa: BLE001
    try:
        from langchain_classic.chains import LLMChain  # LangChain 1.x
    except Exception:  # noqa: BLE001
        LLMChain = None


class _FallbackChain:
    """Tiny stand-in with the same .invoke() contract if LLMChain is unavailable."""

    def __init__(self, prompt, llm):
        self._chain = prompt | llm | StrOutputParser()

    def invoke(self, inputs: dict) -> dict:
        return {"text": self._chain.invoke(inputs)}


def build_llm(api_key: str, model: str, temperature: float = 0.2, json_mode: bool = False) -> ChatOpenAI:
    """Create the ChatOpenAI model. json_mode forces the reply to be a JSON object."""
    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, timeout=60, max_retries=1, **kwargs
    )


def build_assessment_chain(llm: ChatOpenAI):
    """Reusable chain: ChatPromptTemplate (System + Human) -> LLM."""
    prompt = ASSESSMENT_CHAT_TEMPLATE.partial(json_schema=JSON_SCHEMA)
    if LLMChain is not None:
        return LLMChain(llm=llm, prompt=prompt)
    return _FallbackChain(prompt, llm)


def run_assessment(chain, prompt_inputs: dict) -> str:
    """Run the chain and return the raw model text (should be JSON)."""
    result = chain.invoke(prompt_inputs)
    return result["text"] if isinstance(result, dict) else str(result)


def stream_narrative(llm: ChatOpenAI, prompt_inputs: dict, assessment_json: str):
    """Generator that yields the narrative in small chunks (for st.write_stream)."""
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(
        **prompt_inputs, assessment_json=assessment_json
    )
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def demo_message_conversation(llm: ChatOpenAI, profile_text: str):
    """Show SystemMessage / HumanMessage / AIMessage working together."""
    messages = [
        SystemMessage(content=DEMO_SYSTEM),
        HumanMessage(content=profile_text),
        AIMessage(content=DEMO_PREVIOUS_AI_REPLY),  # an earlier assistant turn
        HumanMessage(content=DEMO_FOLLOW_UP),
    ]
    reply = llm.invoke(messages)
    return messages, reply.content


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Cheap check that the key works (lists models, no tokens used)."""
    from openai import OpenAI

    try:
        OpenAI(api_key=api_key).models.list()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, friendly_error(exc)


def friendly_error(exc: Exception) -> str:
    """Turn API exceptions into short messages a patient/student can understand."""
    text = str(exc)
    low = text.lower()
    if "401" in text or "invalid_api_key" in low or "incorrect api key" in low:
        return "The API key was rejected. Please check it and try again."
    if "429" in text or "quota" in low or "rate limit" in low:
        return "OpenAI rate limit or quota reached. Wait a moment or check your billing."
    if "timeout" in low or "connection" in low:
        return "Could not reach OpenAI. Check your internet connection and try again."
    return f"Unexpected error: {text[:200]}"


def benchmark_cache(api_key: str, model: str) -> list[dict]:
    """Call the same tiny prompt twice under each cache type and time both calls."""
    prompt = "In one short sentence, why is drinking water important when you have a fever?"
    rows = []
    for mode in (cm.CACHE_MEMORY, cm.CACHE_SQLITE):
        cm.setup_cache(mode)
        cm.clear_cache(mode)
        llm = build_llm(api_key, model, temperature=0.0)
        t1 = time.perf_counter()
        llm.invoke(prompt)
        first = time.perf_counter() - t1
        t2 = time.perf_counter()
        llm.invoke(prompt)
        second = time.perf_counter() - t2
        rows.append({
            "Cache": mode,
            "1st call (s)": round(first, 2),
            "2nd call (s)": round(second, 3),
            "Speed-up": f"{first / max(second, 0.001):.0f}x",
        })
    return rows
