"""Model configuration. Single place where the LLM and embeddings are created.

Every agent (investigator, planner, verifier) will call get_llm(),
and the indexer/search call get_embeddings(). Changing a model later
means editing only .env (or this file).
"""
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()  # reads .env from the project root

DEFAULT_MODEL = "openai/gpt-oss-120b"  # override via GROQ_MODEL in .env
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"  # override via GOOGLE_EMBEDDING_MODEL


def _require_key(name: str) -> str:
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise RuntimeError(f"{name} missing. Put your real key in .env (see .env.example).")
    return value


def get_llm(temperature: float = 0.0, max_tokens: int = 2048) -> ChatGroq:
    """Return a configured Groq chat model.

    temperature=0 -> deterministic, best for factual/diagnostic work.
    """
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        api_key=_require_key("GROQ_API_KEY"),
        max_tokens=max_tokens,  # caps the reply; also lowers the tokens "requested" per call (TPM limit)
        max_retries=6,          # on a 429 the client waits and retries automatically
    )


def get_structured_llm(schema, temperature: float = 0.0):
    """LLM that must return an object matching a Pydantic `schema`.

    Agents use this so their output is validated data, not free text.
    """
    llm = get_llm(temperature=temperature, max_tokens=3000)
    return llm.with_structured_output(schema, method="function_calling")


def invoke_structured(schema, messages, attempts: int = 3):
    """Call the LLM for structured output, retrying if the model returns broken JSON.

    LLMs occasionally emit malformed tool-call JSON (Groq: "tool_use_failed").
    At temperature 0 a retry would repeat the same mistake, so each retry
    raises the temperature slightly.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            return get_structured_llm(schema, temperature=0.3 * attempt).invoke(messages)
        except Exception as error:  # noqa: BLE001
            text = str(error)
            if "tool_use_failed" in text or "Failed to parse" in text or type(error).__name__ == "OutputParserException":
                last_error = error
                continue
            raise  # other errors (rate limit, bad key...) are handled elsewhere
    raise RuntimeError(
        f"The model returned malformed structured output {attempts} times. "
        "Try again, or switch GROQ_MODEL in .env."
    ) from last_error


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the Google embedding model (text -> vector of numbers)."""
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        google_api_key=_require_key("GOOGLE_API_KEY"),
    )


def ask_llm(question: str) -> str:
    """Send one plain question and return the text answer (used for testing)."""
    return get_llm().invoke(question).content


def list_available_models() -> list[str]:
    """Return Groq model IDs your key can actually use (names change often)."""
    from groq import Groq

    client = Groq(api_key=_require_key("GROQ_API_KEY"))
    return sorted(m.id for m in client.models.list().data)