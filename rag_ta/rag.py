"""Step 5: retrieval-augmented answering.

1. Embed the student's question and fetch the top-k most similar chunks.
2. Put those chunks, with their timestamps, into the prompt.
3. Ask the LLM to answer ONLY from those chunks, cite timestamps, and reply in
   the student's language. If the chunks don't contain the answer, it must say so.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .chunker import fmt_ts
from .config import settings
from .store import LectureStore

SYSTEM_PROMPT = """You are a helpful teaching assistant for a set of recorded lectures.

Rules:
1. Answer ONLY using the lecture excerpts provided. Do not use outside knowledge.
2. If the excerpts do not contain the answer, say clearly that this wasn't covered
   in the lectures, and suggest what the student could ask instead.
3. After each fact you use, cite where it came from like [Lecture title @ 12:34],
   using the timestamps shown on the excerpts.
4. Reply in the same language the student used for the question, even if the
   excerpts are in a different language.
5. Be clear and concise, like a good tutor. Use short paragraphs or bullet points."""


@dataclass
class Source:
    lecture_id: str
    title: str
    start: float
    end: float
    text: str
    score: float

    @property
    def label(self) -> str:
        return f"{self.title} @ {fmt_ts(self.start)}"


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
    "huggingface": "Qwen/Qwen2.5-72B-Instruct",
}


class LLMError(RuntimeError):
    """A problem with the LLM provider (missing key, no credits...), with a readable message."""


def resolve_provider() -> str:
    provider = settings.llm_provider.lower()
    if provider != "auto":
        if provider not in DEFAULT_MODELS:
            raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: auto, {', '.join(DEFAULT_MODELS)}")
        return provider
    for name, key in (("gemini", settings.google_api_key), ("openai", settings.openai_api_key),
                      ("anthropic", settings.anthropic_api_key), ("huggingface", settings.hf_token)):
        if key:
            return name
    raise LLMError("No LLM API key found. Set one of GOOGLE_API_KEY, OPENAI_API_KEY, "
                   "ANTHROPIC_API_KEY or HUGGINGFACEHUB_API_TOKEN in .env (see .env.example).")


@lru_cache(maxsize=1)
def get_llm():
    """Build the chat model for whichever provider is configured.

    All four come from LangChain, so the rest of the code doesn't care which one it is.
    """
    provider = resolve_provider()
    model = settings.llm_model or DEFAULT_MODELS[provider]
    t, n = settings.llm_temperature, settings.llm_max_new_tokens

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=t, max_output_tokens=n,
                                      google_api_key=settings.google_api_key)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=t, max_tokens=n, api_key=settings.openai_api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=t, max_tokens=n, api_key=settings.anthropic_api_key)

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    endpoint = HuggingFaceEndpoint(repo_id=model, task="text-generation", temperature=t,
                                   max_new_tokens=n, huggingfacehub_api_token=settings.hf_token)
    return ChatHuggingFace(llm=endpoint)


def describe_llm() -> str:
    try:
        provider = resolve_provider()
    except LLMError:
        return "no LLM configured"
    return f"{provider}: {settings.llm_model or DEFAULT_MODELS[provider]}"


def _friendly(exc: Exception) -> LLMError:
    msg = str(exc)
    provider = describe_llm()
    if "402" in msg or "Payment Required" in msg or "credits" in msg.lower():
        return LLMError(f"The LLM provider ({provider}) has no credits left. Add credits, or set "
                        "another provider's API key in .env, e.g. a free Gemini key as GOOGLE_API_KEY.")
    if "401" in msg or "403" in msg or "API key" in msg or "api_key" in msg:
        return LLMError(f"The API key for {provider} was rejected. Check the key in .env.")
    if "429" in msg or "rate" in msg.lower() and "limit" in msg.lower():
        return LLMError(f"{provider} is rate-limiting requests. Wait a minute and try again.")
    if "404" in msg or "not found" in msg.lower():
        return LLMError(f"Model not found for {provider}. Set LLM_MODEL in .env to a model your key can use.")
    return LLMError(f"LLM call failed ({provider}): {msg.splitlines()[0][:300]}")


def format_context(sources: list[Source]) -> str:
    blocks = []
    for i, s in enumerate(sources, 1):
        blocks.append(
            f"[Excerpt {i}] {s.title} ({fmt_ts(s.start)} - {fmt_ts(s.end)})\n{s.text}"
        )
    return "\n\n".join(blocks)


def retrieve(store: LectureStore, question: str,
             lecture_ids: Optional[list[str]] = None, k: Optional[int] = None) -> list[Source]:
    hits = store.search(question, k=k, lecture_ids=lecture_ids)
    return [
        Source(
            lecture_id=doc.metadata["lecture_id"],
            title=doc.metadata["title"],
            start=doc.metadata["start"],
            end=doc.metadata["end"],
            text=doc.page_content,
            score=float(score),
        )
        for doc, score in hits
    ]


def answer_question(
    store: LectureStore,
    question: str,
    lecture_ids: Optional[list[str]] = None,
    history: Optional[list[tuple[str, str]]] = None,
) -> Answer:
    """history = previous (question, answer) pairs, so follow-ups like
    'explain that again more simply' make sense."""
    history = history or []

    # for short follow-ups, include the previous question so retrieval has context
    search_query = question
    if history and len(question.split()) < 6:
        search_query = f"{history[-1][0]} {question}"

    sources = retrieve(store, search_query, lecture_ids)
    if not sources:
        return Answer(
            text="No lectures have been added yet, or none match the selected filter. "
                 "Add a lecture first, then ask again."
        )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for past_q, past_a in history[-3:]:  # last 3 turns keep the prompt small
        messages.append(HumanMessage(content=past_q))
        messages.append(AIMessage(content=past_a))
    messages.append(HumanMessage(content=(
        f"Lecture excerpts:\n\n{format_context(sources)}\n\n"
        f"Student's question: {question}"
    )))

    try:
        response = get_llm().invoke(messages)
    except LLMError:
        raise
    except Exception as exc:
        raise _friendly(exc) from exc

    content = response.content
    if isinstance(content, list):  # some providers return a list of content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return Answer(text=content.strip(), sources=sources)
