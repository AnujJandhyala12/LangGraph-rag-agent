import time
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from dotenv import load_dotenv
load_dotenv()

from app.rag import create_retriever
from functools import lru_cache

@lru_cache(maxsize=1)
def get_retriever():
    return create_retriever()

retriever = get_retriever()  # ✅ cached — won't reload on every request
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ API key from .env — never hardcode
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.3,
    max_retries=3,
)

parser = StrOutputParser()
retriever = create_retriever()

# Simple in-memory cache
cache = {}


def _invoke_with_retry(chain, inputs: dict, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            time.sleep(3)  # small buffer before every call
            return chain.invoke(inputs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s, 150s
                logger.warning(f"Rate limited. Waiting {wait}s... retry {attempt+1}/{retries}")
                time.sleep(wait)
            else:
                raise e  # non-rate-limit errors should crash immediately
    raise Exception("Rate limit exceeded after all retries. Wait a few minutes and try again.")


def rag_tool(query: str) -> str:
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer only using the context below.\n\nContext:\n{context}"),
            ("human", "{query}"),
        ])

        chain = prompt | llm | parser
        return _invoke_with_retry(chain, {"context": context, "query": query})

    except Exception as e:
        logger.error(f"RAG error: {e}")
        return f"RAG Error: {str(e)}"


def summarize_tool(query: str) -> str:
    try:
        # ✅ Fetch document chunks first, then summarize
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Summarize the following document content concisely.\n\nDocument:\n{context}"),
            ("human", "{query}"),
        ])

        chain = prompt | llm | parser
        return _invoke_with_retry(chain, {"context": context, "query": query})

    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return f"Summarization Error: {str(e)}"


def direct_tool(query: str) -> str:
    try:
        # ✅ Return cached response if available
        if query in cache:
            logger.info("Cache hit for query.")
            return cache[query]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer clearly and directly."),
            ("human", "{query}"),
        ])

        chain = prompt | llm | parser
        result = _invoke_with_retry(chain, {"query": query})

        cache[query] = result
        return result

    except Exception as e:
        logger.error(f"Direct LLM error: {e}")
        return f"LLM Error: {str(e)}"