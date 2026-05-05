import os
import time
import logging
import joblib
import shap
import pandas as pd
from functools import lru_cache
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.rag import create_retriever

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.3,
    max_retries=3,
)

parser = StrOutputParser()

# ✅ Cached retriever — won't reload on every request
@lru_cache(maxsize=1)
def get_retriever():
    return create_retriever()

retriever = get_retriever()

# ✅ Load ML model — correct path
try:
    model = joblib.load("ml/model.pkl")
    FEATURES = joblib.load("ml/features.pkl")
    logger.info(f"ML model loaded. Features: {FEATURES}")
except Exception as e:
    model = None
    FEATURES = []
    logger.warning(f"ML model not found. Run: python -m ml.train. Error: {e}")

# Simple cache
cache = {}


def _invoke_with_retry(chain, inputs: dict, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            time.sleep(3)
            return chain.invoke(inputs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait}s... retry {attempt+1}/{retries}")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Rate limit exceeded after all retries.")


# ── Credit Risk Tool (with SHAP) ───────────────────────────
def credit_risk_tool(applicant_dict: dict) -> str:
    try:
        if model is None:
            return "ML model not loaded. Run: python -m ml.train"

        df = pd.DataFrame([applicant_dict])

        prob = model.predict_proba(df)[0][1]
        label = "HIGH RISK" if prob > 0.5 else "LOW RISK"

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(df)[0]
        top_features = sorted(
            zip(df.columns, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        shap_summary = ", ".join(
            f"{f} ({'↑' if v > 0 else '↓'}{abs(v):.3f})"
            for f, v in top_features
        )

        raw_result = f"Risk: {label} | Probability: {prob:.1%} | Key factors: {shap_summary}"

        # Ask Claude to explain in plain English
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a credit risk officer explaining a model decision to a customer.
Explain this result clearly in under 150 words. Mention the key risk factors and what they mean.

Result: {raw_result}"""),
            ("human", "Explain this credit risk assessment."),
        ])

        chain = prompt | llm | parser
        return _invoke_with_retry(chain, {"raw_result": raw_result})

    except Exception as e:
        logger.error(f"Credit risk tool error: {e}")
        return f"Prediction Error: {str(e)}"


# ── RAG Tool ───────────────────────────────────────────────
def rag_tool(query: str) -> str:
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a credit risk expert. Answer using only the context below.\n\nContext:\n{context}"),
            ("human", "{query}"),
        ])

        chain = prompt | llm | parser
        return _invoke_with_retry(chain, {"context": context, "query": query})

    except Exception as e:
        logger.error(f"RAG error: {e}")
        return f"RAG Error: {str(e)}"


# ── Summarize Tool ─────────────────────────────────────────
def summarize_tool(query: str) -> str:
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial analyst. Summarize the credit risk report below concisely.\n\nReport:\n{context}"),
            ("human", "{query}"),
        ])

        chain = prompt | llm | parser
        return _invoke_with_retry(chain, {"context": context, "query": query})

    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return f"Summarization Error: {str(e)}"


# ── Direct Tool ────────────────────────────────────────────
def direct_tool(query: str) -> str:
    try:
        if query in cache:
            logger.info("Cache hit.")
            return cache[query]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful credit risk AI assistant. Answer clearly and directly."),
            ("human", "{query}"),
        ])

        chain = prompt | llm | parser
        result = _invoke_with_retry(chain, {"query": query})
        cache[query] = result
        return result

    except Exception as e:
        logger.error(f"Direct error: {e}")
        return f"Error: {str(e)}"