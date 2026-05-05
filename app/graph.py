from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.tools import rag_tool, summarize_tool, direct_tool,credit_risk_tool


class AgentState(TypedDict):
    query: str
    response: str

def start_node(state: AgentState):
    return state

def route_query(state: AgentState):
    query = state.get("query", "").lower()

    if "summarize" in query or "summarise" in query:
        return "summarize"
    elif any(word in query for word in ["document", "report", "dataset", "data"]):
        return "rag"
    elif any(word in query for word in ["default", "rate", "dti", "risk", "loan", "credit", "fico", "income"]):
        return "rag"  # ✅ credit risk questions → RAG
    else:
        return "direct"
    
def credit_risk_node(state: AgentState):
    # expects state["query"] to be a dict of applicant features
    result = credit_risk_tool(state["query"])
    return {"query": state["query"], "response": result}

def rag_node(state: AgentState):
    result = rag_tool(state["query"])
    return {"query": state["query"], "response": result}

def summarize_node(state: AgentState):
    result = summarize_tool(state["query"])
    return {"query": state["query"], "response": result}

def direct_node(state: AgentState):
    result = direct_tool(state["query"])
    return {"query": state["query"], "response": result}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("start", start_node)
    graph.add_node("rag", rag_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("direct", direct_node)
    graph.add_node("credit_risk", credit_risk_node)  # ← new

    graph.set_entry_point("start")
    graph.add_conditional_edges(
        "start",
        route_query,
        {
            "rag": "rag",
            "summarize": "summarize",
            "direct": "direct",
            "credit_risk": "credit_risk",  # ← new
        },
    )
    graph.add_edge("rag", END)
    graph.add_edge("summarize", END)
    graph.add_edge("direct", END)
    graph.add_edge("credit_risk", END)  # ← new
    return graph.compile()