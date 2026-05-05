from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.tools import rag_tool, summarize_tool, direct_tool

class AgentState(TypedDict):
    query: str
    response: str

def start_node(state: AgentState):
    return state

def route_query(state: AgentState):
    query = state.get("query", "").lower()

    if "summarize" in query or "summarise" in query:  # ✅ both spellings
        return "summarize"
    elif "document" in query or "report" in query:
        return "rag"
    else:
        return "direct"

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

    graph.set_entry_point("start")

    graph.add_conditional_edges(
        "start",
        route_query,
        {
            "rag": "rag",
            "summarize": "summarize",
            "direct": "direct",
        },
    )

    graph.add_edge("rag", END)
    graph.add_edge("summarize", END)
    graph.add_edge("direct", END)

    return graph.compile()  