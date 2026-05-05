from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.graph import build_graph

app = FastAPI(title="AI Doc Agent")
graph = build_graph()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query_handler(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = graph.invoke({"query": request.query})
        if result is None:
            raise HTTPException(status_code=500, detail="Graph returned None")
        return {
            "query": request.query,
            "response": result.get("response", "No response generated"),
        }
    except HTTPException:
        raise  # ✅ don't swallow HTTP errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))