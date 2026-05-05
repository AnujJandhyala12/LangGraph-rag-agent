from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.graph import build_graph
from app.tools import credit_risk_tool
import re

app = FastAPI(title="Credit Risk RAG Agent")
graph = build_graph()


class QueryRequest(BaseModel):
    query: str


class ApplicantRequest(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float


def clean_response(text: str) -> str:
    text = text.replace("\\n", "\n")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"#{1,3} ", "", text)
    return text.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")              # ✅ for text questions
async def query_handler(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = graph.invoke({"query": request.query})
        if result is None:
            raise HTTPException(status_code=500, detail="Graph returned None")
        return {
            "query": request.query,
            "response": clean_response(result.get("response", "No response")),  # ✅ must be here
}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")            # ✅ for loan number predictions
def predict(request: ApplicantRequest):
    try:
        applicant_dict = {
            "loan_amnt": request.loan_amnt,
            "int_rate": request.int_rate,
            "annual_inc": request.annual_inc,
            "dti": request.dti,
        }
        result = credit_risk_tool(applicant_dict)
        return {"result": clean_response(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))