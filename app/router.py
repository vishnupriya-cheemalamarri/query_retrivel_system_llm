from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
from .parser import parse_document
from .embedder import chunk_text, embed_chunks
from .retriever import retrieve_clauses
from .llm import generate_answer

router = APIRouter()

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@router.post("/hackrx/run", response_model=RunResponse)
async def run_submission(payload: RunRequest, request: Request):
    try:
        raw_text = parse_document(payload.documents)
        chunks = chunk_text(raw_text)
        embed_chunks(chunks)

        matches = retrieve_clauses(payload.questions)
        answers = []

        for q in payload.questions:
            context = matches.get(q, [])
            answer = generate_answer(q, context)
            answers.append(answer)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
