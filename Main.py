import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

app = FastAPI(title="AI RAG Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str
    context_chunks: list[str]

@app.get("/")
def health_check():
    return {"status": "running", "service": "AI RAG Assistant"}

@app.post("/api/generate-answer")
async def generate_answer(req: QueryRequest):
    try:
        if not req.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
            
        system_instruction = (
            "You are a precise technical assistant. Answer the user prompt "
            "strictly based on the provided context chunks."
        )
        context_str = "\n".join(req.context_chunks)
        
        # LLM query orchestration
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {req.prompt}"}
            ],
            temperature=0.2
        )
        
        return {
            "status": "success",
            "answer": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
