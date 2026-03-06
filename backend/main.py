from fastapi import FastAPI, UploadFile, File
from backend.summarizer import summarize_text
from backend.action_items import extract_action_items
from backend.prioritizer import prioritize_tasks
from backend.vector_search import search_similar

app = FastAPI()

@app.post("/summarize/")
async def summarize_meeting(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8")
    summary = summarize_text(text)
    return {"summary": summary}

@app.post("/action-items/")
async def get_action_items(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8")
    items = extract_action_items(text)
    return {"action_items": items}

@app.post("/prioritize/")
async def prioritize(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8")
    items = extract_action_items(text)
    priorities = prioritize_tasks(items)
    return {"prioritized_tasks": priorities}

@app.post("/search/")
async def vector_search(query: str):
    results = search_similar(query)
    return {"results": results}
