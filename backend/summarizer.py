from transformers import pipeline

def summarize_text(text: str) -> str:
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(text, max_length=100, min_length=20, do_sample=False)
    return summary[0]['summary_text']
