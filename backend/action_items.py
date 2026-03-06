import re

def extract_action_items(text: str):
    # Simple regex-based extraction for demo; replace with ML for production
    lines = text.split('\n')
    action_items = [line for line in lines if re.search(r'\b(action|todo|task)\b', line, re.I)]
    return action_items
