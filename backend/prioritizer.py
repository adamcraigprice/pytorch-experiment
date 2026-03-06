from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prioritize_tasks(tasks):
    # Simple prioritization: sort by length (shorter = higher priority)
    return sorted(tasks, key=len)
