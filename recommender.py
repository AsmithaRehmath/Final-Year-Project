from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def recommend_jobs(user_skills, jobs_df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(jobs_df['skills'])
    user_vec = tfidf.transform([user_skills])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    jobs_df['score'] = similarity[0]
    return jobs_df.sort_values(by='score', ascending=False)
