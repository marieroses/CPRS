#%%
#Import libraries
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
print("Libraries imported")

#%%
#Import data to create random student profiles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_final_careers():
    path = os.path.join(BASE_DIR, '..', 'datasets', 'final_careers.csv')
    return pd.read_csv(path)

def load_initial_skills():
    path = os.path.join(BASE_DIR, '..', 'datasets', 'renamed_careers.csv') 
    return pd.read_csv(path)

final_careers = load_final_careers()
renamed_careers = load_initial_skills()
skills_embedding = np.load(os.path.join(BASE_DIR, '..', 'datasets', 'final_career_embeddings.npy'))

print("Data imported")

#%%
#Creating test student profiles
np.random.seed(20)

student_skills=renamed_careers["Skills"].unique()
student_domains=final_careers["Domain of Interest"].unique()

def create_profiles():
    skills = np.random.choice(student_skills, 7, replace=False)
    domain = np.random.choice(student_domains)
    return {'Skills': skills, 'Domain of Interest': domain}

student_profiles = [create_profiles() for _ in range(100)]
students = pd.DataFrame(student_profiles)

print(students.shape)
#%%
#Run model for further use
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model run succesfully")

#%%
#Version A

#Recommender test function
def Frontend_A_test(domain, skills, recommended_career, embeddings):
    filtered_careers = recommended_career[recommended_career["Domain of Interest"] == domain]
    filtered_indices = filtered_careers.index
    filtered_embeddings = embeddings[filtered_indices]

    skills_string = " ".join(skills)
    user_skills_emb = model.encode([skills_string])

    similarities = cosine_similarity(user_skills_emb, filtered_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:3]

    recommendations = []
    for index in top_indices:
        career = filtered_careers.iloc[index]["Career Path"]
        score = similarities[index] * 100
        recommendations.append((career, score))

    return recommendations

#Overall similarity scores
all_scores = []

for _, row in students.iterrows():
    output = Frontend_A_test(row["Domain of Interest"],row["Skills"],final_careers,skills_embedding)

    for _, score in output:
        all_scores.append(score)

overall_avg_A = sum(all_scores) / len(all_scores)
print(f"Overall average similarity score (Frontend A): {overall_avg_A:.2f}%")

#Similarity scores by domain
domain_similarity_scores = defaultdict(list)

for _, row in students.iterrows():
    domain = row["Domain of Interest"]
    skills = row["Skills"]
    output = Frontend_A_test(domain, skills, final_careers, skills_embedding)

    for _, score in output:
        domain_similarity_scores[domain].append(score)

df_avg = pd.DataFrame([
    {"Domain": domain, "Average Score": sum(scores) / len(scores)}
    for domain, scores in domain_similarity_scores.items() if scores])

df_avg_sorted = df_avg.sort_values(by="Average Score", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_avg_sorted, x="Domain", y="Average Score")
plt.title("Average Similarity Score by Domain (Frontend A)", fontsize=14)
plt.ylabel("Average Similarity Score (%)")
plt.xlabel("Domain of Interest")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

#%%
#Version B

#Recommender test function
def Frontend_B_test(skills, recommended_careers, embeddings):
    skills_string = " ".join(skills)
    user_skills_emb = model.encode([skills_string])
    similarities = cosine_similarity(user_skills_emb, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:3]

    recommendations = []
    for index in top_indices:
        career = recommended_careers.iloc[index]["Career Path"]
        score = similarities[index] * 100
        recommendations.append((career, score))
    return recommendations

#Overall similarity scores
total_scores = []

for i, row in students.iterrows():
    output = Frontend_B_test(row["Skills"], final_careers, skills_embedding)
    
    for _, score in output:
        total_scores.append(score)
    
overall_avg_B = sum(total_scores) / len(total_scores)
print(f"Overall average similarity score (Frontend B): {overall_avg_B:.2f}%")

#Similarity scores distribution for all 100 test profiles 
average_scores_per_student = []

for _, row in students.iterrows():
    skills = row["Skills"]
    output = Frontend_B_test(skills, final_careers, skills_embedding)
    scores = [score for _, score in output]
    avg_score = sum(scores) / len(scores)
    average_scores_per_student.append(avg_score)

df_avg = pd.DataFrame({"Average Similarity Score": average_scores_per_student})

plt.figure(figsize=(10, 6))
sns.histplot(df_avg["Average Similarity Score"], bins=20, kde=True, color="#2a9d8f")
plt.title("Average Similarity Score per Student (Frontend B)", fontsize=14)
plt.xlabel("Average Similarity Score (%)")
plt.ylabel("Number of Students")
plt.grid(True)
plt.tight_layout()
plt.show()