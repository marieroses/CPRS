#%%
#Import libraries
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
#Test App Version A

model = SentenceTransformer("all-MiniLM-L6-v2")

#Define test function
def Frontend_A_test(domain, skills, recommended_career, embeddings):
    
    #Filter out careers and their index which match with the student profile domain
    filtered_careers = recommended_career[recommended_career["Domain of Interest"] == domain]
    filtered_indices = filtered_careers.index

    #Retrieving skill embeddings based on extracted career indices
    filtered_embeddings = embeddings[filtered_indices]

    #Join generated test skills for each student into a single string to be passed through SentenceTransformer
    skills_string = " ".join(skills)
    user_skills_emb = model.encode([skills_string])

    #Calculate cosine similarity between user skills and filtered job skills
    similarities = cosine_similarity(user_skills_emb, filtered_embeddings)[0]

    #Get top 3 highest matching scores and their indices
    top_indices = similarities.argsort()[::-1][:3]

    #Build list of recommendations
    recommendations = []
    for index in top_indices:
        career = filtered_careers.iloc[index]["Career Path"]
        score = similarities[index] * 100
        recommendations.append((career, score))
    return recommendations

#Calculating average similarity score for all the profiles
total_scores = []
total_students = 0

for i, row in students.iterrows():
    output = Frontend_A_test(row["Domain of Interest"], row["Skills"], final_careers, skills_embedding)
    #print(f"Student {i+1} recommendations:")
    
    student_scores = []
    for career, score in output:
        #print(f" - {career} ({score:.2f}%)")
        student_scores.append(score)
    
    # Accumulate scores for average calculation
    if student_scores:  # avoid division by zero
        total_scores.extend(student_scores)
        total_students += 1

# Calculate and print the average similarity score
if total_scores:
    average_score = sum(total_scores) / len(total_scores)
    print(f"\nAverage similarity score across all profiles: {average_score:.2f}%")
else:
    print("No similarity scores found.")

#%%
#Test App Version B

#Define test function
def Frontend_B_test(skills, recommended_careers, embeddings):
    
    #Join generated test skills for each student into a single string to be passed through SentenceTransformer
    skills_string = " ".join(skills)
    user_skills_emb = model.encode([skills_string])

    #Calculate cosine similarity between user skills and filtered job skills
    similarities = cosine_similarity(user_skills_emb, embeddings)[0]

    #Get top 3 highest matching scores and their indices
    top_indices = similarities.argsort()[::-1][:3]

   #Build list of recommendations
    recommendations = []
    for index in top_indices:
        career = recommended_careers.iloc[index]["Career Path"]
        score = similarities[index] * 100
        recommendations.append((career, score))
    return recommendations

#Calculating average similarity score for all the profiles
total_scores = []
total_students = 0

for i, row in students.iterrows():
    output = Frontend_B_test(row["Skills"], final_careers, skills_embedding)
    #print(f"Student {i+1} recommendations:")
    
    student_scores = []
    for career, score in output:
        #print(f" - {career} ({score:.2f}%)")
        student_scores.append(score)
    
    # Accumulate scores for average calculation
    if student_scores:  # avoid division by zero
        total_scores.extend(student_scores)
        total_students += 1

# Calculate and print the average similarity score
if total_scores:
    average_score = sum(total_scores) / len(total_scores)
    print(f"\nAverage similarity score across all profiles: {average_score:.2f}%")
else:
    print("No similarity scores found.")