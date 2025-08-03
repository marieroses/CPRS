# Import libraries
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict

#%% Load datasets
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

#%% 
#User interface setup

#Setting up Streamlit and app style
st.set_page_config(page_title="Career Path Recommender System", layout='wide')
st.title("Career Path Recommender System  (Version B)")
st.markdown("""<style> body {background-color: #e6f2ff;}.stApp {background-color: #e6f2ff;}</style>""", unsafe_allow_html=True)

##Defining skills picklist for user input
skills = renamed_careers["Skills"].unique()
selected_skills = st.multiselect("Select up to 7 skills", options=skills, max_selections=7)

#Build career-to-skill mapping
career_skill_map = defaultdict(set)
for _, row in renamed_careers.iterrows():
    career = row["Career Path"].strip().lower()
    skill = row["Skills"].strip().lower()
    if career and skill:
        career_skill_map[career].add(skill)


model = SentenceTransformer("all-MiniLM-L6-v2")

#If condition to handle exception when no skills are entered
if selected_skills:    
    #Vectorize user-selected skills
    model = SentenceTransformer("all-MiniLM-L6-v2")
    selected_skills_text = " ".join(selected_skills)
    selected_skills_embedding = model.encode([selected_skills_text])

    #Compute cosine similarity between user skills and all careers
    similarities = cosine_similarity(selected_skills_embedding, skills_embedding)[0]

    #Compute cosine similarity between user skills and all careers
    similarities = cosine_similarity(selected_skills_embedding, skills_embedding)[0]

    #Top 3 most similar careers
    top_indices = similarities.argsort()[::-1][:3]
    top_careers = final_careers.iloc[top_indices]
    top_scores = similarities[top_indices]

   #Display calculated recommendations and scores to the user
    st.markdown("<u><h2 style='color:#004080;font-size:25px'>Recommended Career Paths:</h2></u>", unsafe_allow_html=True)

    for i, (index, score) in enumerate(zip(top_careers.index, top_scores)):
        career_title = top_careers.loc[index, 'Career Path']
        description = top_careers.loc[index, 'Description']

        #Normalize the career title to use as a key for skill mapping
        career_key = career_title.strip().lower()

        #Retrieve the set of skills associated with the current career from the skill map 
        job_skills_set = career_skill_map.get(career_key, set())

        #Convert user's selected skills to a lowercase set for case-insensitive comparison
        user_skills_set = set(s.strip().lower() for s in selected_skills)

        #Find which of the user's skills match the skills required for this career
        matched_skills = user_skills_set.intersection(job_skills_set)

        #Compute number of selected skills matching with the career skills
        num_matched = len(matched_skills)
        total = len(user_skills_set)

        #Extract 10 first skills for each recommended career and convert to a dataframe for display
        career_skills_df = renamed_careers[renamed_careers['Career Path'] == career_title]
        top_10_skills = career_skills_df['Skills'].head(10).tolist()
        top_10_skills_cap = [skill.capitalize() for skill in top_10_skills]
        top_10_skills_df = pd.DataFrame(top_10_skills_cap, columns=["Top Skills"])
        top_10_skills_df.index = range(1, len(top_10_skills_df) + 1)

        #Display information to the user
        st.markdown(f"<div style='color:#004080; font-size:20px; font-weight:bold'>{i+1}. {career_title} (Similarity score : {score*100:.2f}% )</div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown("<b>Career description:</b>", unsafe_allow_html=True)
        st.markdown(f"{description}")
        st.markdown("<b>This career path was suggested because you entered following skills:</b>", unsafe_allow_html=True)
        st.markdown(f"{', '.join(matched_skills)} ({num_matched} out of {total} entered skills match this career profile)")
        st.markdown(f"<b>Top skills for {career_title}:</b>", unsafe_allow_html=True)
        st.dataframe(top_10_skills_df, use_container_width=True)
        st.markdown("")

    #Career portal link
    st.markdown(":violet[Curious about how to pursue one of these careers? Have a look at the Irish Career Portal for real-world course options and guidance.]")
    st.link_button("Access here", "https://careersportal.ie/")

else:
    st.markdown(":red[Please select up to 7 skills to get career recommendations.]")
