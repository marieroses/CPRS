#%%
#Install packages and import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import seaborn as sns
import matplotlib.pyplot as plt
print('Imported')

#%%
#Load data and data preprocessing

#Read the 3 ESCO CSV files containing occupations, skills and their connections
initial_occupations=pd.read_csv("../datasets/occupations_en.csv")
initial_skills=pd.read_csv("../datasets/skills_en.csv")
initial_relations=pd.read_csv("../datasets/occupationSkillRelations_en.csv")
print("initial_occupations shape:" + str(initial_occupations.shape))
print("initial_occupations columns: " + str(initial_occupations.columns))
print("initial_skills shape:" + str(initial_skills.shape))
print("initial_skills columns: " + str(initial_skills.columns))
print("initial_relations shape:" + str(initial_relations.shape))
print("initial_relations columns:" + str(initial_relations.columns))

#Select relevant columns for the 3 dataframes
selected_occupations=initial_occupations[["preferredLabel","description","conceptUri"]]
selected_skills=initial_skills[["conceptUri","preferredLabel"]]
selected_relations=initial_relations[["occupationUri","skillUri"]]
print("selected_occupations columns: " + str(selected_occupations.columns))
print("selected_skills columns: " + str(selected_skills.columns))
print("selected_relations columns:" + str(selected_relations.columns))

#Merge the 3 datasets into 1 single dataset (merged_careers)
relations_with_occupations=selected_relations.merge(selected_occupations,how="outer",left_on="occupationUri", right_on="conceptUri")
merged_careers=relations_with_occupations.merge(selected_skills,how="left",left_on="skillUri", right_on="conceptUri")
print("merged_careers shape: " + str(merged_careers.shape))
print("merged_careers columns: " + str(merged_careers.columns))

#Select relevant columns from merged_careers and rename for transparency (renamed_careers)
renamed_careers=merged_careers[["preferredLabel_x","description","preferredLabel_y"]]
renamed_careers=renamed_careers.rename(columns={"preferredLabel_x":"Career Path","description":"Description","preferredLabel_y":"Skills"})
print("renamed_careers shape:" + str(renamed_careers.shape))
print("renamed_careers columns:" + str(renamed_careers.columns))

#Capitalize first letters for career paths
renamed_careers["Career Path"]=renamed_careers["Career Path"].str.title()

#Export renamed_careers for further use to match selected skills with job skills
renamed_careers.to_csv("../datasets/renamed_careers.csv", index=False, encoding='utf-8')

#Investigate distribution of number of skills related to each occupation
skills_nr_before = renamed_careers["Career Path"].value_counts()
print("Min before capping:" + str(skills_nr_before.min()))
print("Max before capping:" + str(skills_nr_before.max()))

fig, ax = plt.subplots()
ax.boxplot(skills_nr_before)
ax.set_xticks([])
ax.set_ylabel("Number of skills per job")
ax.set_title("Distribution of skills per job before capping")
plt.show()

#Capping the number of related skills for each occupation at Q3 (80)
def limit_skills(skills_list):
    return skills_list.head(80)

capped_careers= renamed_careers.groupby("Career Path").apply(limit_skills).reset_index(drop=True)
print("capped_careers shape:" + str(capped_careers.shape))
print("capped_careers columns:" + str(capped_careers.columns))

#Double-checking distribution of number of skills related to each occupation after capping (capped_careers)
skills_nr_after = capped_careers["Career Path"].value_counts()
print("Min after capping:" + str(skills_nr_after.min()))
print("Max after capping:" + str(skills_nr_after.max()))

fig, ax = plt.subplots()
ax.boxplot(skills_nr_after)
ax.set_xticks([])
ax.set_ylabel("Number of skills per job")
ax.set_title("Distribution of skills per job after capping")
plt.show()

#Grouping capped_careers in order to align all related skills with the corresponding occupation (grouped_careers)
grouped_careers=capped_careers.groupby('Career Path')["Skills"].apply(lambda related_skills: ' '.join(related_skills)).reset_index()
print("grouped_careers shape:" + str(grouped_careers.shape))
print("grouped_careers columns:" + str(grouped_careers.columns))

#Adding the description to the dataset (final_careers)
descriptions = renamed_careers[["Career Path", "Description"]].drop_duplicates(subset="Career Path")
final_careers= grouped_careers.merge(descriptions, on="Career Path", how="left")
print("final_careers shape:" + str(final_careers.shape))
print("final_careers columns:" + str(final_careers.columns))

#Checking for missing values
print(final_careers.isna().sum())

#%%
#Domain of interest computing using TF-IDF + K-Means

#Add a column to final_skills with concatenated skills in a single word for TfidfVectorizer usage
final_careers["Concatenated Skills"]=final_careers['Skills'].str.replace(" ","",regex=False)
print("final_careers shape:" + str(final_careers.shape))
print("final_careers columns:" + str(final_careers.columns))

#Vectorizing the concatenated skills using TfidfVectorizer
vectorizer = TfidfVectorizer()
skills_vector = vectorizer.fit_transform(final_careers['Concatenated Skills'])
print("Skills vector matrix:" + str(skills_vector.shape))

#Applying PCA for feature reduction
reduced_skills_vector = PCA(n_components=0.95).fit_transform(skills_vector.toarray())
print("Skills reduced vector matrix:" + str(reduced_skills_vector.shape))

#Applying K-Means for different cluster sizes and plotting result to represent datapoints distribution within the clusters
cluster_sizes = [10, 15, 20, 25, 30, 35]
fitted_KMeans  = {}
cluster_numbers= {}

for n in cluster_sizes:
    km = KMeans(n_clusters=n, random_state=10)
    y = km.fit_predict(reduced_skills_vector)
    fitted_KMeans[n] = km
    cluster_numbers[n] = y

    col = f"Cluster K-{n}-TFIDF"
    final_careers[col] = y.astype(int)

    plt.figure()
    sns.countplot(data=final_careers, x=col, order=sorted(final_careers[col].unique()), palette="mako")
    plt.title(f"Number of Occupations per Cluster (n = {n})")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Occupations")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

#%%
#Domain of interest computing using Sentence Transformer + K-Means

#Vectorizing skills using SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
skills_embedding = model.encode(final_careers["Skills"])
print("Skills embedding matrix:" + str(skills_embedding.shape))

#Saving matrix to be used in the frontend script to save runtime
np.save("../datasets/final_career_embeddings.npy", skills_embedding)

#Applying PCA for feature reduction
reduced_skills_embedding = PCA(n_components=0.95).fit_transform(skills_embedding)
print("Skills reduced embedding matrix:" + str(reduced_skills_embedding.shape))

#Applying K-Means for different cluster sizes and plotting result to represent datapoints distribution within the clusters
cluster_sizes = [10, 15, 20, 25, 30, 35]
fitted_KMeans  = {}
cluster_numbers= {}

for n in cluster_sizes:
    km = KMeans(n_clusters=n, random_state=10)
    y = km.fit_predict(reduced_skills_embedding)
    fitted_KMeans[n] = km
    cluster_numbers[n] = y

    col = f"Cluster K-{n}-EMBED"
    final_careers[col] = y.astype(int)

    plt.figure()
    sns.countplot(data=final_careers, x=col, order=sorted(final_careers[col].unique()), palette="mako")
    plt.title(f"Number of Occupations per Cluster (n = {n})")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Occupations")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

#Extract cluster skills for K=15 to surface domain of interest for each cluster
labels= final_careers["Cluster K-15-EMBED"]

for cluster_id in range(15):
    print("Cluster " + str(cluster_id))
    cluster_skills = final_careers[labels== cluster_id]

    for i, (_, row) in enumerate(cluster_skills.head().iterrows()):
        print(f"{i+1}. {row['Skills'][:200]}")

#Assign domains to each cluster based on extracted skills using chat GPT
domains = {
    0: "Environmental Management & Urban Planning",
    1: "Special Education & Instructional Technology",
    2: "Mechanical and Industrial Engineering",
    3: "Accounting, Finance & Public Administration",
    4: "Industrial Maintenance & Rubber Manufacturing",
    5: "Footwear, Textile & Leather Manufacturing",
    6: "Veterinary & Animal Biosciences",
    7: "Healthcare, Social Work & Alternative Medicine",
    8: "Aviation, Military & Outdoor Advertising",
    9: "Software Development & Digital Technologies",
    10: "Sales, Marketing & Customer Relations",
    11: "Digital Media, Advocacy & Strategic Communication",
    12: "Sustainable Engineering & Automotive Technologies",
    13: "Animation, Multimedia & Performing Arts",
    14: "Logistics, Supply Chain & Maritime Operations"
}
print("Dictionnary created")

#Map domain labels to their respective clusters using the domains dictionary
final_careers['Domain of Interest'] = final_careers['Cluster K-15-EMBED'].map(domains)
final_careers[['Career Path','Domain of Interest']][1000:1051]

#%%
#Export final dataset

#Double-checking final dataset before export
print("final_careers shape:" + str(final_careers.shape))
print("final_careers columns:" + str(final_careers.columns))

#Export final_careers for future use for streamlit
final_careers.to_csv("../datasets/final_careers.csv", index=False, encoding='utf-8')