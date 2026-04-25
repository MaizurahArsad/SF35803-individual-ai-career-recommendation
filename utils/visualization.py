import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def plot_age_distribution(df, save_path=None):
    plt.figure(figsize=(8,4))
    sns.histplot(df['Age'], bins=15, kde=True, color='skyblue')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path: plt.savefig(os.path.join(save_path,"age_distribution.png"))
    plt.show()

def plot_education_count(df, save_path=None):
    plt.figure(figsize=(6,4))
    sns.countplot(x="Education", data=df, hue="Education", palette="Set2", legend=False)
    plt.title("Education Level Counts")
    plt.xlabel("Education Level")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path: plt.savefig(os.path.join(save_path,"education_count.png"))
    plt.show()

def plot_recommended_career_distribution(df, save_path=None):
    plt.figure(figsize=(10,6))
    order = df["Recommended_Career"].value_counts().index
    sns.countplot(y="Recommended_Career", data=df, order=order, hue="Recommended_Career", palette="Set3", legend=False)
    plt.title("Recommended Career Distribution")
    plt.xlabel("Count")
    plt.ylabel("Recommended Career")
    plt.tight_layout()
    if save_path: plt.savefig(os.path.join(save_path,"recommended_career_distribution.png"))
    plt.show()

def plot_recommendation_score_distribution(df, save_path=None):
    plt.figure(figsize=(8,4))
    sns.histplot(df['Recommendation_Score'], bins=10, kde=True, color='olive')
    plt.title("Recommendation Score Distribution")
    plt.xlabel("Recommendation Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path: plt.savefig(os.path.join(save_path,"recommendation_score_distribution.png"))
    plt.show()

def count_items(series):
    items = series.dropna().apply(lambda x: x.split(";"))
    flat_list = [item.strip() for sublist in items for item in sublist]
    return pd.Series(flat_list).value_counts()

def plot_skill_frequency(df, save_path=None):
    skills_count = count_items(df["Skills"])
    plt.figure(figsize=(12,8))
    sns.barplot(x=skills_count.values, y=skills_count.index, palette="viridis")
    plt.title("Skill Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Skill")
    plt.tight_layout()
    if save_path: plt.savefig(os.path.join(save_path,"skill_frequency.png"))
    plt.show()

def plot_interest_frequency(df, save_path=None):
    interests_count = count_items(df["Interests"])
    plt.figure(figsize=(12,8))
    sns.barplot(x=interests_count.values, y=interests_count.index, palette="magma")
    plt.title("Interest Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Interest")
    plt.tight_layout()
    if save_path: plt.savefig(os.path.join(save_path,"interest_frequency.png"))
    plt.show()