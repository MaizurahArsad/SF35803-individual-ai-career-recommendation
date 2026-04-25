import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def plot_age_distribution(df, save_path=None):
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Age'], bins=15, kde=True, color="skyblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path + "age_distribution.png")
    plt.show()

def plot_education_count(df, save_path=None):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Education", data=df, palette="Set2")
    plt.title("Education Level Counts")
    plt.xlabel("Education Level")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path + "education_count.png")
    plt.show()

# Add similar functions for skills, interests, recommended career, recommendation score
