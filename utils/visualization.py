import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def plot_age_distribution(df):
    plt.figure(figsize=(8,4))
    sns.histplot(df['Age'], bins=15, kde=True, color="skyblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

def plot_education_count(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="Education", data=df, palette="Set2")
    plt.title("Education Level Counts")
    plt.xlabel("Education Level")
    plt.ylabel("Count")
    plt.show()
