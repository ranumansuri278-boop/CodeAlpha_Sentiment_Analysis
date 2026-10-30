import pandas as pd
import os
import glob

# Find CSV file inside data folder
csv_files = glob.glob("data/*.csv")

if not csv_files:
    print("Error: CSV file not found in data folder!")
else:
    file_path = csv_files[0]

    # Load dataset
    df = pd.read_csv(file_path, encoding="latin1")

    # Display dataset information
    print("Dataset loaded successfully!")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

# Sentiment Analysis using VADER

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Remove rows where review text is missing
df = df.dropna(subset=["Review_text"]).copy()

# Convert reviews into string
df["Review_Text"] = df["Review_text"].astype(str)

# Function to classify sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review_Text"].apply(get_sentiment)

# Display results
print("\nSentiment Analysis Results:")
print(df[["Review_Text", "Sentiment"]].head(10))

# Count each sentiment
print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# Save sentiment analysis outputs

import os
import matplotlib.pyplot as plt

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

# 1. Save analysed CSV file
df.to_csv("outputs/sentiment_results.csv", index=False)

print("CSV file saved successfully!")

# 2. Create and save Bar Chart
plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.title("Customer Review Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/sentiment_bar_chart.png", dpi=300)
plt.close()

# 3. Create and save Pie Chart
plt.figure(figsize=(7, 7))

sentiment_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sentiment Distribution")
plt.ylabel("")
plt.tight_layout()

plt.savefig("outputs/sentiment_pie_chart.png", dpi=300)
plt.close()

print("csv file saved successfully!")
print("Bar chart saved successfully!")
print("Pie chart saved successfully!")
print("All output files created successfully!")