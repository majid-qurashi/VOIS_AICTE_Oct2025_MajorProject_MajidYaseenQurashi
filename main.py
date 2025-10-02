
#Majid Qurashi- AICTE Major Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load the Dataset ---
# NOTE: The provided file path is assumed to be 'Netflix Dataset.csv'
df = pd.read_csv('Netflix Dataset.csv')

# --- 2. Data Cleaning and Preprocessing ---

# Rename 'Type' column to 'Genre' for clarity in analysis
df.rename(columns={'Type': 'Genre'}, inplace=True)

# Clean and convert the 'Release_Date' column to extract the 'Year Added'
# Assuming 'Release_Date' is the column that needs to be used for time analysis (Year the show was added).
# We will use the 'Date Added' column for when it was added to Netflix to understand Netflix's strategy over time.
df['Date Added'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df.dropna(subset=['Date Added'], inplace=True)
df['Year Added'] = df['Date Added'].dt.year.astype(int)

# Extract content release year from 'Release_Date' to see production trends (using the same column as a proxy or using a separate `Release_Date` if available, but the CSV structure suggests the column is named 'Release_Date' but contains 'Date Added' format, so we use 'Year Added' for *Netflix strategy* analysis as per the problem scope)
# NOTE: The column is labelled 'Release_Date' in the CSV, but contains date strings (e.g., 'August 14, 2020') which are typically "Date Added to Netflix" not the original content release date.
# A new column 'Content Release Year' is created for a more robust analysis.
df['Content Release Year'] = df['Date Added'].dt.year.astype(int)
# In this specific dataset, the 'Release_Date' column holds the *Date Added* to Netflix, so we'll use 'Year Added' for strategic analysis.

# Drop rows where 'Country' or 'Genre' is missing, as they are crucial for the objectives
df.dropna(subset=['Country', 'Genre'], inplace=True)

# --- 3. Objective 1: Movies vs. TV Shows Distribution Over The Years ---

# Group by 'Year Added' and 'Category' (Movie/TV Show) and count
content_distribution = df.groupby(['Year Added', 'Category']).size().reset_index(name='Count')

# Pivot the table for easy plotting
pivot_distribution = content_distribution.pivot(index='Year Added', columns='Category', values='Count').fillna(0)
pivot_distribution = pivot_distribution.loc[pivot_distribution.index >= 2010] # Filter for relevant recent years

# Plot the distribution
plt.figure(figsize=(12, 6))
pivot_distribution.plot(kind='area', stacked=True, ax=plt.gca(), color=['#E50914', '#B81D24']) # Netflix colors
plt.title('Content Volume Trend: Movies vs. TV Shows Added Annually', fontsize=16)
plt.xlabel('Year Added to Netflix', fontsize=12)
plt.ylabel('Number of Titles Added', fontsize=12)
plt.legend(title='Category')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- 4. Objective 2: Genre Analysis ---

# Function to split and count categories (genres and countries often have multiple values separated by commas)
def split_and_count(series, separator=',', top_n=10):
    # Flatten the list of genres/countries
    item_list = series.str.split(separator).explode().str.strip()
    # Count the frequency of each item
    item_counts = Counter(item_list)
    # Convert to DataFrame and return top N
    return pd.DataFrame(item_counts.items(), columns=['Item', 'Count']).sort_values('Count', ascending=False).head(top_n)

# a) Top 10 Most Common Genres Overall
top_genres_df = split_and_count(df['Genre'], separator=',', top_n=10)

# Plot Top Genres
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Item', data=top_genres_df, palette='Reds_d')
plt.title('Top 10 Most Popular Content Genres on Netflix', fontsize=16)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.show()

# b) Trend of Top 3 Genres over Years (Impressive feature)
# Find the top 3 overall genres for trend analysis
top_3_genres = top_genres_df['Item'].head(3).tolist()

# Filter data for only the top 3 genres and explode
trend_df = df[['Year Added', 'Genre']].copy()
trend_df['Genre'] = trend_df['Genre'].str.split(',')
trend_df_exploded = trend_df.explode('Genre')
trend_df_exploded['Genre'] = trend_df_exploded['Genre'].str.strip()
trend_df_filtered = trend_df_exploded[trend_df_exploded['Genre'].isin(top_3_genres)]

# Group by year and genre
genre_trend = trend_df_filtered.groupby(['Year Added', 'Genre']).size().reset_index(name='Count')
genre_trend = genre_trend.loc[genre_trend['Year Added'] >= 2015] # Focus on recent years for trend

# Plot Genre Trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=genre_trend, x='Year Added', y='Count', hue='Genre', marker='o', linewidth=3)
plt.title(f'Annual Trend of Top 3 Genres ({", ".join(top_3_genres)})', fontsize=16)
plt.xlabel('Year Added to Netflix', fontsize=12)
plt.ylabel('Number of Titles Added', fontsize=12)
plt.legend(title='Genre')
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# --- 5. Objective 3: Country-wise Contributions ---

# Top 10 Contributing Countries
top_countries_df = split_and_count(df['Country'], separator=',', top_n=10)

# Plot Top Countries
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Item', data=top_countries_df, palette=sns.color_palette("rocket", 10))
plt.title('Top 10 Contributing Countries to Netflix Catalog', fontsize=16)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()


# --- 6. Strategic Recommendations (Summary of Findings) ---
print("\n" + "="*50)
print("             STRATEGIC CONTENT RECOMMENDATIONS             ")
print("="*50)

# Findings from Objective 1
latest_year = pivot_distribution.index.max()
movies_latest = pivot_distribution.loc[latest_year, 'Movie']
tv_shows_latest = pivot_distribution.loc[latest_year, 'TV Show']
dominant_category = 'Movies' if movies_latest > tv_shows_latest else 'TV Shows'
ratio = round(movies_latest / tv_shows_latest, 1) if tv_shows_latest > 0 else "N/A"

print(f"🎬 **Content Volume Trend (As of {latest_year}):**")
print(f"  - Netflix's content acquisition heavily favors **{dominant_category}**.")
if dominant_category == 'Movies':
    print(f"  - In the last recorded year, Movies outnumbered TV Shows by a ratio of approx. **{ratio}:1**.")
else:
    print(f"  - In the last recorded year, TV Shows outnumbered Movies by a ratio of approx. **{ratio}:1**.")

print("\n🎭 **Genre Popularity:**")
print(f"  - The overall most dominant genres are **{top_3_genres[0]}**, **{top_3_genres[1]}**, and **{top_3_genres[2]}**.")
print(f"  - Recommendation: Given the surge in streaming, focus on high-production **{top_3_genres[0]}** series to retain subscription loyalty.")

print("\n🌍 **Global Content Strategy:**")
top_country = top_countries_df.iloc[0]['Item']
second_country = top_countries_df.iloc[1]['Item']
print(f"  - **{top_country}** is the primary content contributor, far surpassing all others.")
print(f"  - **{second_country}** represents the most significant international market outside the top spot.")
print("  - Recommendation: Strategically acquire more content from the top international markets and explore content in underrepresented regions to increase global subscriber reach.")
print("="*50)