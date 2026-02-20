import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# 1. Create product dataset
# -------------------------------------------------
data = {
    "product_name": [
        "Wireless Mouse", "Gaming Mouse", "Mechanical Keyboard",
        "Bluetooth Headphones", "Wired Headphones", "Smartphone",
        "Laptop", "Tablet", "Smartwatch", "Fitness Tracker"
    ],
    "category": [
        "Accessories", "Accessories", "Accessories",
        "Audio", "Audio", "Electronics",
        "Electronics", "Electronics", "Wearables", "Wearables"
    ],
    "tags": [
        "wireless usb mouse", "gaming rgb mouse", "mechanical rgb keyboard",
        "bluetooth wireless headphones", "wired headphones",
        "android smartphone", "gaming laptop", "android tablet",
        "smart watch", "fitness health tracker"
    ],
    "description": [
        "A wireless mouse with ergonomic design",
        "High precision gaming mouse with RGB lights",
        "Mechanical keyboard with backlit keys",
        "Bluetooth headphones with noise cancellation",
        "Wired headphones with clear sound",
        "Smartphone with high resolution camera",
        "Laptop with powerful performance",
        "Tablet with large display",
        "Smartwatch with heart rate monitor",
        "Fitness tracker for daily activity monitoring"
    ]
}

df = pd.DataFrame(data)

# -------------------------------------------------
# 2. Text cleaning function
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["category_clean"] = df["category"].apply(clean_text)
df["tags_clean"] = df["tags"].apply(clean_text)
df["description_clean"] = df["description"].apply(clean_text)

# -------------------------------------------------
# 3. Feature weighting
# -------------------------------------------------
CATEGORY_WEIGHT = 3
TAGS_WEIGHT = 2
DESCRIPTION_WEIGHT = 1

df["combined_features"] = (
    (df["category_clean"] + " ") * CATEGORY_WEIGHT +
    (df["tags_clean"] + " ") * TAGS_WEIGHT +
    (df["description_clean"] + " ") * DESCRIPTION_WEIGHT
)

# -------------------------------------------------
# 4. TF-IDF Vectorization
# -------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# -------------------------------------------------
# 5. Cosine similarity
# -------------------------------------------------
cosine_sim = cosine_similarity(tfidf_matrix)

# -------------------------------------------------
# 6. Recommendation function
# -------------------------------------------------
def recommend_products(product_name, top_n=3):
    if product_name not in df["product_name"].values:
        return pd.DataFrame({"Error": ["Selected product not found in database."]})

    product_index = df[df["product_name"] == product_name].index[0]
    similarity_scores = list(enumerate(cosine_sim[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]

    product_indices = [i[0] for i in similarity_scores]
    result_df = df.loc[product_indices, ["product_name", "category", "tags"]].copy()
    result_df["similarity_score"] = [round(i[1], 3) for i in similarity_scores]

    return result_df

# -------------------------------------------------
# 7. Streamlit UI
# -------------------------------------------------
st.title("🛒 Product Recommendation System")
st.write("Content-based recommender using TF-IDF and cosine similarity")

# Show dataset
st.subheader("📦 Product Dataset")
st.dataframe(df[["product_name", "category", "tags", "description"]])

# Dropdown for product selection
st.subheader("🔽 Select a Product")
selected_product = st.selectbox("Choose a product:", [""] + df["product_name"].tolist())

# Button
if st.button("Get Recommendations"):
    if selected_product == "":
        st.error("Please select a product first.")
    else:
        recommendations = recommend_products(selected_product)

        st.subheader("✅ Top 3 Recommended Products")
        st.dataframe(recommendations)
