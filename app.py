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
# 2. Text cleaning
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
# 4. TF-IDF
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
        return None

    idx = df[df["product_name"] == product_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]

    indices = [i[0] for i in scores]
    result_df = df.loc[indices, ["product_name", "category", "tags", "description"]].copy()
    result_df["similarity_score"] = [round(i[1], 3) for i in scores]

    return result_df

# -------------------------------------------------
# 7. Streamlit UI
# -------------------------------------------------
st.title("🛒 Product Recommendation System")

st.subheader("📦 Product Dataset")
st.dataframe(df[["product_name", "category", "tags", "description"]])

selected_product = st.selectbox("Select a product:", [""] + df["product_name"].tolist())

if st.button("Get Recommendations"):
    if selected_product == "":
        st.error("Please select a product.")
    else:
        # Show selected product details
        st.subheader("🟢 Selected Product")
        product_row = df[df["product_name"] == selected_product].iloc[0]

        st.markdown(f"""
        **Name:** {product_row['product_name']}  
        **Category:** {product_row['category']}  
        **Tags:** {product_row['tags']}  
        **Description:** {product_row['description']}
        """)

        # Get recommendations
        recommendations = recommend_products(selected_product)

        st.subheader("⭐ Recommended Products")

        # Display recommendations as cards
        cols = st.columns(3)

        for col, (_, row) in zip(cols, recommendations.iterrows()):
            with col:
                st.markdown("### 🛍️ " + row["product_name"])
                st.markdown(f"**Category:** {row['category']}")
                st.markdown(f"**Tags:** {row['tags']}")
                st.markdown(f"**Description:** {row['description']}")
                st.markdown(f"**Similarity:** {row['similarity_score']}")
                st.markdown("---")
