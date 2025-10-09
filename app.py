import streamlit as st
import joblib
from random import choice
import pandas as pd
import os
import csv

# -----------------------------
# Load trained model, vectorizer, encoder
# -----------------------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Feedback storage file
# -----------------------------
FEEDBACK_FILE = "feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["text", "predicted", "correct"]).to_csv(FEEDBACK_FILE, index=False)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake/Real Classifier 🚀",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")
st.sidebar.write("""
This is an interactive **Fake / Real News Classifier**.  
It uses **XGBoost + TF-IDF** to detect spam emails or fake news.  
Type your text below or try a random example!
""")

# -----------------------------
# Main Title
# -----------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📧 Fake / Real News Classifier 🚀</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Example Buttons
# -----------------------------
example_texts = [
    "Congratulations!!! You won $1000 gift card, claim now.",
    "NASA launches rover to Mars successfully.",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
    "Government announces new education policy today."
]

col1, col2 = st.columns(2)
with col1:
    if st.button("Random Example 1"):
        st.session_state["text_input"] = choice(example_texts)
with col2:
    if st.button("Random Example 2"):
        st.session_state["text_input"] = choice(example_texts)

# -----------------------------
# Text Input
# -----------------------------
text_input = st.text_area("Enter your text here:", value=st.session_state.get("text_input", ""))

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict 🕵️"):
    if text_input.strip() != "":
        X_new = vectorizer.transform([text_input])
        y_pred = model.predict(X_new)
        y_prob = model.predict_proba(X_new)[0][1]

        label = encoder.inverse_transform(y_pred)[0]

        # ✅ store in session state so feedback buttons can use it
        st.session_state["last_text"] = text_input
        st.session_state["last_label"] = label

        if label.lower() == "spam":
            st.markdown(f"<h2 style='color: red;'>⚠️ Prediction: {label} ({y_prob*100:.2f}% confidence)</h2>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"<h2 style='color: green;'>✅ Prediction: {label} ({(1-y_prob)*100:.2f}% confidence)</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to classify!")

# -----------------------------
# Feedback Buttons (always visible if last prediction exists)
# -----------------------------
if "last_text" in st.session_state and "last_label" in st.session_state:
    st.write("Was this prediction correct?")
    col1, col2 = st.columns(2)

    def save_feedback(text, predicted, correct):
        file_exists = os.path.isfile(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if os.stat(FEEDBACK_FILE).st_size == 0:  # write header only if file empty
                writer.writerow(["text", "predicted", "correct"])
            writer.writerow([text, predicted, correct])

    with col1:
        if st.button("✅ Correct - Real"):
            save_feedback(st.session_state["last_text"], st.session_state["last_label"], "real")
            st.success("Feedback saved! (Marked as Real)")

    with col2:
        if st.button("⚠️ Correct - Fake"):
            save_feedback(st.session_state["last_text"], st.session_state["last_label"], "spam")
            st.success("Feedback saved! (Marked as Fake)")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & XGBoost</p>", unsafe_allow_html=True)




