# 📰 Spam-News-Classifier

A machine learning web app built and hosted on **Hugging Face Spaces** to classify text as **Spam** or **News** using natural language processing (NLP) techniques.

---

## 🚀 Overview
This project provides an easy-to-use web interface where users can input a message, headline, or short text and instantly get a prediction:  
**Is it spam or genuine news?**

It’s powered by a trained ML model and deployed using **Streamlit + Hugging Face Spaces** for real-time inference.

---

## 🔍 Features
- 🧠 Classifies text into two categories — *Spam* or *News*  
- ⚡ Instant live predictions  
- 🎨 Simple and clean web UI  
- 🧾 Shows confidence score (optional)  
- ☁️ Hosted free on Hugging Face Spaces

---

## 🧠 Model & Dataset
- **Algorithm:** Machine Learning / NLP (e.g., Logistic Regression, RandomForest, XGBoost, or custom fine-tuned transformer)
- **Dataset:** Custom dataset combining spam messages and Indian news headlines
- **Text Preprocessing:** Tokenization, stopword removal, punctuation cleanup, TF-IDF vectorization
- **Classes:**
  - 🟥 **Spam:** Misleading or promotional content
  - 🟩 **News:** Legitimate, informative news text
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Model** | Scikit-learn / XGBoost |
| **Deployment** | Hugging Face Spaces |
| **Data Handling** | Pandas, NumPy, Joblib |

---

## 🧑‍💻 How to Use
1. Visit the live app here 👉 [**Spam-News-Classifier**](https://huggingface.co/spaces/faizal76/Spam-News-Classifier)
2. Enter any message, sentence, or headline in the input box  
3. Click **Predict**  
4. Instantly view the classification result  
5. Optionally test with multiple examples

---

## ⚙️ Run Locally (Developer Guide)
If you want to clone and run this project locally:

```bash
# 1️⃣ Clone the repo
git clone https://huggingface.co/spaces/faizal76/Spam-News-Classifier
cd Spam-News-Classifier

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run app.py
