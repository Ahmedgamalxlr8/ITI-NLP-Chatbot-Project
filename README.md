# E-Commerce AI Support Assistant

An end-to-end multi-stage AI customer support chatbot built with Python, PyTorch, Qdrant, Groq, and Streamlit. The system processes user inquiries through a four-module NLP pipeline to ensure accurate, sentiment-aware, and grounded responses.

---

## System Architecture

* **Module 1 (Language Identification)**: Detects the language of incoming messages using a trained TF-IDF and Logistic Regression pipeline.
* **Module 2 (Sentiment & Emotion Classification)**: Evaluates user emotional tone and maps fine-grained classifications into distinct sentiment buckets (`frustrated`, `neutral`, `satisfied`) using a fine-tuned DistilBERT model.
* **Module 3 (Intent Classification & Routing)**: Categorizes inquiries using TF-IDF and Logistic Regression trained on the Bitext customer support dataset, incorporating confidence-score gating to catch out-of-scope queries.
* **Module 4 (Retrieval-Augmented Generation)**: Embeds support chunks using `all-MiniLM-L6-v2`, indexes them in a local persistent **Qdrant** vector store, and queries Groq-hosted LLMs for grounded response generation with tone modulation.

---

## Project Structure

```text
├── 01_language_detection.ipynb        # Module 1 notebook
├── 02_sentiment_classifier_distilbert.ipynb # Module 2 notebook
├── 03_intent_classifier.ipynb         # Module 3 notebook
├── 04_rag_pipeline.ipynb              # Module 4 notebook
├── language_detector.joblib           # Saved language model artifact
├── intent_classifier.joblib           # Saved intent model artifact
├── qdrant_db/                         # Persistent Qdrant vector database folder
├── app.py                             # Streamlit web application script
└── requirements.txt                   # Project dependencies

```

---

## Installation & Setup

1. **Clone or download** the project repository and navigate to the project directory in your terminal.
2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Set your Groq API Key**:
Get a free API key from the [Groq Console](https://console.groq.com/keys) and input it directly into the application sidebar.

---

## Running the Application

Start the local interactive user interface by running:

```bash
streamlit run app.py

```

Streamlit will automatically launch and open the application in your browser at `http://localhost:8501`.

