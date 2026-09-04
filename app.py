import os
import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce AI Support Assistant",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Resource Loader (Cached for Fast Execution)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Models & Vector Database...")
def load_all_models(
    lang_path="language_detector.joblib",
    intent_path="intent_classifier.joblib",
    sentiment_path="./distilbert_emotion_checkpoints",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    qdrant_path="./qdrant_db"
):
    # 1. Language Detector
    lang_model = joblib.load(lang_path)

    # 2. Intent Classifier
    intent_model = joblib.load(intent_path)

    # # 3. Sentiment Model (DistilBERT)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # If using checkpoints folder, locate the latest checkpoint or load root
    # tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
    # sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_path).to(device)
    # sentiment_model.eval()
    # 3. Sentiment Model (DistilBERT)
    # 3. Sentiment Model (DistilBERT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # Load from the base checkpoint architecture (or your specific fine-tuned save folder if you have it)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6).to(device)
    sentiment_model.eval()

    # 4. Sentence Embeddings & Qdrant
    embedder = SentenceTransformer(embedding_model)
    qdrant = QdrantClient(path=qdrant_path)

    return {
        "lang_model": lang_model,
        "intent_model": intent_model,
        "sentiment_model": sentiment_model,
        "sentiment_tokenizer": tokenizer,
        "sentiment_device": device,
        "embedder": embedder,
        "qdrant": qdrant
    }

EMOTION_TO_SENTIMENT = {
    "anger": "frustrated",
    "fear": "frustrated",
    "sadness": "frustrated",
    "surprise": "neutral",
    "joy": "satisfied",
    "love": "satisfied"
}

# ---------------------------------------------------------
# 3. Pipeline Analysis & Inference Functions
# ---------------------------------------------------------
def analyze_message(text: str, models: dict) -> dict:
    # A. Language Detection
    lang = models["lang_model"].predict([text])[0]

    # B. Sentiment / Emotion Classification
    inputs = models["sentiment_tokenizer"](
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(models["sentiment_device"])
    with torch.no_grad():
        logits = models["sentiment_model"](**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        idx = torch.argmax(probs).item()
        emotion = models["sentiment_model"].config.id2label[idx]
        sentiment = EMOTION_TO_SENTIMENT.get(emotion, "neutral")
        emotion_conf = probs[idx].item()

    # C. Intent Classification
    intent_probs = models["intent_model"].predict_proba([text])[0]
    max_idx = np.argmax(intent_probs)
    intent_conf = float(intent_probs[max_idx])
    intent = models["intent_model"].classes_[max_idx]

    if intent_conf < 0.45:
        intent = "out_of_scope"

    return {
        "language": lang,
        "emotion": emotion,
        "sentiment": sentiment,
        "emotion_conf": emotion_conf,
        "intent": intent,
        "intent_conf": intent_conf
    }

def retrieve_chunks(query: str, models: dict, top_k: int = 3, threshold: float = 0.45):
    query_vector = models["embedder"].encode(query, convert_to_numpy=True).tolist()
    
    response = models["qdrant"].query_points(
        collection_name="customer_support_kb",
        query=query_vector,
        limit=top_k,
        score_threshold=threshold
    )

    chunks = []
    for res in response.points:
        chunks.append({
            "score": round(res.score, 4),
            "instruction": res.payload.get("instruction"),
            "response": res.payload.get("response"),
            "intent": res.payload.get("intent"),
            "category": res.payload.get("category")
        })
    return chunks

def generate_answer(query: str, sentiment: str, chunks: list, groq_api_key: str, model_id: str) -> str:
    if not groq_api_key:
        return "⚠️ Please enter your Groq API Key in the left sidebar to generate responses."

    if not chunks:
        if sentiment == "frustrated":
            return "I completely understand your frustration and apologize for the inconvenience. Our support base does not have the exact information for your request. I am escalating this to a human specialist."
        return "I apologize, but our documentation does not contain enough information to resolve your request. Would you like me to connect you with a human support agent?"

    context_str = ""
    for i, c in enumerate(chunks, 1):
        context_str += f"Support Reference {i} [{c['category']} / {c['intent']}]:\n{c['response']}\n\n"

    system_prompt = (
        "You are a helpful, professional customer support assistant for an online retailer. "
        "Answer the customer's question using ONLY the information in the retrieved support responses below. "
        f"If the customer sounds frustrated ({sentiment}), acknowledge that empathetically before answering. "
        "If the retrieved context does not cover the question, say so honestly and offer to escalate to a human "
        "agent rather than guessing."
    )

    user_prompt = (
        f"Context (retrieved past support responses):\n"
        f"{context_str.strip()}\n\n"
        f"Customer question: \"{query}\""
    )

    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=512
    )
    return completion.choices[0].message.content

# ---------------------------------------------------------
# 4. Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings & Keys")
    groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    model_choice = st.selectbox("LLM Model", ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "mixtral-8x7b-32768"], index=0)
    try:
        models = load_all_models()
        st.success("✅ Models & Vector Store Connected")
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        st.stop()

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.pipeline_logs = []
        st.rerun()

# ---------------------------------------------------------
# 5. Session State
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! Welcome to Customer Support. How can I help you today?"}
    ]
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []

# ---------------------------------------------------------
# 6. Main Interface Layout
# ---------------------------------------------------------
st.title("🛍️ AI Customer Support Assistant (RAG Pipeline)")
st.caption("Module 1 (Language) ➔ Module 2 (Sentiment) ➔ Module 3 (Intent) ➔ Module 4 (Qdrant & Groq)")

col_chat, col_debug = st.columns([1.6, 1.4], gap="large")

with col_chat:
    st.subheader("💬 Chat Interface")
    chat_box = st.container(height=500)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_input = st.chat_input("Ask a question about orders, refunds, shipping, or accounts...")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Run Analysis (Modules 1-3)
    diag = analyze_message(user_input, models)

    # Route & Retrieve (Module 4)
    retrieved = []
    if diag["intent"] == "greeting_smalltalk":
        response = "Hello! I am happy to help you with your orders, billing, deliveries, and account settings."
        route = "DIRECT_GREETING (Bypassed Retrieval)"
    elif diag["intent"] == "out_of_scope":
        response = "I apologize, but that question is outside my support domain (e-commerce retail assistance). Would you like to speak to an agent?"
        route = "FALLBACK_OUT_OF_SCOPE"
    else:
        retrieved = retrieve_chunks(user_input, models, top_k=3)
        route = "PRIORITY_EMPATHY_RAG" if (diag["intent"] == "complaint" or diag["sentiment"] == "frustrated") else "STANDARD_RAG"
        response = generate_answer(user_input, diag["sentiment"], retrieved, groq_key, model_choice)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.pipeline_logs.append({
        "query": user_input,
        "diag": diag,
        "route": route,
        "retrieved": retrieved
    })
    st.rerun()

with col_debug:
    st.subheader("🔬 Pipeline Diagnostic Monitor")
    if st.session_state.pipeline_logs:
        last = st.session_state.pipeline_logs[-1]
        
        with st.container(border=True):
            st.markdown(f"**Query:** *\"{last['query']}\"*")
            st.divider()
            
            st.markdown("#### 1. Language Detection")
            st.code(f"Language Code: {last['diag']['language'].upper()}")

            st.markdown("#### 2. Sentiment & Emotion")
            tone_badge = "🔴 FRUSTRATED" if last['diag']['sentiment'] == "frustrated" else ("🟢 SATISFIED" if last['diag']['sentiment'] == "satisfied" else "🔵 NEUTRAL")
            st.write(f"**Emotion:** `{last['diag']['emotion']}` ({last['diag']['emotion_conf']*100:.1f}%)")
            st.write(f"**Tone Adjustment:** {tone_badge}")

            st.markdown("#### 3. Intent & Routing")
            st.write(f"**Predicted Intent:** `{last['diag']['intent']}` ({last['diag']['intent_conf']*100:.1f}%)")
            st.write(f"**Route Action:** `{last['route']}`")

            st.markdown("#### 4. Retrieved Grounding Context (Qdrant)")
            if last["retrieved"]:
                for i, r in enumerate(last["retrieved"], 1):
                    with st.expander(f"Reference #{i} | Similarity: {r['score']} | [{r['intent']}]"):
                        st.markdown(f"**Matched FAQ Question:**\n*{r['instruction']}*")
                        st.markdown(f"**Agent Answer Text:**\n{r['response']}")
            else:
                st.caption("No vector database retrieval triggered for this action.")
    else:
        st.info("Submit a question in the chat box on the left to inspect the step-by-step pipeline outputs in real time.")