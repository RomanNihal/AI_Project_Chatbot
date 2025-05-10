import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Load chatbot data from pickle
@st.cache_resource
def load_chatbot():
    with open("chatbot.pkl", "rb") as f:
        data = pickle.load(f)
    model = SentenceTransformer(data["model_name"])
    return data["faqs"], data["questions"], data["embeddings"], model

faqs, questions, embeddings, model = load_chatbot()

# Chatbot logic
def answer_question(user_input):
    user_input_lower = user_input.lower()
    user_keywords = set(user_input_lower.split())

    # Keyword match
    keyword_scores = []
    for question in questions:
        faq_keywords = set(question.lower().split())
        overlap = len(faq_keywords & user_keywords)
        keyword_scores.append(overlap)

    # Semantic similarity
    user_embedding = model.encode([user_input])
    semantic_scores = cosine_similarity(user_embedding, embeddings)[0]

    # Combine scores
    combined_scores = [0.5 * k + 0.5 * s for k, s in zip(keyword_scores, semantic_scores)]
    best_idx = int(np.argmax(combined_scores))

    if combined_scores[best_idx] < 0.4:
        return "❌ Sorry, I couldn't find an answer for that."

    return faqs[best_idx]['answer']

# Streamlit App UI
st.title("🤖 Smart Chatbot")
st.markdown("Ask me anything about BASMAH!")

# Reset chat button
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []

# Chat history session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    response = answer_question(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "user" else "assistant"):
        st.markdown(message)