# 🤖 Smart Chatbot

An intelligent FAQ chatbot designed to improve donor support experiences. This AI assistant can accurately respond to donation-related questions using a hybrid approach combining keyword matching and semantic similarity (via SBERT).
![Chatbot](https://github.com/user-attachments/assets/454fc6ad-b00f-4e9f-bd07-2f642285452e)

---

## Features

- ✅ Load and process a custom JSON-based FAQ knowledge base
- 🔍 Combines **Keyword Matching** + **Semantic Similarity** (Hybrid Search)
- 🧠 Uses **SBERT (Sentence-BERT)** for context-aware matching
- 💬 Handles paraphrased questions with high accuracy
- ❌ Gracefully returns fallback message if no confident match
- 💾 Model can be saved as `.pkl` for reuse or integration into web apps

---

## Use Cases

- Enhance the donor or customer support experience
- Automate responses to frequently asked questions
- Reduce repetitive manual workload for staff or volunteers
- Integrate with existing websites, chat platforms, or support systems

---

## Built With

- **Python**
- **Sentence-Transformers (SBERT)** – for semantic embeddings
- **Scikit-learn** – for cosine similarity calculations
- **NumPy / JSON** – for data and storage
- **Streamlit** – for the chat UI
