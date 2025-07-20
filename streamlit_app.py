
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pickle
import re
from title_suggester import suggest_titles

# Load models and components
tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
model = AutoModel.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")

with open("regressor_model.pkl", "rb") as f:
    regressor = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Helper functions
def extract_features(text):
    num_words = len(text.split())
    num_chars = len(text)
    has_number = int(bool(re.search(r'\d', text)))
    has_question_word = int(any(q in text for q in ["لماذا", "كيف", "ما", "متى", "أين"]))
    return [num_words, num_chars, has_number, has_question_word]

def get_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    cls_embedding = output.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()

def predict_reads(title):
    embedding = get_embedding(title)
    features = extract_features(title)
    combined = np.concatenate((embedding, features)).reshape(1, -1)
    scaled = scaler.transform(combined)
    prediction = int(regressor.predict(scaled)[0])
    return prediction

# Streamlit App Layout
st.set_page_config(page_title="AI Title Assistant", layout="centered")
st.title("🧠 AI Title Assistant")
st.write("ادخل عنوان مقالك لتقدير عدد القراءات، واقتراح عناوين بديلة أفضل")

# Input
title_input = st.text_input("✍️ أدخل عنوان المقال")

if title_input:
    st.subheader("🔮 التوقع:")
    predicted_reads = predict_reads(title_input)
    st.success(f"📈 عدد القراءات المتوقع: {predicted_reads}")

    st.subheader("💡 اقتراحات بديلة:")
    with st.spinner("يتم توليد عناوين بديلة..."):
        suggestions = suggest_titles(title_input)

    ranked = []
    for alt in suggestions:
        alt_reads = predict_reads(alt)
        ranked.append((alt, alt_reads))

    ranked.sort(key=lambda x: x[1], reverse=True)

    for idx, (alt, score) in enumerate(ranked, 1):
        st.markdown(f"**{idx}. {alt}**  — 🔢 `{score}` قراءة متوقعة")
