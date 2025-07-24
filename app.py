import streamlit as st
import pickle
import faiss
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
import language_tool_python
from spellchecker import SpellChecker
from openai import OpenAI

# === CONFIGURATION GROQ ===
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],  # 👈 clé stockée de manière sécurisée dans Streamlit Cloud
    base_url="https://api.groq.com/openai/v1"
)

# === CORRECTEURS ===
tool = language_tool_python.LanguageTool('fr')
spell = SpellChecker(language='fr')

def corriger_question(question):
    mots = question.split()
    corrigé = " ".join([spell.correction(m) if spell.unknown([m]) else m for m in mots])
    return tool.correct(corrigé)

def normalize(text):
    text = text.lower()
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

def clean_text(text):
    return "\n".join(line.strip() for line in text.split("\n") if len(line.strip()) > 1)

# === CHARGEMENT DES DONNÉES ===
with open("chunks.pkl", "rb") as f:
    chunk_id_to_text_and_tag = pickle.load(f)

index = faiss.read_index("faiss_index.bin")
encoder_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def detect_compagnie(question):
    norm_q = normalize(question)
    tags_norm = [normalize(tag) for _, tag in chunk_id_to_text_and_tag.values()]
    for comp in set(tags_norm):
        if comp in norm_q:
            return comp
    return None

def ask_groq(prompt, max_tokens=300):
    try:
        # 💬 Si salutation simple, répondre brièvement
        salutations = ["bonjour", "salut", "bonsoir", "coucou"]
        if any(s in prompt.lower() for s in salutations):
            return "Bonjour !"

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Tu es un assistant spécialisé pour une agence de voyage. Réponds de manière claire et chaleureuse."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERREUR API] {str(e)}"

def rag_answer_llama(question, top_k=5, max_prompt_tokens=800):
    question = corriger_question(question)
    compagnie = detect_compagnie(question)
    question_embedding = encoder_model.encode([question])

    if compagnie:
        compagnie_normalisee = normalize(compagnie)
        ids = [i for i, (_, tag) in chunk_id_to_text_and_tag.items()
               if normalize(tag) == compagnie_normalisee]
        if not ids:
            return f"Aucune information trouvée pour : {compagnie}"
        embeddings = encoder_model.encode([chunk_id_to_text_and_tag[i][0] for i in ids])
        sub_index = faiss.IndexFlatL2(embeddings.shape[1])
        sub_index.add(np.array(embeddings))
        D, I = sub_index.search(np.array(question_embedding), top_k)
        indices = [ids[i] for i in I[0]]
    else:
        D, I = index.search(np.array(question_embedding), top_k)
        indices = I[0]

    context = ""
    for i in indices:
        chunk, tag = chunk_id_to_text_and_tag[i]
        chunk = clean_text(chunk)
        if len((context + chunk + question).split()) > max_prompt_tokens:
            break
        context += chunk + "\n"

    if not context.strip():
        return "Je n’ai pas trouvé d’information pertinente."

    prompt = f"""Voici des informations :\n{context}\n\nQuestion : {question}\nRéponds de manière professionnelle et utile :"""
    return ask_groq(prompt)

# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="Assistant ✈️", page_icon="🧳")
st.title("🤖 Assistant RAG - Agence de Voyage")

question = st.text_input("Pose ta question ici 👇")
if question:
    reponse = rag_answer_llama(question)
    st.markdown(f"**✉️ Réponse :** {reponse}")
