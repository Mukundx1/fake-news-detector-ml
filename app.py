# src/app.py
import os, re, string, joblib, pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config("truthlens — Fake News Detector", layout="centered")

# ---------- Robust loader for your pipeline dict ----------
def smart_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_resource
def load_pipeline(pipeline_path="model/fake_news_pipeline.pkl"):
    """
    Loads a pipeline dict expected to contain keys 'model' and 'tfidf'.
    Returns (model, tfidf).
    """
    pipe = smart_load(pipeline_path)
    if isinstance(pipe, dict) and "model" in pipe and "tfidf" in pipe:
        return pipe["model"], pipe["tfidf"]
    # if user saved (model, tfidf) tuple
    if isinstance(pipe, (list, tuple)) and len(pipe) >= 2:
        return pipe[0], pipe[1]
    # if they accidentally saved model only
    if hasattr(pipe, "predict"):
        st.warning("Loaded object looks like a model only; TF-IDF not found.")
        return pipe, None
    raise ValueError("Pipeline file does not contain 'model' and 'tfidf'.")

# ---------- Minimal deterministic cleaner ----------
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def simple_clean(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if s == "":
        return ""
    s = re.sub(r"http\S+|www\.\S+|https\S+", " ", s)
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    tokens = [w for w in s.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ---------- Explainability helper for linear models ----------
def explain_text_from_model(model, tfidf, raw_text, topk=8):
    """
    Returns:
      {'pred': int, 'probs': [p_fake, p_real] or None,
       'top_pos': [(token,contrib)...], 'top_neg': [...]}
    """
    if tfidf is None:
        raise RuntimeError("TF-IDF vectorizer not provided. Cannot preprocess text.")
    cleaned = simple_clean(raw_text)
    X = tfidf.transform([cleaned])
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    pred = int(model.predict(X)[0])

    # if model is linear and has coef_
    if not hasattr(model, "coef_"):
        return {"pred": pred, "probs": probs.tolist() if probs is not None else None, "top_pos": [], "top_neg": []}

    coefs = model.coef_[0]
    try:
        feature_names = np.array(tfidf.get_feature_names_out())
    except Exception:
        feature_names = np.array(tfidf.get_feature_names())

    x_arr = X.toarray()[0]
    contributions = coefs * x_arr

    pos_idx = np.argsort(contributions)[-topk:][::-1]
    neg_idx = np.argsort(contributions)[:topk]

    top_pos = [(feature_names[i], float(contributions[i])) for i in pos_idx if x_arr[i] > 0]
    top_neg = [(feature_names[i], float(contributions[i])) for i in neg_idx if x_arr[i] > 0]

    return {"pred": pred, "probs": probs.tolist() if probs is not None else None, "top_pos": top_pos, "top_neg": top_neg}

# ---------- simple substring-based highlighter ----------
def highlight_text(raw_text, pos_tokens, neg_tokens):
    if raw_text is None:
        return ""
    text = str(raw_text)
    color_map = {}
    for t,_ in pos_tokens:
        color_map[t] = "#b6f2c7"
    for t,_ in neg_tokens:
        color_map[t] = "#ffccd5"
    tokens_sorted = sorted(color_map.keys(), key=len, reverse=True)
    html = text
    for tk in tokens_sorted:
        escaped = re.escape(tk)
        html = re.sub(r"(?i)\b" + escaped + r"\b",
                      lambda m: f"<mark style='background:{color_map[tk]}; padding:0.12rem 0.25rem; border-radius:3px'>{m.group(0)}</mark>",
                      html)
    return html

# ---------- UI layout ----------
st.title("🗞️ truthlens — Fake News Detector")
st.write("Paste article text or headline, or upload a .txt file. The model predicts Real (1) or Fake (0) and shows token-level explainability.")

st.sidebar.header("Model settings")
pipeline_path = st.sidebar.text_input("Pipeline path", "model/fake_news_pipeline.pkl")
topk = st.sidebar.slider("Top tokens to show", 3, 12, 8)

with st.spinner("Loading model..."):
    try:
        model, tfidf = load_pipeline(pipeline_path)
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        st.stop()

# input area
st.subheader("Input")
col1, col2 = st.columns([3,1])
with col1:
    text_input = st.text_area("Paste article / headline", height=250)
with col2:
    uploaded = st.file_uploader("Upload .txt file (optional)", type=["txt"])
    example = st.selectbox("Quick examples", ("", "Breaking: Celebrity dies mysteriously!", "The minister announced new economic reforms in parliament.", "Scientists discover new exoplanet."))

# session history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Predict
if st.button("Predict"):
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8", errors="ignore")
    elif example:
        raw = example
    else:
        raw = text_input

    if not raw or str(raw).strip() == "":
        st.warning("Please provide input text (paste, example, or upload).")
    else:
        with st.spinner("Running model..."):
            try:
                expl = explain_text_from_model(model, tfidf, raw, topk=topk)
                pred = expl["pred"]
                probs = expl["probs"]
            except Exception as e:
                st.error(f"Error during prediction/explain: {e}")
                st.stop()

        label = "REAL" if pred == 1 else "FAKE"
        st.markdown(f"### Prediction: **{label}**")
        if probs:
            st.write(f"Probabilities → Fake: **{probs[0]:.3f}**, Real: **{probs[1]:.3f}**")

        st.markdown("**Input (token highlight)**")
        highlighted = highlight_text(raw, expl.get("top_pos", []), expl.get("top_neg", []))
        st.markdown(highlighted, unsafe_allow_html=True)

        st.markdown("**Top tokens supporting REAL**")
        if expl.get("top_pos"):
            st.table(pd.DataFrame(expl["top_pos"], columns=["token","contribution"]))
        else:
            st.write("No positive tokens found.")

        st.markdown("**Top tokens supporting FAKE**")
        if expl.get("top_neg"):
            st.table(pd.DataFrame(expl["top_neg"], columns=["token","contribution"]))
        else:
            st.write("No negative tokens found.")

        st.session_state["history"].insert(0, {"text": raw, "pred": int(pred), "probs": expl.get("probs"), "top_pos": expl.get("top_pos"), "top_neg": expl.get("top_neg")})
        if len(st.session_state["history"]) > 50:
            st.session_state["history"] = st.session_state["history"][:50]

# display history
if st.session_state["history"]:
    st.subheader("Recent session history")
    for i, rec in enumerate(st.session_state["history"][:10]):
        st.write(f"**{i+1}.** Pred: {'REAL' if rec['pred']==1 else 'FAKE'} — Prob(real) {rec['probs'][1] if rec['probs'] else None:.3f}")
        st.write(rec["text"][:250] + ("..." if len(rec["text"]) > 250 else ""))
        if st.button(f"Show details #{i+1}", key=f"show{i}"):
            st.write("Top pos:", rec["top_pos"])
            st.write("Top neg:", rec["top_neg"])

st.markdown("---")
st.caption("Model loaded from model/fake_news_pipeline.pkl (dict with 'model' and 'tfidf'). Do not load untrusted pickles.")
