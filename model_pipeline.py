# --- model_pipeline.py ---
import os
import re
import cv2
import joblib
import pickle
import tempfile
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from docx import Document

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# ---------------------------
# Keyword categories (fallback)
# ---------------------------
sample_keywords = {
    'Food': ['restaurant', 'food', 'burger', 'pizza', 'eat', 'meal', 'kitchen'],
    'Travel': ['uber', 'taxi', 'train', 'flight', 'bus', 'ola', 'cab'],
    'Office Supplies': ['pen', 'notebook', 'printer', 'office', 'paper', 'stapler'],
    'Accommodation': ['hotel', 'room', 'stay', 'inn', 'resort', 'check-in'],
    'Other': []
}

def train_category_model():
    texts, labels = [], []
    for cat, words in sample_keywords.items():
        for w in words:
            texts.append(w)
            labels.append(cat)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    joblib.dump((clf, vectorizer), "category_model.pkl")

if not os.path.exists("category_model.pkl"):
    train_category_model()

clf, vectorizer = joblib.load("category_model.pkl")

# ---------------------------
# File handling
# ---------------------------
def convert_to_image(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        return path
    elif ext == ".pdf":
        images = convert_from_path(path, first_page=1, last_page=1)
        temp_path = tempfile.mktemp(suffix=".jpg")
        images[0].save(temp_path, "JPEG")
        return temp_path
    elif ext == ".docx":
        doc = Document(path)
        for rel in doc.part._rels:
            rel = doc.part._rels[rel]
            if "image" in rel.target_ref:
                image_data = rel.target_part.blob
                temp_path = tempfile.mktemp(suffix=".jpg")
                with open(temp_path, "wb") as f:
                    f.write(image_data)
                return temp_path
        raise ValueError("❌ No image found in DOCX file.")
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")

# ---------------------------
# Image preprocessing / OCR
# ---------------------------
def rotate_and_preprocess(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def extract_text_from_file(path):
    img_path = convert_to_image(path)
    rotated = rotate_and_preprocess(img_path)
    text = pytesseract.image_to_string(rotated)
    return text

# ---------------------------
# Text → fields
# ---------------------------
def extract_fields(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    merchant = next((line for line in lines[:5] if any(c.isalpha() for c in line)), "Unknown")

    # Date
    date = "Not Found"
    for pattern in [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{1,2}(st|nd|rd|th)?\s+\w+\s+\d{2,4}',
        r'\w+\s+\d{1,2},\s+\d{4}',
        r'\d{4}-\d{2}-\d{2}'
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            date = m.group()
            break

    # Time
    time_val = "Not Found"
    m = re.search(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', text)
    if m:
        time_val = m.group()

    # Amount
    amount = "Not Found"
    currency_pattern = r'£?\d+[.,]?\d*\d'
    amount_keywords = ['total', 'subtotal', 'amount due', 'grand total', 'payment']
    for line in lines:
        if any(kw in line.lower() for kw in amount_keywords):
            m = re.search(currency_pattern, line)
            if m:
                amount = m.group()
                if not amount.startswith('£'):
                    amount = '£' + amount
                break
    if amount == "Not Found":
        all_amounts = re.findall(r'\d+[.,]?\d*', text)
        if all_amounts:
            max_val = max(map(float, [a.replace(',', '') for a in all_amounts]))
            amount = f"£{max_val:.2f}"

    # Keyword category (fallback)
    category = "Other"
    lower_text = text.lower()
    for cat, keywords in sample_keywords.items():
        if any(kw in lower_text for kw in keywords):
            category = cat
            break

    return {"Merchant": merchant, "Date": date, "Time": time_val, "Amount": amount, "Category": category}

def extract_entities(text):
    return extract_fields(text)

def preprocess_features(entities):
    return f"{entities['Merchant']} {entities['Date']} {entities['Time']} {entities['Amount']}"

# ---------------------------
# Feedback (category) training
# ---------------------------
def load_training_data_from_corrections(db_connection_fn):
    conn = db_connection_fn()
    cur = conn.cursor()
    cur.execute("""
        SELECT merchant, date, time, amount, category FROM corrected_receipts
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=["merchant", "date", "time", "amount", "category"])

def retrain_model(df, model_path="model_feedback.pkl"):
    if df is None or df.empty:
        return None
    df["text"] = (
        df["merchant"].astype(str) + " "
        + df["date"].astype(str) + " "
        + df["time"].astype(str) + " "
        + df["amount"].astype(str)
    )
    X, y = df["text"], df["category"]
    pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    return pipeline

def should_trigger_retraining(get_connection_fn, threshold=10):
    conn = get_connection_fn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM corrected_receipts")
    total = cur.fetchone()[0]
    cur.execute("SELECT last_count FROM retrain_log ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    last_count = row[0] if row else 0
    cur.close(); conn.close()
    return total - last_count >= threshold

def update_retrain_log(get_connection_fn):
    conn = get_connection_fn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM corrected_receipts")
    total = cur.fetchone()[0]
    cur.execute("INSERT INTO retrain_log (last_count) VALUES (%s)", (total,))
    conn.commit()
    cur.close(); conn.close()

# ---------------------------
# Isolation Forest (anomaly)
# ---------------------------
def _amount_to_float(amount_str):
    try:
        cleaned = str(amount_str).replace("£", "").replace(",", "")
        val = float(re.findall(r"(\d+\.?\d*)", cleaned)[0])
        return val
    except Exception:
        return None

def load_anomaly_training_data(db_connection_fn):
    """
    Pull reviewed receipts and return a DataFrame with 'AmountVal' (float).
    """
    conn = db_connection_fn()
    cur = conn.cursor()
    cur.execute("""
        SELECT amount
        FROM receipts
        WHERE anomaly_status IN ('approved','rejected')
          AND amount ~ '^[£]?[0-9,]+(\\.[0-9]{1,2})?$'
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()

    df = pd.DataFrame(rows, columns=["amount"])
    df["AmountVal"] = df["amount"].apply(_amount_to_float)
    df = df.dropna(subset=["AmountVal"]).reset_index(drop=True)
    return df[["AmountVal"]]

def train_isolation_forest(df_amounts, model_path="anomaly_iforest.pkl", contamination=0.05, random_state=42):
    """
    Train IF on AmountVal and save.
    """
    if df_amounts is None or df_amounts.empty:
        return None
    X = df_amounts[["AmountVal"]].values
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    joblib.dump(model, model_path)
    return model

def predict_anomaly_on_amount(amount_str, model_path="anomaly_iforest.pkl"):
    """
    Returns (label, score): -1 anomaly, 1 normal; score lower = more anomalous.
    If model missing or amount invalid -> (None, None).
    """
    val = _amount_to_float(amount_str)
    if val is None:
        return (None, None)
    try:
        model = joblib.load(model_path)
    except Exception:
        return (None, None)
    X = np.array([[val]])
    label = int(model.predict(X)[0])
    score = float(model.decision_function(X)[0])
    return (label, score)

# ---------------------------
# Inference (category + anomaly meta)
# ---------------------------
def predict_receipt(path):
    text = extract_text_from_file(path)
    entities = extract_entities(text)

    # Category: use feedback model if available; else fallback keywords
    processed = preprocess_features(entities)
    try:
        with open("model_feedback.pkl", "rb") as f:
            cat_model = pickle.load(f)
        category = cat_model.predict([processed])[0]
    except Exception:
        category = entities.get("Category", "Other")
    entities["Category"] = category

    # Isolation Forest anomaly meta
    label, score = predict_anomaly_on_amount(entities.get("Amount"))
    entities["IForestLabel"] = label     # -1 anomaly, 1 normal, None if unavailable
    entities["IForestScore"] = score

    return entities