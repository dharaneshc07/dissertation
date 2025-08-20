
# Imports & App Setup

import os
import re
import shutil
from datetime import datetime

import bcrypt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from zoneinfo import ZoneInfo

from model_pipeline import (
    predict_receipt,
    convert_to_image,
    load_training_data_from_corrections,
    retrain_model,
    should_trigger_retraining,
    update_retrain_log,
)

st.set_page_config(
    page_title="125 – 126 Agentic-AI to optimise expense submission process",
    layout="wide",
)
st.title("125 – 126 Agentic-AI to optimise expense submission process")
# Safe debug banner 
try:
    st.caption(
        f"DB host: {_get_secret('DB_HOST','?')} | "
        f"port: {_get_secret('DB_PORT','?')}"
    )
except Exception:
    pass


# Keep Streamlit session consistent
for key, val in {"page": "landing", "role": None, "user": None, "enable_edit": False}.items():
    if key not in st.session_state:
        st.session_state[key] = val



# DB Connection & Auth (single source of truth)

# Secrets helper + DB pool
import os, streamlit as st
from psycopg2.pool import SimpleConnectionPool

def _get_secret(name: str, default: str | None = None) -> str | None:
    """Prefer Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

@st.cache_resource(show_spinner=False)
def db_pool() -> SimpleConnectionPool:
    return SimpleConnectionPool(
        minconn=1,
        maxconn=50,
        dbname=_get_secret("DB_NAME", "neondb"),
        user=_get_secret("DB_USER", "neondb_owner"),
        password=_get_secret("DB_PASSWORD", ""),
        host=_get_secret("DB_HOST", "localhost"),
        port=_get_secret("DB_PORT", "5432"),
        sslmode=_get_secret("DB_SSLMODE", "require"),
    )

def get_connection():
    pool = db_pool()
    try:
        return pool.getconn()
    except Exception as e:
        st.error(f"DB connection failed ❌: {type(e).__name__}: {e}")
        return None

def release_connection(conn):
    if conn:
        try:
            db_pool().putconn(conn)
            st.write("✅ ")
        except Exception as e:
            st.write(f"⚠️ Failed to return connection: {e}")

from contextlib import contextmanager

def _resolve_display_image(image_path: str | None) -> str | None:
    """
    Try to resolve an on-disk image we can show.
    - If the given path exists, use it.
    - If it looks like an absolute path that doesn't exist, try its basename under 'uploads/'.
    - If it's a PDF/DOCX or missing, return None (UI will show a friendly note).
    """
    if not image_path:
        return None

    # If path exists as-is, use it
    if os.path.exists(image_path):
        # Don’t try to open PDFs/DOCX directly in st.image
        if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            return image_path
        return None

    # If absolute or foreign path try basename inside uploads
    base = os.path.basename(image_path)
    guess = os.path.join("uploads", base)
    if os.path.exists(guess) and guess.lower().endswith((".jpg", ".jpeg", ".png")):
        return guess

    return None

# ---- Receipt preview helper ----
def display_receipt(image_path: str) -> bool:
    """
    Try to show a preview of a stored receipt.
    Returns True if an image was shown, else False.
    """
    try:
        # convert_to_image handles pdf/docx -> jpg path for preview
        preview = convert_to_image(image_path)
        # If convert_to_image returns a temp path, prefer that
        candidate = preview if preview and os.path.exists(preview) else image_path
        if candidate and os.path.exists(candidate):
            st.image(candidate, width=250)
            return True
    except Exception as e:
        # Optional: uncomment for debugging
        # st.caption(f"Preview error: {type(e).__name__}: {e}")
        pass
    return False

    
@contextmanager
def get_conn():
    pool = db_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        try:
            pool.putconn(conn)
        except Exception:
            pass

def check_credentials(username: str, password: str):
    """
    Return role ('admin' / 'employee') if credentials are valid; else None.
    """
    conn = get_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash, role FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        cur.close()
        if row:
            pw_hash, role = row
            if bcrypt.checkpw(password.encode(), pw_hash.encode()):
                return role
    except Exception as e:
        st.error(f"❌ Login failed: {e}")
    finally:
        release_connection(conn)
    return None

def create_user(username: str, password: str, role: str):
    """Create a new user with hashed password."""
    conn = get_connection()
    if not conn:
        st.error("No DB connection.")
        return
    try:
        cur = conn.cursor()
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
            (username, hashed, role),
        )
        conn.commit()
        cur.close()
        st.success(f"{role.title()} account created for '{username}'!")
    except Exception as e:
        conn.rollback()
        st.error(f"❌ Could not create user: {e}")
    finally:
        release_connection(conn)

# tiny connectivity banner 
if "db_checked" not in st.session_state:
    st.session_state.db_checked = True
    _c = get_connection()
    if _c:
        try:
            with _c.cursor() as _cur:
                _cur.execute("SELECT 1;")
            st.caption("🔌 Database: connected")
        except Exception as e:
            st.caption(f"🔌 Database: connection error — {e}")
        finally:
            release_connection(_c)

# DB Operations (Receipts, Feedback, Admin)

def insert_receipt(username, merchant, date, time, amount, category, was_corrected, image_path):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO receipts (username, merchant, date, time, amount, category, was_corrected, image_path, anomaly_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (username, merchant, date, time, amount, category, was_corrected, image_path, None))
    conn.commit()
    cur.close()
    conn.close()

def insert_corrected_receipt(username, merchant, date, time, amount, category, original_image):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO corrected_receipts (username, merchant, date, time, amount, category, original_image)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (username, merchant, date, time, amount, category, original_image))
    conn.commit()
    cur.close()
    conn.close()

def fetch_receipts(username=None):
    conn = get_connection()
    cur = conn.cursor()
    if username:
        cur.execute("""
            SELECT merchant, date, time, amount, category, was_corrected, uploaded_at, image_path, anomaly_status
            FROM receipts WHERE username = %s ORDER BY uploaded_at DESC
        """, (username,))
    else:
        cur.execute("""
            SELECT username, merchant, date, time, amount, category, was_corrected, uploaded_at, image_path, anomaly_status
            FROM receipts ORDER BY uploaded_at DESC
        """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def fetch_corrected_receipts():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT username, merchant, date, time, amount, category, original_image, corrected_at
        FROM corrected_receipts
        ORDER BY corrected_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def fetch_users():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE role = 'employee'")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return [u[0] for u in users]

def fetch_flagged_receipts():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, merchant, date, time, amount, category, uploaded_at, image_path, anomaly_status
        FROM receipts
        WHERE amount ~ '^[£]?[0-9]+(\\.[0-9]{1,2})?$'
          AND CAST(REPLACE(amount, '£', '') AS FLOAT) > 100
          AND anomaly_status IS NULL
        ORDER BY uploaded_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def update_anomaly_status_by_id(receipt_id: int, decision: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE receipts
        SET anomaly_status = %s
        WHERE id = %s
    """, (decision, receipt_id))
    conn.commit()
    cur.close()
    conn.close()

def update_anomaly_status(username, uploaded_at, decision):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE receipts
        SET anomaly_status = %s
        WHERE username = %s AND uploaded_at = %s
    """, (decision, username, uploaded_at))
    conn.commit()
    cur.close()
    conn.close()

def delete_employee_and_data(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM receipts WHERE username = %s", (username,))
    cur.execute("DELETE FROM corrected_receipts WHERE username = %s", (username,))
    cur.execute("DELETE FROM anomaly_feedback WHERE username = %s", (username,))
    cur.execute("DELETE FROM users WHERE username = %s", (username,))
    conn.commit()
    cur.close()
    conn.close()

def get_corrected_receipts():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT merchant, date, time, amount, category FROM corrected_receipts")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def insert_anomaly_feedback(username, merchant, date, time, amount, category, decision, uploaded_at):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO anomaly_feedback (username, merchant, date, time, amount, category, decision, uploaded_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (username, uploaded_at) DO UPDATE
        SET decision = EXCLUDED.decision
    """, (username, merchant, date, time, amount, category, decision, uploaded_at))
    conn.commit()
    cur.close()
    conn.close()



# Upload UI & Prediction 
def render_upload_ui(user):
    import uuid
    col1, col2 = st.columns(2)
    os.makedirs("uploads", exist_ok=True)

    # Ensure session keys exist
    for k in ["raw_path", "preview_path", "predicted", "enable_edit"]:
        st.session_state.setdefault(k, "" if k.endswith("path") else ({} if k == "predicted" else False))

    with col1:
        st.markdown("### 📤 Upload Receipt")
        uploaded = st.file_uploader("Upload receipt", type=["jpg", "png", "jpeg", "pdf", "docx",])
        if uploaded:
            # Persist file to disk
            ext = uploaded.name.split(".")[-1].lower()
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            uid = uuid.uuid4().hex[:6]
            raw_path = f"uploads/{user}_{ts}_{uid}.{ext}"
            with open(raw_path, "wb") as f:
                f.write(uploaded.read())

            # preview and predict
            try:
                tmp_img = convert_to_image(raw_path)
                preview_path = f"uploads/preview_{user}_{ts}_{uid}.jpg"
                shutil.copy(tmp_img, preview_path)

                st.image(preview_path, caption="Uploaded Receipt", use_column_width=True)
                predicted = predict_receipt(raw_path)
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                return

            # Store in session for the Submit click rerun
            st.session_state.raw_path = raw_path
            st.session_state.preview_path = preview_path
            st.session_state.predicted = predicted or {}

            # Show predicted 
            for k, v in st.session_state.predicted.items():
                st.text_input(k, v, disabled=True, key=f"pred_{k}")

            st.session_state.enable_edit = st.radio(
                "Is the prediction accurate?", ["Yes", "Needs Correction"], index=0
            ) == "Needs Correction"

    with col2:
        st.markdown("### ✏️ Confirm or Correct Details")

        # Use predicted values as defaults
        p = st.session_state.predicted if isinstance(st.session_state.predicted, dict) else {}

        merchant = st.text_input("Merchant", p.get("Merchant", ""), disabled=not st.session_state.enable_edit, key="inp_merchant")
        date = st.text_input("Date", p.get("Date", ""), disabled=not st.session_state.enable_edit, key="inp_date")
        time_ = st.text_input("Time", p.get("Time", ""), disabled=not st.session_state.enable_edit, key="inp_time")
        amount = st.text_input("Amount", p.get("Amount", ""), disabled=not st.session_state.enable_edit, key="inp_amount")
        category = st.selectbox(
            "Category",
            ["Food", "Travel", "Office Supplies", "Accommodation", "Other"],
            index=0 if not p.get("Category") else  ["Food","Travel","Office Supplies","Accommodation","Other"].index(p.get("Category")) if p.get("Category") in ["Food","Travel","Office Supplies","Accommodation","Other"] else 0,
            disabled=not st.session_state.enable_edit,
            key="inp_category"
        )

        if st.button("Submit Receipt"):
            # Use session-stored paths from the upload step
            final_path = st.session_state.preview_path

            # Force storing a safe relative preview if possible
            if st.session_state.preview_path:
                final_path = st.session_state.preview_path

            insert_receipt(
                user, merchant, date, time_, amount, category,
                st.session_state.enable_edit,
                final_path  # <— this should be uploads/preview_*.jpg
            )

            if not final_path:
                st.error("No uploaded file found. Please upload a receipt first.")
                return

            # When user said “Yes” , use predictions
            if not st.session_state.enable_edit:
                merchant = p.get("Merchant", "")
                date = p.get("Date", "")
                time_ = p.get("Time", "")
                amount = p.get("Amount", "")
                category = p.get("Category", "Other")

            # Minimal validation
            if not merchant or not amount:
                st.error("Merchant and Amount are required.")
                return

            try:
                insert_receipt(user, merchant, date, time_, amount, category, st.session_state.enable_edit, final_path)

                if st.session_state.enable_edit:
                    insert_corrected_receipt(user, merchant, date, time_, amount, category, final_path)

                    # trigger feedback retrain
                    from model_pipeline import should_trigger_retraining, update_retrain_log, retrain_model, load_training_data_from_corrections
                    if should_trigger_retraining(get_connection):
                        df_fb = load_training_data_from_corrections(get_connection)
                        retrain_model(df_fb, model_path="model_feedback.pkl")
                        update_retrain_log(get_connection)
                        st.info("🔁 Model retrained using recent corrections.")

                # Clear session vars after successful submit
                st.session_state.raw_path = ""
                st.session_state.preview_path = ""
                st.session_state.predicted = {}
                st.session_state.enable_edit = False

                st.success("✅ Receipt saved successfully.")
            except Exception as e:
                # Show full DB error to diagnose
                st.exception(e)



# Receipt History & Admin Views

def render_receipts(user):
    st.subheader("📂 Your Receipt History")
    rows = fetch_receipts(user)
    if not rows:
        st.info("No receipts found.")
        return
    for r in rows:
        merchant, date, time, amount, category, was_corrected, uploaded_at, image_path, anomaly_status = r
        displayed = display_receipt(image_path)
        if not displayed:
            st.caption(f"🗂 Stored path: {image_path}")

        st.write(f"**Merchant:** {merchant} | **Date:** {date} | **Time:** {time}")
        st.write(f"**Amount:** {amount} | **Category:** {category} | **Corrected:** {'Yes' if was_corrected else 'No'}")

        try:
            amt_val = float(str(amount).replace("£", "").replace(",", ""))
        except:
            amt_val = 0.0

        if anomaly_status:
            st.info(f"🔎 Anomaly Decision: **{str(anomaly_status).upper()}**")
        elif amt_val > 100:
            st.warning("⚠️ Anomaly review is pending for this receipt.")
        st.divider()

def render_all_receipts():
    st.subheader("📂 All Employee Receipts")
    rows = fetch_receipts()
    if not rows:
        st.info("No receipts found.")
        return
    for r in rows:
        username, merchant, date, time, amount, category, was_corrected, uploaded_at, image_path, anomaly_status = r
        displayed = display_receipt(image_path)
        if not displayed:
            st.caption(f"🗂 Stored path: {image_path}")

        st.write(f"**User:** {username} | **Merchant:** {merchant} | **Date:** {date} | **Time:** {time}")
        st.write(f"**Amount:** {amount} | **Category:** {category} | **Corrected:** {'Yes' if was_corrected else 'No'}")

        if anomaly_status:
            st.info(f"🔎 Admin Decision: **{str(anomaly_status).upper()}**")
        else:
            try:
                amt_val = float(str(amount).replace("£", "").replace(",", ""))
            except:
                amt_val = 0.0
            if amt_val > 100:
                st.warning("⚠️ Anomaly decision pending")
        st.divider()



# Part 6: Analytics (Reviewed‑only, UK dates)

def _normalize_review(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    s = re.sub(r"[^a-z]", "", s)   
    return s if s in {"approved", "rejected"} else None

def render_analytics(user=None):
    import re
    import numpy as np

    st.subheader("📊 Spending Analytics")
    rows = fetch_receipts(user)
    if not rows:
        st.info("No data found.")
        return

    # Column headers from fetch_receipts()
    if user:
        columns = ["Merchant", "Date", "Time", "Amount", "Category", "Corrected", "Uploaded", "Image", "Anomaly"]
    else:
        columns = ["User", "Merchant", "Date", "Time", "Amount", "Category", "Corrected", "Uploaded", "Image", "Anomaly"]

    df = pd.DataFrame(rows, columns=columns)
    st.caption(f"Total rows loaded: {len(df)}")

    # Clean Amount to numeric
    df["AmountVal"] = (
        df["Amount"].astype(str)
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
    )
    df["AmountVal"] = pd.to_numeric(df["AmountVal"], errors="coerce")

    # Normalize anomaly field
    def _norm_anomaly(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        s = re.sub(r'[^a-z]', '', s) 
        return s if s in ("approved", "rejected") else None

    df["AnomalyNorm"] = df["Anomaly"].apply(_norm_anomaly)

    
    THRESHOLD = 100.0
    is_pending = df["Anomaly"].isna() | (df["Anomaly"].astype(str).str.strip().str.lower().isin(["", "none"]))
    pending_high = is_pending & (df["AmountVal"] > THRESHOLD)

    # Keep everything EXCEPT pending-high rows
    df = df[~pending_high].copy()

    # If nothing left, explain why
    if df.empty:
        st.info("No receipts to show (pending high-value receipts are excluded until reviewed).")
        return

    # Parse Date (UK dd/mm/yyyy first) & drop invalid
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date", "AmountVal"]).copy()

    if df.empty:
        st.warning("No valid data to show after cleaning Date/Amount.")
        return

    # Monthly spend 
    df["MonthPeriod"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("MonthPeriod", sort=True)["AmountVal"].sum().sort_index()
    if not monthly.empty:
        fig1, ax1 = plt.subplots()
        ax1.pie(
            monthly.values,
            labels=[p.strftime("%b %Y") for p in monthly.index],
            autopct="%1.1f%%",
            startangle=90
        )
        ax1.set_title("Monthly Spend")
        st.pyplot(fig1)
    else:
        st.info("No monthly data available to display.")

    # Category spend (pie)
    by_cat = df.groupby("Category")["AmountVal"].sum().sort_values(ascending=False)
    if not by_cat.empty:
        fig2, ax2 = plt.subplots()
        ax2.pie(by_cat.values, labels=by_cat.index, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Spend by Category")
        st.pyplot(fig2)
    else:
        st.info("No category data available to display.")



# Anomaly Review & Admin Tools

def render_anomaly():
    st.subheader("🚨 Anomaly Review (High-Value Receipts)")
    flagged = fetch_flagged_receipts()
    if not flagged:
        st.info("No anomalies detected.")
        return

    for row in flagged:
        (rid, username, merchant, date, time, amount, category,
         uploaded_at, image_path, anomaly_status) = row

        try:
            preview = convert_to_image(image_path)
            if preview and os.path.exists(preview):
                st.image(preview, width=250)
            else:
                st.info(f"🗂 File stored at: {image_path}")
        except Exception:
            st.warning(f"Unable to preview image. File stored at: {image_path}")

        st.write(f"**Employee:** {username}")
        st.write(f"**Merchant:** {merchant} | **Date:** {date} | **Time:** {time}")
        st.write(f"**Amount:** {amount} | **Category:** {category}")

        if anomaly_status is None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Approve (#{rid})"):
                    update_anomaly_status_by_id(rid, "approved")
                    insert_anomaly_feedback(username, merchant, date, time, amount, category, "approved", uploaded_at)
                    st.success("Marked as APPROVED.")
                    st.rerun()
            with col2:
                if st.button(f"❌ Reject (#{rid})"):
                    update_anomaly_status_by_id(rid, "rejected")
                    insert_anomaly_feedback(username, merchant, date, time, amount, category, "rejected", uploaded_at)
                    st.warning("Marked as REJECTED.")
                    st.rerun()
        else:
            st.success(f"✔️ Already reviewed: **{anomaly_status.upper()}**")
            
def render_admin_controls():
    import re

    THRESHOLD = 100.0 
    st.subheader("🧑‍💼 View Receipts by Employee")

    employees = fetch_users()
    selected_emp = st.selectbox("Select Employee", employees)
    if not selected_emp:
        return

    rows = fetch_receipts(selected_emp)
    if not rows:
        st.info("No receipts found for this employee.")
        return

    for r in rows:
        merchant, date, time, amount, category, was_corrected, uploaded_at, image_path, anomaly_status = r

        # Preview image if possible
        try:
            preview = convert_to_image(image_path)
            if preview and os.path.exists(preview):
                st.image(preview, width=200)
            else:
                st.info(f"🗂 File stored at: {image_path}")
        except Exception:
            st.warning(f"Unable to preview image. File stored at: {image_path}")

        # Basic details
        st.write(f"**Merchant:** {merchant} | **Date:** {date} | **Time:** {time}")
        st.write(f"**Amount:** {amount} | **Category:** {category} | **Corrected:** {'Yes' if was_corrected else 'No'}")

        # strip currency symbols/commas and keep digits + dot
        amt_str = re.sub(r"[^\d.]", "", str(amount))
        try:
            amt_val = float(amt_str) if amt_str else None
        except ValueError:
            amt_val = None

        # Normalize anomaly_status for checks/display 
        a = ("" if anomaly_status is None else str(anomaly_status)).strip().lower()
        has_decision = a not in ("", "none")  # approved/rejected stored here

       
        # Pending ONLY if no decision yet AND amount exceeds threshold
        needs_review = (not has_decision) and (amt_val is not None and amt_val > THRESHOLD)

        if has_decision:
            st.info(f"🔎 Anomaly Decision: **{str(anomaly_status).upper()}**")
        elif needs_review:
            st.warning("⚠️ Anomaly decision pending")
        else:
            st.success("✅ No anomaly review required")

        st.divider()

    # Danger zone
    st.warning("⚠️ Danger Zone: Delete this employee and all their data")
    if st.button(f"🗑️ Delete {selected_emp} from system", key=f"delete_emp_{selected_emp}"):
        delete_employee_and_data(selected_emp)
        st.success(f"✅ Employee '{selected_emp}' and all related data have been deleted.")
        st.rerun()

def render_retrain_section():
    st.subheader("🔁 Retrain Model from Feedback")
    if st.button("🔄 Retrain Model"):
        try:
            df = load_training_data_from_corrections(get_connection)
            retrained = retrain_model(df, "model_feedback.pkl")
            if retrained:
                st.success("✅ Model retrained and saved successfully.")
            else:
                st.warning("⚠️ No data available for retraining.")
        except Exception as e:
            st.error(f"❌ Failed to retrain model: {e}")

def render_correction_export():
    st.subheader("📤 Export Corrected Receipts for Model Training")
    rows = fetch_corrected_receipts()
    if not rows:
        st.info("No corrected receipts available.")
        return

    df = pd.DataFrame(
        rows,
        columns=["Username", "Merchant", "Date", "Time", "Amount", "Category", "Original Image", "Corrected At"]
    )

    # Convert DB timestamps (assumed UTC) local display time
    try:
        tz_name = _get_secret("APP_TZ", "Europe/London")
    except Exception:
        tz_name = "Europe/London"
    local_tz = ZoneInfo(tz_name)


    ts_utc = pd.to_datetime(df["Corrected At"], errors="coerce", utc=True)
    ts_local = ts_utc.dt.tz_convert(local_tz).dt.tz_localize(None)
    df["Corrected At"] = ts_local

    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name="corrected_receipts.csv",
        mime="text/csv"
    )


# Routing & Logout

def render_logout():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

if st.session_state.page == "landing":
    st.subheader("Select an Action")
    choice = st.radio("Continue as:", ["Login", "Signup (Admin)", "Signup (Employee)"])
    if choice == "Login":
        role = st.radio("Role", ["Admin", "Employee"])
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
        if login_btn:
            actual = check_credentials(username, password)
            if actual == role.lower():
                st.session_state.page = "dashboard"
                st.session_state.user = username
                st.session_state.role = role.lower()
                st.rerun()
            else:
                st.error("Invalid credentials or role mismatch.")
    elif "Signup" in choice:
        role = "admin" if "Admin" in choice else "employee"
        st.subheader(f"🆕 Create {role.title()} Account")
        with st.form("signup_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submit_btn = st.form_submit_button("Create Account")
        if submit_btn:
            if not new_user or not new_pass or not confirm_pass:
                st.error("Please fill all fields.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            else:
                create_user(new_user, new_pass, role)

elif st.session_state.page == "dashboard":
    st.sidebar.success(f"Logged in as: {st.session_state.user} ({st.session_state.role})")
    render_logout()

    if st.session_state.role == "employee":
        menu = st.sidebar.radio("📁 Menu", ["Upload Receipt", "Your Receipt History", "Analytics"])
        if menu == "Upload Receipt":
            render_upload_ui(st.session_state.user)
        elif menu == "Your Receipt History":
            render_receipts(st.session_state.user)
        elif menu == "Analytics":
            render_analytics(st.session_state.user)

    elif st.session_state.role == "admin":
        menu = st.sidebar.radio(
            "🛠 Admin Menu",
            ["Upload Receipt", "Your Receipts", "All Employee Receipts", "Anomaly Review",
             "Analytics", "Manage Employees", "Retrain Model from Corrections", "Export Corrections"]
        )
        if menu == "Upload Receipt":
            render_upload_ui(st.session_state.user)
        elif menu == "Your Receipts":
            render_receipts(st.session_state.user)
        elif menu == "All Employee Receipts":
            render_all_receipts()
        elif menu == "Anomaly Review":
            render_anomaly()
        elif menu == "Analytics":
            render_analytics()
        elif menu == "Manage Employees":
            render_admin_controls()
        elif menu == "Retrain Model from Corrections":
            render_retrain_section()
        elif menu == "Export Corrections":
            render_correction_export()



# One‑click Feedback Model Trainer

def train_feedback_model():
    rows = get_corrected_receipts()
    if not rows:
        st.warning("⚠️ Not enough corrected data to retrain the model.")
        return
    texts, labels = [], []
    for merchant, date, time, amount, category in rows:
        texts.append(f"{merchant} {date} {time} {amount}")
        labels.append(category)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    X = CountVectorizer().fit_transform(texts)
    model = LogisticRegression(max_iter=1000).fit(X, labels)
    joblib.dump(model, "feedback_model.pkl")
    st.success("✅ Feedback model retrained and saved.")
