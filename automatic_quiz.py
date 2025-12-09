import streamlit as st
import os
import re
import time
import gdown
import smtplib
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from fpdf import FPDF

def show_page():
    st.title("🤖 Auto Quiz Generator")

    st.markdown("""
    ### How it works:
    1️⃣ Paste your **Google Drive public folder link (Anyone can view)**  
    2️⃣ Enter your **Gmail + App Password**  
    3️⃣ Enter **Recipient Email**  
    4️⃣ App downloads docs → Gemma2 generates quizzes 🧠 → Converts to PDF → Emails them 📬
    """)

    folder_link = st.text_input("🔗 Google Drive Folder Link (Anyone with link can view)")
    sender_email = st.text_input("📧 Your Gmail")
    app_password = st.text_input("🔑 Your Gmail App Password", type="password")
    recipient_email = st.text_input("🎯 Recipient Email")

    if st.button("🚀 Generate & Send Quizzes"):
        if not all([folder_link, sender_email, app_password, recipient_email]):
            st.warning("⚠️ Please fill all fields.")
            return

        try:
            # -----------------------------
            # Extract Google Drive folder ID
            # -----------------------------
            match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_link)
            if not match:
                st.error("❌ Invalid folder link. Please paste correct format.")
                return
            folder_id = match.group(1)

            st.info("📂 Fetching files from Google Drive folder...")
            os.makedirs("downloads", exist_ok=True)

            file_links = gdown.download_folder(
                id=folder_id,
                output="downloads",
                quiet=True,
                use_cookies=False
            )

            if not file_links:
                st.error("❌ No downloadable files found. Make sure folder is shared publicly.")
                return

            st.success(f"✅ Found {len(file_links)} file(s). Generating quizzes...")

            # -----------------------------
            # Process each file
            # -----------------------------
            for i, file_path in enumerate(file_links, start=1):
                filename = os.path.basename(file_path)
                st.write(f"🤖 Generating quiz for: {filename} ...")

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text_data = f.read().strip()

                    # -----------------------------
                    # Generate quiz using Ollama (quantized model)
                    # -----------------------------
                    prompt = f"""
                    You are a teacher. Read the following document and create a short quiz:
                    - 5 Multiple Choice Questions (MCQs) with 4 options each (A–D)
                    - 3 Short answer questions
                    Keep it concise and clear.
                    Document content:
                    {text_data}
                    """

                    process = subprocess.Popen(
                        ["ollama", "run", "gemma2:2b"],
                        # ["ollama", "run", "phi3:latest"],
                        # ["ollama", "run", "llama3"],
                        # ["ollama", "run", "llama3:8b-instruct-q4_K_M"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    output, error = process.communicate(prompt)

                    if process.returncode != 0:
                        st.error(f"⚠️ Ollama Error for {filename}: {error}")
                        continue

                    quiz_text = output.strip()

                    # -----------------------------
                    # Convert to PDF (safe encoding)
                    # -----------------------------
                    pdf_name = f"quiz_{i}.pdf"
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    quiz_text_clean = quiz_text.replace("∞", "infinity").replace("±", "+/-").replace("≥", ">=").replace("≤", "<=")
                    safe_text = quiz_text_clean.encode("latin-1", "replace").decode("latin-1")
                    pdf.multi_cell(0, 10, safe_text)
                    pdf.output(pdf_name, "F")
                    # -----------------------------
                    # Email the generated quiz
                    # -----------------------------
                    msg = MIMEMultipart()
                    msg["From"] = sender_email
                    msg["To"] = recipient_email
                    msg["Subject"] = f"📘 AI Quiz Generated: {filename}"

                    msg.attach(MIMEText(
                        "Attached is the AI-generated quiz based on your document.", 
                        "plain", "utf-8"
                    ))

                    with open(pdf_name, "rb") as f:
                        attach = MIMEApplication(f.read(), _subtype="pdf")
                        attach.add_header("Content-Disposition", "attachment", filename=pdf_name)
                        msg.attach(attach)

                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                        server.login(sender_email, app_password)
                        server.send_message(msg)

                    st.success(f"✅ Quiz generated and sent for: {filename}")
                    os.remove(pdf_name)
                    time.sleep(1.5)

                except Exception as inner_e:
                    st.error(f"❌ Error processing {filename}: {inner_e}")

            st.success("🎉 All quizzes generated and sent successfully!")

        except Exception as e:
            st.error(f"❌ Fatal Error: {e}")
    st.markdown("---")