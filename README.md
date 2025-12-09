# Ollama-RAG-AI-Study-Assistant 🎓

**Ollama-RAG-AI-Study-Assistant** is a comprehensive Streamlit web application that helps students learn smarter by leveraging AI. The app includes 6 interactive features to assist in document Q&A, quiz generation, lecture exploration, live quizzes, personalized study planning, and concept map creation.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [License](#license)

---

## **Project Overview**
The AI Study Assistant is designed to help learners save time and gain insights from their learning material using AI models like Gemma2 and Ollama. Users can interact with documents, generate quizzes, explore lectures, plan study schedules, and visualize knowledge through concept maps.

---

## **Features**

### **Page 1: Chatbot (Document Q&A)**
- Upload **PDF, PPTX, DOCX, TXT** files.
- Ask questions directly from uploaded documents.
- AI provides **context-aware answers**.

### **Page 2: Auto Quiz Generator**
- Input your **email, Google Drive folder link, password, and receiver email**.
- AI reads the documents in the folder.
- Generates **quizzes for all documents**.
- Converts quizzes into **PDF** and emails them to the receiver.

### **Page 3: Explore Lectures**
- Input a **keyword** (e.g., Machine Learning).
- Fetches **top 20 YouTube videos** for the keyword.
- Extracts **20 comments per video**.
- Performs **sentiment analysis** on comments.
- Displays **top 10 videos** based on positive sentiment.

### **Page 4: Live Quiz**
- Upload documents.
- Model generates a **multiple-choice quiz** on screen.
- Users can **solve quizzes directly**.
- Generates **results instantly**.

### **Page 5: AI-Powered Study Planner**
- Enter **study hours per day**, **days per week**, and **your study goals**.
- Model generates a **personalized weekly study plan**.
- Helps organize time efficiently for effective learning.

### **Page 6: Concept Map Generator**
- Enter a **topic or paragraph**.
- Model generates an **interactive concept map** of key concepts and relationships.
- Users can **download the map as HTML** for offline viewing.

---

## **Installation**

1. Create and activate a virtual environment (recommended):
```bash
conda create -n env_name python=3.11 -y
conda activate env_name
```
1. Install dependencies:
```bash
pip install -r requirements.txt
```
1. Run the app:
```bash
streamlit run app.py
```
## AI-Study-Assistant/
DSTT-Term-Project/
-  app.py # Main Streamlit app
-  automatic_quiz.py # Auto Quiz Generator page
-  explore_lectures.py # Explore Lectures page
-  live_quiz.py # Live Quiz page
-  study_planner.py # AI Study Planner page
-  concept_map.py # Concept Map Generator page
-  client_secrets.json # Google API credentials (for Auto Quiz Generator)
-  assets/ # Static assets like images
   - logo.png # App logo
-  docs/ # Folder to store uploaded documents (runtime)
-  requirements.txt # Project dependencies



---

## License

MIT License

Copyright (c) 2025 Muhammad Hassan Saboor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


