import streamlit as st
import requests
import json
import re

def show_page():
    # -------------------------------
    # Page Title
    # -------------------------------
    st.markdown("<h1 class='main-title'>🎯 AI-Powered Study Planner</h1>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to the **AI Study Planner** — powered by **Ollama + Streamlit** 💡  
    This tool generates a **personalized weekly study plan** based on your learning goals and available study hours.
    """)

    # -------------------------------
    # Planner Settings
    # -------------------------------
    with st.container():
        st.markdown("### ⚙️ Planner Settings")
        model_name = "gemma2:2b"
        # model_name = "phi3:latest"
        col1, col2 = st.columns(2)
        with col1:
            available_hours = st.slider("📘 Study Hours per Day", 1, 10, 4, key="planner_hours")
        with col2:
            days = st.slider("📅 Days per Week", 3, 7, 5, key="planner_days")
        st.info("💡 *Tip:* Use realistic hours for consistent progress.")

    st.markdown("---")

    # -------------------------------
    # User Inputs
    # -------------------------------
    st.subheader("📘 Tell me about your study goals")
    study_goal = st.text_area(
        "What do you want to achieve? (e.g., Learn Machine Learning in 1 month, Prepare for exams, Revise Python, etc.)",
        placeholder="Type your study goal here...",
        key="planner_goal"
    )

    st.markdown("")

    # -------------------------------
    # Generate Study Plan
    # -------------------------------
    if st.button("🚀 Generate Study Plan", key="planner_generate"):
        if not study_goal.strip():
            st.warning("Please enter your study goal first.")
            st.stop()

        with st.spinner("Generating your personalized study plan... ⏳"):
            try:
                # Prepare Ollama API request
                url = "http://localhost:11434/api/generate"
                headers = {"Content-Type": "application/json"}

                prompt = f"""
                You are an AI Study Planner. 
                Generate a detailed study plan **only in valid JSON** (no extra text, no Markdown).

                User Information:
                Goal: {study_goal}
                Daily Study Hours: {available_hours}
                Days per Week: {days}

                For each day, include:
                [
                {{
                    "day": "Monday",
                    "subject": "Topic Name",
                    "start_time": "9:00 AM",
                    "task_type": "Theory / Practice / Revision",
                    "description": "Brief explanation of what to focus on"
                }}
                ]
                Return ONLY the JSON array. No notes, no explanation, no markdown.
                """

                payload = {"model": model_name, "prompt": prompt, "stream": False}
                response = requests.post(url, headers=headers, json=payload)

                if response.status_code != 200:
                    st.error(f"Error calling Ollama API: {response.text}")
                    st.stop()

                raw_text = response.json().get("response", "").strip()

                # Extract valid JSON safely
                match = re.search(r"\[.*\]", raw_text, re.DOTALL)
                json_text = match.group(0) if match else None

                if json_text:
                    try:
                        plan_data = json.loads(json_text)
                        st.success("✅ Study Plan Generated Successfully!")

                        for day_plan in plan_data:
                            with st.expander(f"📅 {day_plan.get('day', 'Unknown Day')}"):
                                st.markdown(f"""
                                        **Subject:** {day_plan.get('subject', 'N/A')}  
                                        **Time:** {day_plan.get('start_time', 'N/A')}  
                                        **Type:** {day_plan.get('task_type', 'N/A')}  
                                        **Description:** {day_plan.get('description', 'No details provided')}
                                """)
                    except json.JSONDecodeError:
                        st.error("⚠️ Could not parse cleaned JSON. Displaying raw text:")
                        st.code(raw_text)
                else:
                    st.error("⚠️ No valid JSON found in the AI response.")
                    st.code(raw_text)

            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")