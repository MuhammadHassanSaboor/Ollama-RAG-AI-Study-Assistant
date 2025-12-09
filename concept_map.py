import streamlit as st
import networkx as nx
from pyvis.network import Network
import ollama
import json
import re
import tempfile

def show_page():
    # ---------------------------
    # Page Title
    # ---------------------------
    st.markdown("<h1 class='main-title'>🧠 Concept Map Generator</h1>", unsafe_allow_html=True)
    st.markdown("""
        Type any topic or paragraph below and generate an **interactive concept map**  
        showing key ideas and relationships. Fully offline using **Ollama Gemma2:2B**.
        """)

    # ---------------------------
    # User Input
    # ---------------------------
    prompt = st.text_area(
        "Enter a topic or description",
        height=150,
        placeholder="Example: Explain how photosynthesis works",
        key="concept_map_prompt"
    )

    generate_btn = st.button("✨ Generate Concept Map", key="concept_map_generate")

    # ---------------------------
    # Helper: Clean & extract JSON safely
    # ---------------------------
    def extract_json_from_text(text):
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return None
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r"(\w+):", r'"\1":', json_str)
            json_str = json_str.replace("'", '"')
            try:
                return json.loads(json_str)
            except Exception:
                return None

    # ---------------------------
    # Helper: Get concepts from Ollama
    # ---------------------------
    def extract_concepts_ollama(user_prompt: str):
        system_instruction = """
            You are a concept map generator.
            Extract up to 15 important concepts (nodes) and their relationships (edges)
            from the provided text or topic.

            Return ONLY a valid JSON object with this exact structure:
            {
            "nodes": [{"id": "Concept1"}, {"id": "Concept2"}],
            "edges": [{"source": "Concept1", "target": "Concept2", "label": "relation"}]
            }
            Do not add explanations, markdown, or any text before or after the JSON.
        """
        full_prompt = system_instruction + "\n\nTopic:\n" + user_prompt
        try:
            response = ollama.chat(
                model="gemma2:2b",
                # model="llama2.1",
                # model="mistral:latest",
                # model="phi3:latest",
                messages=[{"role": "user", "content": full_prompt}]
            )
            output = response['message']['content'].strip()
            data = extract_json_from_text(output)

            if not data:
                repair_prompt = f"Fix and output only valid JSON for this text:\n{output}"
                repair = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": repair_prompt}])
                data = extract_json_from_text(repair['message']['content'])

            if not data:
                st.error("⚠️ Could not parse valid JSON output. Try simplifying your topic.")
                data = {"nodes": [], "edges": []}

        except Exception as e:
            st.error(f"❌ Error generating concept map: {e}")
            data = {"nodes": [], "edges": []}

        return data

    # ---------------------------
    # Helper: Render interactive PyVis graph
    # ---------------------------
    def render_concept_map(data):
        G = nx.DiGraph()
        for node in data.get("nodes", []):
            G.add_node(node["id"])
        for edge in data.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            label = edge.get("label", "")
            if src and tgt:
                G.add_edge(src, tgt, title=label)

        net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
        node_shapes = ["box", "ellipse", "circle", "diamond", "triangle"]
        category_colors = ["#5DADE2", "#58D68D", "#F5B041", "#AF7AC5", "#F1948A"]
        num_nodes = len(G.nodes())
        max_size = 35
        min_size = 20

        for i, node in enumerate(G.nodes()):
            size = max_size - (i * (max_size - min_size) / max(1, num_nodes-1))
            net.add_node(
                node,
                label=node,
                shape=node_shapes[i % len(node_shapes)],
                size=size,
                color=category_colors[i % len(category_colors)],
                font={"size": 18, "face": "arial", "color": "black"}
            )

        for u, v, d in G.edges(data=True):
            net.add_edge(
                u,
                v,
                title=d.get("title", ""),
                color="#555555",
                width=3,
                arrows="to",
                smooth={"type": "curvedCW", "roundness": 0.3}
            )

        net.set_options("""
        {
        "layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "directed"}},
        "edges": {"smooth": {"type": "cubicBezier"}},
        "physics": {"enabled": false}
        }
        """)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            html_path = tmp.name

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=700, scrolling=True)
        st.download_button("⬇️ Download Concept Map (HTML)", html_content, file_name="concept_map.html")

    # ---------------------------
    # Main logic
    # ---------------------------
    if generate_btn and prompt.strip():
        with st.spinner("Generating concept map using Gemma 2B... please wait"):
            data = extract_concepts_ollama(prompt)

        if data.get("nodes"):
            st.success(f"Generated {len(data['nodes'])} concepts!")
            render_concept_map(data)
        else:
            st.warning("No valid concepts found. Try simplifying your topic.") 