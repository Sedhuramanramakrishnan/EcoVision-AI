import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from collections import Counter

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="EcoVision AI | Smart Sustainability",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- AESTHETIC DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --forest: #285943;
    --green: #467C5B;
    --sage: #E6EEE2;
    --cream: #F8F7F0;
    --ink: #263A2E;
    --muted: #718074;
}

.stApp {
    background: linear-gradient(145deg, #FAF9F3 0%, #F0F5EC 100%);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(248, 249, 242, 0.85);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E4EDE0, #F3F5EB);
    border-right: 1px solid #D8E3D4;
}

[data-testid="stSidebar"] * {
    color: #294C37;
}

.hero {
    background: linear-gradient(120deg, #244C37, #497B59);
    border-radius: 25px;
    padding: 34px 35px;
    color: #FFFFFF;
    margin-bottom: 25px;
    box-shadow: 0 12px 30px rgba(40, 89, 67, 0.13);
}

.hero h1 {
    font-family: 'Playfair Display', Georgia, serif;
    color: #FFFFFF !important;
    font-size: clamp(30px, 4vw, 43px);
    margin: 8px 0;
}

.hero p {
    color: #E1EDE1 !important;
    font-size: 15px;
    line-height: 1.7;
    max-width: 680px;
}

.eyebrow {
    color: #D7E8BE;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-size: 11px;
    font-weight: 700;
}

.section-heading {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 25px;
    color: #294C37;
    margin: 18px 0 5px;
}

.section-caption {
    color: #718074;
    font-size: 13px;
    margin-bottom: 18px;
}

.tip-card {
    background: #FFFFFF;
    border: 1px solid #E1E9DC;
    border-left: 4px solid #5D8965;
    border-radius: 12px;
    padding: 15px 17px;
    margin: 9px 0;
}

.footer {
    text-align: center;
    padding: 25px 10px 5px;
    color: #819080;
    font-size: 12px;
}

.stButton > button {
    background: #285943;
    color: white;
    border: 1px solid #285943;
    border-radius: 12px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #3B7653;
    color: white;
    border-color: #3B7653;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7);
    border: 1px dashed #9CB49A;
    border-radius: 16px;
    padding: 12px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E0E8DB;
    border-radius: 15px;
    padding: 16px;
}

@media (max-width: 640px) {
    .hero {
        padding: 23px 20px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


# ---------- SORTING GUIDANCE ----------
waste_tips = {
    "bottle": "Check whether the bottle is plastic or glass. Empty it and follow your local recycling rules.",
    "cup": "Check the material. Reuse it where possible and verify whether your local facility accepts it.",
    "cell phone": "Consider an authorised electronic-waste collection centre.",
    "book": "Reuse or donate it, or recycle it if accepted locally.",
    "banana": "Food scraps may be suitable for composting where available.",
    "apple": "Food scraps may be suitable for composting where available.",
    "orange": "Food scraps may be suitable for composting where available.",
    "plastic bag": "Reuse it if possible. Check local plastic-film collection rules.",
    "can": "Empty it and check local metal-recycling requirements.",
    "wine glass": "Check local glass-disposal rules. Special glass may not be accepted with ordinary bottles."
}


# ---------- SIDEBAR NAVIGATION ----------
with st.sidebar:
    st.markdown("## 🌿 EcoVision AI")
    st.caption("SMART SUSTAINABILITY LAB")
    st.divider()
    st.markdown("### Navigation")

    page = st.radio(
        "Choose a page",
        [
            "🏡 Image Analysis",
            "📊 Eco Dashboard",
            "♻️ Sorting Guidance",
            "📥 Export Report"
        ],
        key="navigation_page",
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### About the project")
    st.write(
        "EcoVision AI uses computer vision to detect everyday objects "
        "and provide general waste-sorting guidance."
    )
    st.markdown("### Technology")
    st.caption("🧠 YOLOv8 — Object detection")
    st.caption("🐍 Python — Application logic")
    st.caption("🎨 Streamlit — Web interface")
    st.caption("📊 Pandas — Data and reports")
    st.divider()
    st.caption("HackZen 2.0 · Computer Vision Track")


# ---------- LOAD SAVED RESULTS ----------
saved_rows = st.session_state.get("eco_rows", [])
has_results = len(saved_rows) > 0

if has_results:
    df = pd.DataFrame(saved_rows)
    counts = Counter(df["Object"])


# ---------- IMAGE ANALYSIS PAGE ----------
if page == "🏡 Image Analysis":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">TECHNOLOGY FOR A GREENER TOMORROW</div>
        <h1>See objects.<br>Think sustainably.</h1>
        <p>
            Welcome to EcoVision AI — an intelligent visual assistant
            that identifies everyday objects and helps you explore
            responsible disposal choices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Image Analysis</div>'
        '<div class="section-caption">Upload a photograph to begin your AI-powered analysis.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image from your computer",
        type=["jpg", "jpeg", "png"],
        key="eco_image_uploader"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your uploaded photograph", use_container_width=True)

            if st.button("🌿 Analyse Image", type="primary"):
                with st.spinner("Analysing your image..."):
                    model = load_model()
                    result = model.predict(image, conf=0.35)[0]
                    rows = []

                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        name = result.names[class_id]
                        confidence = float(box.conf[0])
                        rows.append({
                            "Object": name,
                            "Confidence (%)": round(confidence * 100, 2)
                        })

                    st.session_state["eco_result_image"] = result.plot()
                    st.session_state["eco_rows"] = rows

                st.rerun()
        except Exception as error:
            st.error("Image analysis failed.")
            st.exception(error)

    rows = st.session_state.get("eco_rows", [])
    if rows:
        st.markdown("### 🌿 Analysis Results")
        detected_image = st.session_state.get("eco_result_image")
        if detected_image is not None:
            st.image(
                detected_image,
                channels="BGR",
                caption="Detected objects",
                use_container_width=True
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    elif "eco_rows" in st.session_state:
        st.warning("No objects detected. Try another clear photograph.")


# ---------- ECO DASHBOARD PAGE ----------
elif page == "📊 Eco Dashboard":
    st.title("TEST - ECO DASHBOARD")

    if has_results:
        total = len(df)
        unique = len(counts)
        average_confidence = df["Confidence (%)"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Objects Detected", total)
        col2.metric("Object Categories", unique)
        col3.metric("Average Confidence", f"{average_confidence:.1f}%")

        st.markdown("### Object Frequency")
        chart_data = pd.DataFrame(
            counts.items(), columns=["Object", "Count"]
        ).set_index("Object")
        st.bar_chart(chart_data)

        st.markdown("### Detection Details")
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No analysis results yet. Open Image Analysis and analyse a photograph first.")


# ---------- SORTING GUIDANCE PAGE ----------
elif page == "♻️ Sorting Guidance":
    st.markdown(
        '<div class="section-heading">♻️ Sorting Guidance</div>'
        '<div class="section-caption">Explore disposal suggestions for detected objects.</div>',
        unsafe_allow_html=True
    )

    if has_results:
        for name, count in counts.items():
            tip = waste_tips.get(
                name.lower(),
                "The material is not confirmed by the model. Check the item's composition and local disposal rules."
            )
            st.markdown(
                f"""
                <div class="tip-card">
                    <strong>🌱 {name.title()}</strong>
                    <span style="color:#718074;"> · {count} detected</span>
                    <p style="margin:8px 0 0;color:#526457;">{tip}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.info("Object detection cannot confirm material composition. Always check local disposal rules.")
    else:
        st.info("Analyse an image first to receive sorting suggestions.")


# ---------- EXPORT REPORT PAGE ----------
elif page == "📥 Export Report":
    st.markdown(
        '<div class="section-heading">📥 Export Report</div>'
        '<div class="section-caption">Download your latest object detection results.</div>',
        unsafe_allow_html=True
    )

    if has_results:
        st.dataframe(df, hide_index=True, use_container_width=True)
        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download CSV Report",
            data=csv_data,
            file_name="ecovision_detection_report.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("The report contains detected object names and confidence scores. Confidence is not the same as model accuracy.")
    else:
        st.info("Analyse an image first to create a report.")


# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    🌿 EcoVision AI · HackZen 2.0<br>
    Computer Vision for Environmental Awareness
</div>
""", unsafe_allow_html=True)