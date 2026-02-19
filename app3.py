import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Multiclass Clinical Predictor", layout="wide")
st.markdown("""
<style>

/* ===============================
   MAIN BACKGROUND
=============================== */
.stApp {
    background: linear-gradient(135deg, #1CB5E0, #000851);
    color: #e5e7eb;
}

/* Remove top white header */
header {
    background: transparent !important;
}

/* ===============================
   FILE UPLOADER CONTAINER
=============================== */
div[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

/* ===============================
   UNIVERSAL BUTTON STYLE
   (Predict + File Upload + Others)
=============================== */
.stButton > button,
div[data-testid="stFileUploader"] button {

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    min-width: 140px !important;
    height: 40px !important;
    padding: 0 22px !important;

    background: linear-gradient(135deg, #14b8a6, #0ea5e9) !important;
    color: white !important;

    border-radius: 10px !important;
    border: none !important;

    font-size: 14px !important;
    font-weight: 600 !important;

    white-space: nowrap !important;
    overflow: hidden !important;

    transition: all 0.25s ease !important;
}

/* Target exact element with higher specificity */
div[data-testid="stFileUploader"] 
section[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Remove inner instruction wrapper background */
div[data-testid="stFileUploader"] 
section[data-testid="stFileUploaderDropzone"] 
div[data-testid="stFileUploaderDropzoneInstructions"] {
    background: transparent !important;
}

/* Remove any inline style background */
section[data-testid="stFileUploaderDropzone"][style] {
    background: transparent !important;
    background-color: transparent !important;
}         
/* Hover */
.stButton > button:hover,
div[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #0ea5e9, #14b8a6) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(20,184,166,0.35);
}

/* Active Click */
.stButton > button:active,
div[data-testid="stFileUploader"] button:active {
    transform: scale(0.96);
}

/* ===============================
   FORCE SINGLE CLEAN BORDER
=============================== */

/* OUTER NUMBER INPUT WRAPPER */
div[data-testid="stNumberInput"] > div {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    transition: all 0.25s ease;
}

/* REMOVE RED ERROR STATE COMPLETELY */
div[data-testid="stNumberInput"][aria-invalid="true"] > div,
div[data-testid="stNumberInput"] div[aria-invalid="true"] {
    border: 1px solid #1f2937 !important;
    box-shadow: none !important;
}

/* INNER BASEWEB INPUT */
div[data-baseweb="input"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* HOVER */
div[data-testid="stNumberInput"]:hover > div {
    border: 1px solid #14b8a6 !important;
}

/* FOCUS */
div[data-testid="stNumberInput"]:focus-within > div {
    border: 0.1px solid #14b8a6 !important;
    box-shadow: 0 0 0 2px rgba(20,184,166,0.25) !important;
}

/* REMOVE ANY DEFAULT OUTLINES */
div[data-testid="stNumberInput"] *,
div[data-baseweb="input"] * {
    outline: none !important;
}

/* INPUT TEXT */
input {
    color: #f9fafb !important;
    font-size: 14px !important;
}

/* ===============================
   +/- BUTTONS CLEAN
=============================== */
button[kind="secondary"] {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #cbd5e1 !important;
    border-radius: 6px !important;
    height: 30px !important;
    width: 30px !important;
    transition: 0.2s ease;
}

button[kind="secondary"]:hover {
    background: #14b8a6 !important;
    color: white !important;
    border: 1px solid #14b8a6 !important;
}

/* ===============================
   RADIO BUTTON FIX
=============================== */
div[role="radiogroup"] label {
    color: linear-gradient(135deg, #0ea5e9, #14b8a6) !important;!important;
}

div[role="radiogroup"] label:hover {
    color: #14b8a6 !important;
}

/* ===============================
   LABEL STYLE
=============================== */
label {
    color: #9ca3af !important;
    font-size: 13px !important;
}
            
/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #111827) !important;
}

</style>
""", unsafe_allow_html=True)



# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    with open("xgboost_multiclass_model2.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder2.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model()

# -------------------------
# Header
# -------------------------
st.markdown('<div class="big-title">Clinical Multiclass Prediction System</div>', unsafe_allow_html=True)
st.write("Real time decision support for febrile illnesses")

# -------------------------
# Feature configuration
# -------------------------
binary_features = [
    "breathlessness", "cough", "joint_pain", "seizures",
    "myalgia", "vomiting", "rash", "jaundice", "eschar"
]

# -------------------------
# Sidebar
# -------------------------
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Prediction", "Batch Prediction"]
)

# =========================
# SINGLE PREDICTION
# =========================
if mode == "Single Prediction":

    st.markdown('<div class="section-title">Patient Input</div>', unsafe_allow_html=True)

    try:
        feature_names = model.get_booster().feature_names
    except:
        st.error("Feature names not found in model")
        st.stop()

    input_data = {}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(feature_names):

        col = [col1, col2, col3][i % 3]

        with col:
            if feature in binary_features:
                val = st.selectbox(
                    feature.replace("_", " ").title(),
                    ["No", "Yes"]
                )
                input_data[feature] = 1 if val == "Yes" else 0
            else:
                input_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=0.0
                )

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.markdown(
            f'<div class="pred-box">Predicted Disease: {predicted_label}</div>',
            unsafe_allow_html=True
        )

        prob_df = pd.DataFrame({
            "Disease": label_encoder.classes_,
            "Probability": probabilities[0]
        }).sort_values("Probability", ascending=False)

        st.bar_chart(
            prob_df.set_index("Disease")
        )

# =========================
# BATCH PREDICTION
# =========================
if mode == "Batch Prediction":

    st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)
    st.write("Upload CSV with trained feature columns")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview")
        st.dataframe(input_df.head())

        if st.button("Run Batch Prediction"):

            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)

            input_df["Prediction"] = label_encoder.inverse_transform(predictions)

            prob_df = pd.DataFrame(
                probabilities,
                columns=[f"Prob_{c}" for c in label_encoder.classes_]
            )

            final_df = pd.concat([input_df, prob_df], axis=1)

            st.success("Batch prediction completed")
            st.dataframe(final_df)

            csv = final_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results",
                data=csv,
                file_name="clinical_predictions.csv",
                mime="text/csv"
            )