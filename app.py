from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -------------------------------------------------------
# Page configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -------------------------------------------------------
# Custom styling
# -------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #172554 0%, transparent 35%),
                linear-gradient(135deg, #07111f 0%, #0f172a 55%, #111827 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.2rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            background: linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.30),
                rgba(15, 23, 42, 0.75)
            );
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .hero-description {
            color: #cbd5e1;
            font-size: 1.05rem;
            margin: 0;
        }

        .section-title {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .result-card {
            padding: 1.7rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(15, 23, 42, 0.82);
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.22);
        }

        .low-result {
            border-left: 7px solid #22c55e;
        }

        .high-result {
            border-left: 7px solid #ef4444;
        }

        .result-title {
            color: #ffffff;
            font-size: 1.55rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .result-text {
            color: #cbd5e1;
            font-size: 1rem;
        }

        div.stButton > button,
        div.stFormSubmitButton > button {
            width: 100%;
            height: 3.2rem;
            border: none;
            border-radius: 13px;
            color: white;
            font-size: 1.05rem;
            font-weight: 700;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
            transition: 0.2s;
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            transform: translateY(-2px);
            color: white;
            border: none;
        }

        [data-testid="stMetric"] {
            padding: 1rem;
            border-radius: 15px;
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------
# Load model artifacts
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    app_folder = Path(__file__).resolve().parent

    loaded_model = joblib.load(
        app_folder / "knn_heart_model.pkl"
    )
    loaded_scaler = joblib.load(
        app_folder / "heart_scaler.pkl"
    )
    loaded_columns = joblib.load(
        app_folder / "heart_columns.pkl"
    )

    return loaded_model, loaded_scaler, loaded_columns


try:
    model, scaler, expected_columns = load_artifacts()
except FileNotFoundError as error:
    st.error(f"Required model file was not found: {error}")
    st.stop()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">❤️ Heart Disease Predictor</div>
        <p class="hero-description">
            A machine-learning project that estimates a heart-disease
            classification from the information entered below.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This application is an educational ML project. Its result is not a "
    "medical diagnosis and should not replace professional medical advice."
)


# -------------------------------------------------------
# Input form
# -------------------------------------------------------
st.markdown(
    '<div class="section-title">Enter health information</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    left_column, right_column = st.columns(2, gap="large")

    with left_column:
        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

        sex = st.selectbox(
            "Sex",
            options=["Male", "Female"]
        )

        chest_pain = st.selectbox(
            "Chest pain type",
            options=["ASY", "ATA", "NAP", "TA"],
            help=(
                "ASY: Asymptomatic, ATA: Atypical angina, "
                "NAP: Non-anginal pain, TA: Typical angina"
            )
        )

        resting_bp = st.slider(
            "Resting blood pressure (mm Hg)",
            min_value=80,
            max_value=200,
            value=120
        )

        cholesterol = st.slider(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting blood sugar above 120 mg/dL",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No"
        )

    with right_column:
        resting_ecg = st.selectbox(
            "Resting ECG result",
            options=["Normal", "ST", "LVH"]
        )

        max_hr = st.slider(
            "Maximum heart rate achieved",
            min_value=60,
            max_value=220,
            value=150
        )

        exercise_angina = st.selectbox(
            "Exercise-induced angina",
            options=["N", "Y"],
            format_func=lambda value: "Yes" if value == "Y" else "No"
        )

        oldpeak = st.slider(
            "Oldpeak / ST depression",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "Peak exercise ST slope",
            options=["Flat", "Up", "Down"]
        )

        st.write("")
        st.write("")
        submitted = st.form_submit_button(
            "Run Heart Disease Prediction"
        )


# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
if submitted:

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_M": int(sex == "Male"),

        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA": int(chest_pain == "TA"),

        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST": int(resting_ecg == "ST"),

        "ExerciseAngina_Y": int(exercise_angina == "Y"),

        "ST_Slope_Flat": int(st_slope == "Flat"),
        "ST_Slope_Up": int(st_slope == "Up")
    }

    input_df = pd.DataFrame([raw_input])

    # Match the columns and order used during training
    input_df = input_df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])

    # KNN supports predict_proba
    model_score = float(
        model.predict_proba(scaled_input)[0][1] * 100
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Prediction result</div>',
        unsafe_allow_html=True
    )

    result_column, gauge_column = st.columns(
        [1, 1.25],
        gap="large"
    )

    with result_column:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card high-result">
                    <div class="result-title">
                        Higher-risk classification
                    </div>
                    <div class="result-text">
                        The model classified this input as belonging to
                        the heart-disease class.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card low-result">
                    <div class="result-title">
                        Lower-risk classification
                    </div>
                    <div class="result-text">
                        The model classified this input as belonging to
                        the non-heart-disease class.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")
        metric_one, metric_two = st.columns(2)

        metric_one.metric(
            "Predicted class",
            "Heart disease" if prediction == 1 else "No heart disease"
        )

        metric_two.metric(
            "Model score",
            f"{model_score:.1f}%"
        )

        st.caption(
            "The model score is based on neighbouring samples used by KNN. "
            "It is not a clinically validated probability."
        )

    with gauge_column:
        gauge_color = "#ef4444" if prediction == 1 else "#22c55e"

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=model_score,
                number={"suffix": "%", "font": {"color": "white"}},
                title={
                    "text": "Model heart-disease score",
                    "font": {"color": "#cbd5e1"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#cbd5e1"
                    },
                    "bar": {"color": gauge_color},
                    "bgcolor": "#1e293b",
                    "bordercolor": "#475569",
                    "steps": [
                        {"range": [0, 40], "color": "#14532d"},
                        {"range": [40, 70], "color": "#854d0e"},
                        {"range": [70, 100], "color": "#7f1d1d"}
                    ]
                }
            )
        )

        gauge.update_layout(
            height=320,
            margin=dict(l=25, r=25, t=70, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # Input summary graph
    st.markdown(
        '<div class="section-title">Numeric input summary</div>',
        unsafe_allow_html=True
    )

    summary_data = pd.DataFrame(
        {
            "Measurement": [
                "Age",
                "Resting BP",
                "Cholesterol",
                "Maximum HR",
                "Oldpeak"
            ],
            "Value": [
                age,
                resting_bp,
                cholesterol,
                max_hr,
                oldpeak
            ]
        }
    )

    summary_chart = go.Figure(
        go.Bar(
            x=summary_data["Measurement"],
            y=summary_data["Value"],
            marker={
                "color": [
                    "#60a5fa",
                    "#818cf8",
                    "#a78bfa",
                    "#34d399",
                    "#f59e0b"
                ],
                "line": {"width": 0}
            },
            text=summary_data["Value"],
            textposition="outside"
        )
    )

    summary_chart.update_layout(
        height=360,
        xaxis_title=None,
        yaxis_title="Entered value",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#e2e8f0"},
        margin=dict(l=30, r=20, t=30, b=30)
    )

    summary_chart.update_xaxes(showgrid=False)
    summary_chart.update_yaxes(
        gridcolor="rgba(148,163,184,0.15)"
    )

    st.plotly_chart(
        summary_chart,
        use_container_width=True,
        config={"displayModeBar": False}
    )