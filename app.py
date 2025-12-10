import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF

# -----------------------------
# TensorFlow: optional (local only)
# -----------------------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

# Try to import SHAP, but allow app to run without it
try:
    import shap
except Exception:
    shap = None

# -----------------------------
# Paths & Constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
LSTM_PATH = BASE_DIR / "models" / "lstm_fd001_best.h5"
GRU_PATH = BASE_DIR / "models" / "gru_fd001_best.h5"  # optional, if you trained it

SEQ_LEN = 30

# Column names for NASA CMAPSS FD001 format
COL_NAMES = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

# Same features used in Week 2 & Week 3
USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14"
]
FEATURE_COLUMNS = USEFUL_SENSORS + ["setting_1", "setting_2", "setting_3"]  # 12 features


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource
def load_model(path: Path):
    """
    Load a Keras model when TensorFlow is available.
    In cloud/demo mode (no TF), this should not be called.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available in this environment.")
    return tf.keras.models.load_model(path)


@st.cache_data
def load_sample_fd001() -> pd.DataFrame:
    """Load sample NASA FD001 test set from local repo."""
    test_path = DATA_DIR / "test_FD001.txt"
    df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COL_NAMES)
    return df


def preprocess_engine_from_df(engine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a single-engine DataFrame and scale the selected features.
    """
    engine_df = engine_df.sort_values("cycle").reset_index(drop=True)
    scaler = MinMaxScaler()
    scaled_engine = engine_df.copy()
    scaled_engine[FEATURE_COLUMNS] = scaler.fit_transform(engine_df[FEATURE_COLUMNS])
    return scaled_engine


def make_sequences(df: pd.DataFrame, features: list[str], seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Convert a time-ordered DataFrame into overlapping sequences for LSTM/GRU input.
    Output shape: (num_sequences, seq_len, num_features)
    """
    values = df[features].values
    X = []
    for i in range(len(values) - seq_len + 1):
        X.append(values[i:i + seq_len])
    return np.array(X)


def get_engine_status(rul: float) -> str:
    """
    Map RUL value to a health status string.
    """
    if rul > 50:
        return "🟢 Healthy — No maintenance needed"
    elif rul > 20:
        return "🟡 Warning — Schedule maintenance"
    else:
        return "🔴 Critical — Maintenance required now"


def failure_probability(rul: float, max_rul: float = 150.0) -> float:
    """
    Simple failure probability heuristic based on RUL.
    Closer to zero RUL -> higher probability.
    """
    prob = 1.0 - (rul / max_rul)
    return float(max(0.0, min(1.0, prob)))


def simple_baseline_rul(eng_df: pd.DataFrame) -> float:
    """
    Very simple baseline RUL: assume failure around 200 cycles.
    Used when TensorFlow/deep model is not available (Streamlit Cloud).
    """
    last_cycle = eng_df["cycle"].max()
    return max(0.0, 200.0 - float(last_cycle))


def plot_sensor_trends(engine_df: pd.DataFrame, engine_id: int):
    """
    Plot key sensor trends over cycles for a given engine.
    """
    fig, ax = plt.subplots()
    for sensor in ["sensor_2", "sensor_3", "sensor_7", "sensor_11"]:
        if sensor in engine_df.columns:
            ax.plot(engine_df["cycle"], engine_df[sensor], label=sensor)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Sensor value")
    ax.set_title(f"Sensor trends for Engine {engine_id}")
    ax.legend()
    st.pyplot(fig)


def compute_shap_for_engine(model, scaled_engine: pd.DataFrame):
    """
    Compute SHAP values for the last sequence of an engine.
    Returns feature names and aggregated SHAP importance values.
    """
    if shap is None or not TF_AVAILABLE:
        return None, None

    X = make_sequences(scaled_engine, FEATURE_COLUMNS, SEQ_LEN)
    if X.shape[0] == 0:
        return None, None

    last_seq = X[-1:]  # shape (1, seq_len, features)
    background = X[:50] if X.shape[0] > 50 else X

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(last_seq)[0]  # (seq_len, features)

    # Aggregate absolute SHAP across time dimension
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return FEATURE_COLUMNS, mean_abs_shap


def compute_rul_timeline(engine_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    For a single engine, compute predicted RUL at each time step
    (using sliding sequences), to visualize degradation over time.
    Only valid when a deep model is available.
    """
    if not TF_AVAILABLE:
        return pd.DataFrame()

    engine_df = engine_df.sort_values("cycle").reset_index(drop=True)
    scaled_engine = preprocess_engine_from_df(engine_df)
    X = make_sequences(scaled_engine, FEATURE_COLUMNS, SEQ_LEN)
    if X.shape[0] == 0:
        return pd.DataFrame()

    preds = model.predict(X, verbose=0).ravel()
    # Predictions align with cycles starting at SEQ_LEN
    cycles = engine_df["cycle"].values[SEQ_LEN - 1:]
    out = pd.DataFrame({"cycle": cycles[: len(preds)], "predicted_RUL": preds})
    return out


def build_engine_report_pdf(
    engine_id: int,
    sim_cycle: int,
    predicted_rul: float,
    failure_prob: float,
    status_text: str,
) -> bytes:
    """
    Create a simple 1-page PDF report summarizing engine health.
    Returns raw PDF bytes for download.
    FPDF default fonts are NOT Unicode, so we strip emojis and fancy characters.
    """
    # ---- sanitize status text (remove emojis and fancy dash) ----
    safe_status = status_text
    safe_status = safe_status.replace("🟢", "HEALTHY")
    safe_status = safe_status.replace("🟡", "WARNING")
    safe_status = safe_status.replace("🔴", "CRITICAL")
    safe_status = safe_status.replace("—", "-")  # replace long dash with normal dash

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Aircraft Engine Health Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Engine ID: {engine_id}", ln=True)
    pdf.cell(0, 8, f"Data up to cycle: {sim_cycle}", ln=True)
    pdf.cell(0, 8, f"Predicted Remaining Useful Life: {predicted_rul:.2f} cycles", ln=True)
    pdf.cell(0, 8, f"Failure Probability: {failure_prob * 100:.1f} %", ln=True)
    pdf.multi_cell(0, 8, f"Health Status: {safe_status}")

    pdf.ln(5)
    pdf.multi_cell(
        0,
        6,
        "Recommendation: Plan maintenance according to airline policy and "
        "use this report together with engineering judgement."
    )

    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(
        0,
        6,
        "Generated by: Goitom Abirha - Data Scientist (Predictive Maintenance)",
        ln=True,
    )

    # In some fpdf versions, output(dest="S") returns a bytearray, not str.
    result = pdf.output(dest="S")
    if isinstance(result, str):
        pdf_bytes = result.encode("latin-1")
    else:
        pdf_bytes = bytes(result)  # bytearray or already bytes

    return pdf_bytes


@st.cache_data
def compute_fleet_rul(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Compute RUL + status + failure probability for all engines in the dataset.
    - If TensorFlow + models are available: use LSTM/GRU predictions.
    - Otherwise: use a simple baseline RUL based on last cycle.
    """
    use_deep_model = False
    model = None

    if TF_AVAILABLE:
        if model_name == "GRU" and GRU_PATH.exists():
            model = load_model(GRU_PATH)
            use_deep_model = True
        elif LSTM_PATH.exists():
            model = load_model(LSTM_PATH)
            use_deep_model = True

    records = []
    for eid in sorted(df["engine_id"].unique()):
        eng_df = df[df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        if len(eng_df) < SEQ_LEN:
            continue

        if use_deep_model:
            scaled_engine = preprocess_engine_from_df(eng_df)
            X = make_sequences(scaled_engine, FEATURE_COLUMNS, SEQ_LEN)
            if X.shape[0] == 0:
                continue

            pred = model.predict(X, verbose=0)
            rul_raw = float(pred[-1])
            rul = max(0.0, rul_raw)  # clamp at 0 so we don't show negative RUL
        else:
            # Baseline: no TF available
            rul = simple_baseline_rul(eng_df)

        status = get_engine_status(rul)
        prob = failure_probability(rul)

        records.append({
            "engine_id": eid,
            "last_cycle": int(eng_df["cycle"].max()),
            "predicted_RUL": rul,
            "failure_probability": prob,
            "status": status,
        })

    fleet_df = pd.DataFrame(records).sort_values("predicted_RUL")
    return fleet_df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    layout="wide"
)

st.title("✈️ Aircraft Engine Predictive Maintenance Dashboard")
st.markdown("**Developed by: Goitom Abirha — Data Scientist (Predictive Maintenance & Aviation Analytics)**")

st.markdown(
    """
This dashboard uses deep learning models (**LSTM / GRU**) trained on the NASA **CMAPSS FD001** dataset  
to predict **Remaining Useful Life (RUL)** and visualize the health of a fleet of turbofan engines.
"""
)

if not TF_AVAILABLE:
    st.info(
        "Running in **demo mode** (Streamlit Cloud): TensorFlow is not available here.\n\n"
        "- Fleet view uses a simple baseline RUL estimate based on last cycle.\n"
        "- Full LSTM/GRU deep learning predictions are available in your local `tf` conda environment."
    )

# -----------------------------
# Sidebar: data source & model choice
# -----------------------------
st.sidebar.header("Data Source")

st.sidebar.markdown("---")
st.sidebar.markdown("**About this project**")
st.sidebar.markdown(
    "Built by **Goitom Abirha**\n\n"
    "Personal portfolio project focused on airline / MRO predictive maintenance "
    "using NASA CMAPSS turbofan data."
)
st.sidebar.markdown("---")

data_option = st.sidebar.radio(
    "Choose data source:",
    ("Use sample NASA FD001 test data", "Upload your own file")
)

if data_option == "Use sample NASA FD001 test data":
    df = load_sample_fd001()
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CMAPSS-format engine file (.txt or .csv)",
        type=["txt", "csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
        if df.shape[1] == len(COL_NAMES):
            df.columns = COL_NAMES
        else:
            st.error(f"Uploaded file has {df.shape[1]} columns, expected {len(COL_NAMES)}.")
            st.stop()
    else:
        st.warning("Please upload a file or select the sample dataset.")
        st.stop()

if not set(["engine_id", "cycle"]).issubset(df.columns):
    st.error("Data does not contain required 'engine_id' and 'cycle' columns.")
    st.stop()

# Model selection (if GRU model available)
available_models = ["LSTM"]
if GRU_PATH.exists():
    available_models.append("GRU")

model_choice = st.sidebar.radio("Select model", available_models)

# -----------------------------
# Main layout: tabs for Fleet vs Engine
# -----------------------------
tab_fleet, tab_engine = st.tabs(["🌍 Fleet Overview", "🔍 Single Engine Detail"])

# =============================
# TAB 1: Fleet Overview
# =============================
with tab_fleet:
    st.subheader(f"Fleet RUL Overview ({model_choice} model)")

    fleet_df = compute_fleet_rul(df, model_choice)

    if fleet_df.empty:
        st.warning("Not enough cycles per engine to compute RUL sequences.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Engine RUL Table")
            display_df = fleet_df.copy()
            display_df["predicted_RUL"] = display_df["predicted_RUL"].round(2)
            display_df["failure_probability"] = (display_df["failure_probability"] * 100).round(1)
            display_df.rename(columns={
                "predicted_RUL": "RUL (cycles)",
                "failure_probability": "Failure probability (%)"
            }, inplace=True)
            st.dataframe(display_df, use_container_width=True)

        with col2:
            st.markdown("#### RUL by Engine (sorted)")
            chart_df = fleet_df.sort_values("predicted_RUL")
            fig = px.bar(
                chart_df,
                x="predicted_RUL",
                y="engine_id",
                orientation="h",
                labels={"predicted_RUL": "RUL (cycles)", "engine_id": "Engine ID"},
                title="Engines by Remaining Useful Life",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown(
            "Engines at the **top of the table** have the **lowest RUL** and should be prioritized for maintenance."
        )

        # Top risk engines (lowest RUL)
        st.markdown("#### Top 5 Highest-Risk Engines")
        top_n = 5
        top_risk = fleet_df.nsmallest(top_n, "predicted_RUL").copy()
        top_risk["predicted_RUL"] = top_risk["predicted_RUL"].round(2)
        top_risk["failure_probability"] = (top_risk["failure_probability"] * 100).round(1)
        top_risk.rename(columns={
            "predicted_RUL": "RUL (cycles)",
            "failure_probability": "Failure probability (%)"
        }, inplace=True)
        st.table(top_risk[["engine_id", "last_cycle", "RUL (cycles)", "Failure probability (%)", "status"]])

# =============================
# TAB 2: Single Engine Detail
# =============================
with tab_engine:
    st.subheader("Single Engine View")

    engine_ids = sorted(df["engine_id"].unique().tolist())
    selected_engine = st.selectbox("Select Engine ID", engine_ids)

    engine_df = df[df["engine_id"] == selected_engine].sort_values("cycle").reset_index(drop=True)

    # Simulation slider: predict as of an earlier cycle if desired
    max_cycle_available = int(engine_df["cycle"].max())
    min_allowed_cycle = int(engine_df["cycle"].min()) + SEQ_LEN

    if max_cycle_available <= min_allowed_cycle:
        st.warning(
            f"Engine {selected_engine} has only {max_cycle_available} cycles — "
            f"not enough room for simulation. Using full data."
        )
        sim_cycle = max_cycle_available
    else:
        sim_cycle = st.slider(
            "Simulate engine state up to cycle",
            min_value=min_allowed_cycle,
            max_value=max_cycle_available,
            value=max_cycle_available,
            step=1,
        )

    engine_sim_df = engine_df[engine_df["cycle"] <= sim_cycle].reset_index(drop=True)
    st.caption(f"Using data up to cycle {sim_cycle} for prediction.")

    st.markdown(f"### Engine {selected_engine} — Raw Data (first 10 rows)")
    st.dataframe(engine_sim_df.head(10), use_container_width=True)

    with st.expander("Show sensor trend plot"):
        plot_sensor_trends(engine_sim_df, selected_engine)

    # Ensure required feature columns exist
    missing_features = [col for col in FEATURE_COLUMNS if col not in engine_sim_df.columns]
    if missing_features:
        st.error(f"Missing required feature columns: {missing_features}")
        st.stop()

    # ---------- Deep model available (local) ----------
    if TF_AVAILABLE and (LSTM_PATH.exists() or GRU_PATH.exists()):
        # Choose which model to use for this engine
        if model_choice == "GRU" and GRU_PATH.exists():
            model = load_model(GRU_PATH)
        else:
            model = load_model(LSTM_PATH)

        scaled_engine = preprocess_engine_from_df(engine_sim_df)
        X_sequences = make_sequences(scaled_engine, FEATURE_COLUMNS, SEQ_LEN)

        if X_sequences.shape[0] == 0:
            st.error(
                f"Engine {selected_engine} does not have at least {SEQ_LEN} cycles "
                "up to the selected simulation point. Cannot create sequences."
            )
        else:
            rul_pred = model.predict(X_sequences, verbose=0)
            predicted_rul_raw = float(rul_pred[-1])
            predicted_rul = max(0.0, predicted_rul_raw)
            prob = failure_probability(predicted_rul)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Remaining Useful Life", f"{predicted_rul:.2f} cycles")

            with col2:
                st.metric("Failure Probability", f"{prob * 100:.1f} %")

            with col3:
                st.markdown("### Health Status")
                st.write(get_engine_status(predicted_rul))

            st.markdown("---")
            st.markdown(
                f"Model input shape for this engine: `{X_sequences.shape[0]} sequences × "
                f"{SEQ_LEN} cycles × {len(FEATURE_COLUMNS)} features`"
            )

            # ---------- PDF report download ----------
            report_bytes = build_engine_report_pdf(
                engine_id=selected_engine,
                sim_cycle=sim_cycle,
                predicted_rul=predicted_rul,
                failure_prob=prob,
                status_text=get_engine_status(predicted_rul),
            )
            st.download_button(
                label="📄 Download Engine Health PDF",
                data=report_bytes,
                file_name=f"engine_{selected_engine}_health_report.pdf",
                mime="application/pdf",
            )

            # ---------- RUL timeline ----------
            with st.expander("View predicted RUL degradation over time"):
                timeline_df = compute_rul_timeline(engine_sim_df, model)
                if timeline_df.empty:
                    st.write("Not enough data to build a RUL timeline.")
                else:
                    fig_line = px.line(
                        timeline_df,
                        x="cycle",
                        y="predicted_RUL",
                        labels={"cycle": "Cycle", "predicted_RUL": "Predicted RUL (cycles)"},
                        title="Predicted Remaining Useful Life over time",
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

            # ---------- Feature Importance (SHAP with safe fallback) ----------
            with st.expander("Explain prediction"):
                try:
                    feature_names, shap_scores = compute_shap_for_engine(model, scaled_engine)

                    if feature_names is None:
                        raise Exception("SHAP not available")

                    shap_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": shap_scores
                    }).sort_values("importance", ascending=True)

                    fig_imp, ax_imp = plt.subplots(figsize=(5, 6))
                    ax_imp.barh(shap_df["feature"], shap_df["importance"])
                    ax_imp.set_xlabel("Mean |SHAP value| (impact on RUL)")
                    ax_imp.set_title("Feature importance for this engine")
                    st.pyplot(fig_imp)

                except Exception:
                    st.warning("SHAP unavailable — showing approximate feature importance instead.")

                    try:
                        # Universal fallback: variance-based feature importance
                        last_seq = X_sequences[-1]  # shape: (seq_len, num_features)
                        variances = np.var(last_seq, axis=0)

                        fi_df = pd.DataFrame({
                            "feature": FEATURE_COLUMNS,
                            "importance": variances
                        }).sort_values("importance", ascending=True)

                        fig_fi, ax_fi = plt.subplots(figsize=(5, 6))
                        ax_fi.barh(fi_df["feature"], fi_df["importance"])
                        ax_fi.set_xlabel("Feature variance (proxy importance)")
                        ax_fi.set_title("Feature importance (variance-based fallback)")
                        st.pyplot(fig_fi)

                    except Exception as e:
                        st.error(f"Could not compute fallback feature importance: {e}")

    # ---------- Demo mode (no deep model) ----------
    else:
        st.warning(
            "Deep learning models (LSTM/GRU) are not available in this environment.\n\n"
            "Showing baseline RUL estimate using last cycle instead."
        )

        predicted_rul = simple_baseline_rul(engine_sim_df)
        prob = failure_probability(predicted_rul)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Estimated Remaining Useful Life", f"{predicted_rul:.2f} cycles")

        with col2:
            st.metric("Failure Probability (baseline)", f"{prob * 100:.1f} %")

        with col3:
            st.markdown("### Health Status (baseline)")
            st.write(get_engine_status(predicted_rul))

        st.markdown("---")
        st.markdown(
            "Baseline mode uses a simple rule-of-thumb estimate around 200 cycles to approximate RUL. "
            "For full LSTM/GRU predictions, run this app locally with TensorFlow installed."
        )

        report_bytes = build_engine_report_pdf(
            engine_id=selected_engine,
            sim_cycle=sim_cycle,
            predicted_rul=predicted_rul,
            failure_prob=prob,
            status_text=get_engine_status(predicted_rul),
        )
        st.download_button(
            label="📄 Download Engine Health PDF (baseline)",
            data=report_bytes,
            file_name=f"engine_{selected_engine}_health_report_baseline.pdf",
            mime="application/pdf",
        )

    # ---------- Anomaly Detection (works in both modes) ----------
    with st.expander("Detect sensor anomalies (z-score)"):
        window = st.slider("Rolling window size", min_value=10, max_value=100, value=30, step=5)
        z_threshold = st.slider("Z-score threshold", min_value=2.0, max_value=5.0, value=3.0, step=0.5)

        sensors_to_check = ["sensor_2", "sensor_3", "sensor_7", "sensor_11"]
        anomalies = []

        for sensor in sensors_to_check:
            if sensor not in engine_sim_df.columns:
                continue

            rolling_mean = engine_sim_df[sensor].rolling(window=window).mean()
            rolling_std = engine_sim_df[sensor].rolling(window=window).std()

            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1e-6)

            z = (engine_sim_df[sensor] - rolling_mean) / rolling_std
            mask = z.abs() > z_threshold

            for idx in engine_sim_df[mask].index:
                anomalies.append({
                    "cycle": int(engine_sim_df.loc[idx, "cycle"]),
                    "sensor": sensor,
                    "value": float(engine_sim_df.loc[idx, sensor]),
                    "z_score": float(z.loc[idx])
                })

        if anomalies:
            anom_df = pd.DataFrame(anomalies).sort_values("cycle")
            st.write("Detected anomalies:")
            st.dataframe(anom_df, use_container_width=True)
        else:
            st.write("No anomalies detected with current settings.")
