import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ========== PAGE SETUP ==========
TIME_COL = "Date"
ID_COL = "station_id"
TARGET_COL = "estimated_demand_kWh"
EXOG_COLS = ["public_holiday", "school_holiday", "is_weekend", "Avg_Temp", "Avg_Humidity", "Avg_Wind"]

GLOBAL_MODEL_PATH = "model_gru.keras"         # your model path
GLOBAL_SCALER_PATH = "scaler_all.joblib"      # your scaler path
GLOBAL_TAIL_PATH = "tail.npy"                 # optional seed for the model
CLUSTER_ARTIFACT_ROOT = "artifacts"

st.set_page_config(page_title="EVAT — GRU Forecast by Station", page_icon="⚡", layout="wide")
st.title("⚡ EVAT — GRU Forecast per Cluster / Station")
st.caption("Select a station, adjust external factors → get a **forecast line chart**.")

# ========== UTILITY FUNCTIONS ==========
@st.cache_data(show_spinner=False)
def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    needed = [TIME_COL, ID_COL, TARGET_COL] + EXOG_COLS
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"history.csv missing columns: {miss}")
    return df

@st.cache_data(show_spinner=False)
def load_station_cluster_map(path: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    if not {"station_id", "geo_cluster"}.issubset(m.columns):
        raise ValueError("station_to_cluster.csv must contain station_id and geo_cluster columns")
    m["geo_cluster"] = pd.to_numeric(m["geo_cluster"], errors="coerce")
    if m["geo_cluster"].isna().any():
        bad = m[m["geo_cluster"].isna()]["station_id"].tolist()[:10]
        raise ValueError(f"NaN values in geo_cluster (example station_id: {bad})")
    if (m["geo_cluster"] < 0).any():
        bad = m[m["geo_cluster"] < 0]["station_id"].tolist()[:10]
        raise ValueError(f"Negative values in geo_cluster (example station_id: {bad})")
    dup = m.duplicated("station_id", keep=False)
    if dup.any():
        raise ValueError("Duplicate station_id found in station_to_cluster.csv")
    return m[["station_id", "geo_cluster"]]

def cluster_dir_candidates(cid: int) -> list:
    cid = int(cid)
    return [
        os.path.join("artifacts", "clusters", str(cid)),
        os.path.join("artifacts", f"cluster_{cid}"),
    ]

@st.cache_resource(show_spinner=False)
def load_artifacts_for_cluster(geo_cluster: int):
    tried = []
    for cdir in cluster_dir_candidates(geo_cluster):
        mpath = os.path.join(cdir, "model_gru.keras")
        spath = os.path.join(cdir, "scaler_all.joblib")
        tpath = os.path.join(cdir, "tail.npy")
        tried.append((cdir, mpath, spath, tpath))
        if os.path.exists(mpath) and os.path.exists(spath):
            model = load_model(mpath)
            scaler = joblib.load(spath)
            tail_scaled = np.load(tpath) if os.path.exists(tpath) else None
            return model, scaler, tail_scaled, model.input_shape[1], model.input_shape[2]
    raise FileNotFoundError(f"Model/scaler not found for cluster {geo_cluster}.")

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [TIME_COL, ID_COL, TARGET_COL] + EXOG_COLS
    return df[cols].copy()

def scale_future_exog(future_exog_df: pd.DataFrame, scaler, n_feat: int) -> np.ndarray:
    ex = future_exog_df[EXOG_COLS].to_numpy().astype(float)
    ex_scaled = _scale_matrix_like_training(ex, scaler)
    return ex_scaled

def make_future_exog_overrides(base_row: pd.Series, horizon: int, overrides: dict) -> pd.DataFrame:
    rows = []
    for _ in range(horizon):
        r = {c: base_row.get(c, np.nan) for c in EXOG_COLS}
        r.update(overrides)
        rows.append(r)
    return pd.DataFrame(rows)

def _scale_matrix_like_training(mat: np.ndarray, scaler) -> np.ndarray:
    h, w = mat.shape
    flat = mat.reshape(-1, 1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(h, w)

def forecast_with_exog(model, scaler, exog_scaled, seed_scaled, horizon):
    """
    Dự báo với mô hình GRU, sử dụng exog features để ảnh hưởng đến dự báo từng bước.
    """
    seq_len, n_feat = seed_scaled.shape  # seq_len là độ dài chuỗi, n_feat là số đặc trưng
    seq = seed_scaled.copy()  # Sao chép seed_scaled để cập nhật dần
    out_scaled = []  # Danh sách để lưu kết quả dự báo

    for t in range(horizon):
        exog_input = exog_scaled[t]  # Lấy dòng t-th từ exog features
        # Tạo đầu vào mới cho mô hình (chuỗi seed + exog)
        next_input = np.append(seq[-seq_len:, 0], exog_input)  # Ghép seed và exog lại với nhau

        # Ensure next_input is reshaped properly for the model (batch_size, seq_len, n_feat)
        x = next_input.reshape(1, seq_len, n_feat)  # reshaping để phù hợp với input của GRU

        # Dự báo target (yhat_scaled)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]  # lấy giá trị dự báo đầu tiên

        # Ghép bước tiếp theo vào chuỗi
        next_vec = np.empty((n_feat,), dtype=float)  # Tạo một mảng rỗng cho vector tiếp theo
        next_vec[0] = yhat_scaled  # Đặt giá trị dự báo vào vị trí đầu tiên
        next_vec[1:] = exog_input  # Đặt exog vào phần còn lại của next_vec
        seq = np.vstack([seq, next_vec])  # Thêm giá trị tiếp theo vào chuỗi

        out_scaled.append(yhat_scaled)  # Lưu giá trị dự báo vào out_scaled

    # Tạo mảng dummy để chuyển đổi về giá trị ban đầu
    dummy = np.zeros((horizon, n_feat))  # Khởi tạo mảng rỗng cho dummy
    dummy[:, 0] = np.array(out_scaled)  # Đặt giá trị dự báo vào cột đầu tiên của dummy
    inv = scaler.inverse_transform(dummy)[:, 0]  # Đảo ngược giá trị dự báo bằng scaler
    return inv  # Trả về kết quả dự báo đã được đảo ngược



# ==========/ SIDEBAR ==========
st.sidebar.subheader("Data paths")
hist_path = st.sidebar.text_input("history.csv", "history.csv")
map_path  = st.sidebar.text_input("station_to_cluster.csv", "station_to_cluster.csv")
horizon   = st.sidebar.number_input("Horizon (steps)", min_value=6, max_value=24*7, value=24, step=6)

st.sidebar.subheader("External factors (override)")
ph = st.sidebar.selectbox("Public holiday", [0, 1], index=0)
sh = st.sidebar.selectbox("School holiday", [0, 1], index=0)
we = st.sidebar.selectbox("Weekend", [0, 1], index=0)
t_avg = st.sidebar.slider("Avg_Temp (°C)", -5.0, 45.0, 24.0, 0.5)
h_avg = st.sidebar.slider("Avg_Humidity (%)", 0.0, 100.0, 60.0, 1.0)
w_avg = st.sidebar.slider("Avg_Wind (m/s)", 0.0, 20.0, 3.0, 0.2)

# ==========/ LOAD ==========
# ========== Load and preprocess data ==========
df_hist = load_history("history.csv")
map_df = load_station_cluster_map("station_to_cluster.csv")

# Check if target column exists
if TARGET_COL not in df_hist.columns:
    st.error(f"Không tìm thấy cột {TARGET_COL} trong history.csv")
    st.stop()

# Đồng bộ kiểu station_id giữa 2 file
df_hist[ID_COL] = pd.to_numeric(df_hist[ID_COL], errors="raise")
map_df["station_id"] = pd.to_numeric(map_df["station_id"], errors="raise")

# Select station ID
stations = sorted(df_hist[ID_COL].unique().tolist())
station_id = st.selectbox("Station", stations)

# Get geo_cluster for the selected station
row = map_df.loc[map_df["station_id"] == station_id, "geo_cluster"]
if row.empty:
    st.error(f"Không tìm thấy geo_cluster cho station_id={station_id}")
    st.stop()
geo_cluster = int(row.iloc[0])
st.write(f"**Cluster:** `{geo_cluster}` • **Station:** `{station_id}`")

# Load model, scaler, and other artifacts
model, scaler, tail_scaled_opt, SEQ_LEN, N_FEAT = load_artifacts_for_cluster(geo_cluster)

# Build feature matrix for historical data
df_feat = build_feature_matrix(df_hist)

# If tail.npy exists and matches the required shape, use it, else build from history
if tail_scaled_opt is not None and tail_scaled_opt.shape == (SEQ_LEN, N_FEAT):
    seed_scaled = tail_scaled_opt
else:
    seed_scaled = take_last_sequence_scaled(df_feat, station_id, SEQ_LEN, scaler)

# Get current exog template and override with sidebar inputs
last_row = df_feat[df_feat[ID_COL] == station_id].tail(1).iloc[0]
overrides = {
    "public_holiday": int(ph),
    "school_holiday": int(sh),
    "is_weekend": int(we),
    "Avg_Temp": float(t_avg),
    "Avg_Humidity": float(h_avg),
    "Avg_Wind": float(w_avg),
}

# Generate exog features for the forecast horizon
future_exog = make_future_exog_overrides(last_row, horizon, overrides)

# Scale the exog features for the future
exog_future_scaled = scale_future_exog(future_exog, scaler, N_FEAT)

# ==========/ Forecast with the model and exog features ==========
yhat = forecast_with_exog(model, scaler, exog_future_scaled, seed_scaled, horizon)

# ==========/ Plotting the results ==========
# Prepare historical and forecast data for plotting
hist_tail = df_hist[df_hist[ID_COL] == station_id].sort_values(TIME_COL).tail(SEQ_LEN).copy()
t0 = hist_tail[TIME_COL].iloc[-1]
freq = infer_freq(hist_tail[TIME_COL])
future_times = [t0 + (i+1) * freq for i in range(horizon)]

df_plot_hist = pd.DataFrame({
    "timestamp": hist_tail[TIME_COL],
    "value": hist_tail[TARGET_COL],
    "type": "History"
})

df_plot_fcst = pd.DataFrame({
    "timestamp": future_times,
    "value": yhat,
    "type": "Forecast"
})

# Combine historical and forecast data for plotting
df_plot = pd.concat([df_plot_hist, df_plot_fcst], ignore_index=True)

# Create the plot
chart = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("timestamp:T", title="Time"),
    y=alt.Y("value:Q", title="Demand (kWh)"),
    color=alt.Color("type:N", sort=["History", "Forecast"])
).properties(width="container", height=380, title=f"Station {station_id} — GRU Forecast ({horizon} steps)")

# Display the chart
st.altair_chart(chart, use_container_width=True)

# ==========/ Export the forecast to CSV ==========
with st.expander("Export"):
    st.download_button(
        "Download Forecast CSV",
        data=df_plot_fcst.to_csv(index=False),
        file_name=f"forecast_station_{station_id}.csv",
        mime="text/csv"
    )

