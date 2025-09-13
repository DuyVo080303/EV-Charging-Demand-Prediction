import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import timedelta
from tensorflow.keras.models import load_model
import joblib

# ========== PAGE SETUP ==========
TIME_COL = "Date"
ID_COL = "station_id"
TARGET_COL = "estimated_demand_kWh"
EXOG_COLS = ["public_holiday","school_holiday","is_weekend",
             "Avg_Temp","Avg_Humidity","Avg_Wind"]

GLOBAL_MODEL_PATH = "model_gru.keras"         # bạn đã cung cấp
GLOBAL_SCALER_PATH = "scaler_all.joblib"      # bạn đã cung cấp
GLOBAL_TAIL_PATH = "tail.npy"                 # optID_COLional (seed đã scale)
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
        raise ValueError(f"history.csv thiếu cột: {miss}")
    return df

@st.cache_data(show_spinner=False)
def load_station_cluster_map(path: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    if not {"station_id", "geo_cluster"}.issubset(m.columns):
        raise ValueError("station_to_cluster.csv cần cột: station_id, geo_cluster")
    # Ép kiểu và validate
    m["geo_cluster"] = pd.to_numeric(m["geo_cluster"], errors="coerce")
    if m["geo_cluster"].isna().any():
        bad = m[m["geo_cluster"].isna()]["station_id"].tolist()[:10]
        raise ValueError(f"Có NaN ở geo_cluster (ví dụ station_id: {bad})")
    if (m["geo_cluster"] < 0).any():
        bad = m[m["geo_cluster"] < 0]["station_id"].tolist()[:10]
        raise ValueError(f"Có giá trị âm (-1) ở geo_cluster (ví dụ station_id: {bad})")
    # Đảm bảo 1-1
    dup = m.duplicated("station_id", keep=False)
    if dup.any():
        raise ValueError("station_to_cluster.csv có station_id trùng lặp.")
    return m[["station_id", "geo_cluster"]]

def cluster_dir_candidates(cid: int) -> list:
    cid = int(cid)
    return [
        os.path.join("artifacts", "clusters", str(cid)),   # cấu trúc của bạn
        os.path.join("artifacts", f"cluster_{cid}"),       # phòng khi có cấu trúc cũ
    ]

@st.cache_resource(show_spinner=False)
def load_artifacts_for_cluster(geo_cluster: int):
    # Thử các đường dẫn ứng viên
    tried = []
    for cdir in cluster_dir_candidates(geo_cluster):
        mpath = os.path.join(cdir, "model_gru.keras")
        spath = os.path.join(cdir, "scaler_all.joblib")
        tpath = os.path.join(cdir, "tail.npy")
        tried.append((cdir, mpath, spath, tpath))
        if os.path.exists(mpath) and os.path.exists(spath):
            # Debug: liệt kê thư mục dùng
            st.write("✅ Using cluster artifacts from:", cdir)
            try:
                st.write("Contents:", os.listdir(cdir))
            except Exception:
                pass
            model = load_model(mpath)
            scaler = joblib.load(spath)
            tail_scaled = np.load(tpath) if os.path.exists(tpath) else None
            return model, scaler, tail_scaled, model.input_shape[1], model.input_shape[2]

    # Nếu không tìm thấy, in debug rõ ràng rồi raise
    st.write("❌ Could not find artifacts for cluster:", geo_cluster)
    st.write("CWD:", os.getcwd())
    for cdir, mpath, spath, _ in tried:
        st.write("Tried:", cdir,
                 "| model exists?", os.path.exists(mpath),
                 "| scaler exists?", os.path.exists(spath))
        # Thử liệt kê để nhìn thấy thực tế trong deploy
        if os.path.exists(cdir):
            try:
                st.write("Dir contents:", os.listdir(cdir))
            except Exception:
                pass
    raise FileNotFoundError(
        f"Không tìm thấy model/scaler cho cụm {geo_cluster}. "
        f"Yêu cầu các file 'model_gru.keras' và 'scaler.joblib' trong một trong các thư mục: "
        + ", ".join(cluster_dir_candidates(geo_cluster))
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [TIME_COL, ID_COL, TARGET_COL] + EXOG_COLS
    return df[cols].copy()


def build_exog_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trả về ma trận đặc trưng (features) chỉ gồm các external features (exog).
    """
    return df[EXOG_COLS].copy()

def scale_exog_features(exog_df: pd.DataFrame, scaler) -> np.ndarray:
    """
    Scale các external features (exog).
    """
    exog_array = exog_df.to_numpy().astype(float)
    exog_scaled = scaler.transform(exog_array)
    return exog_scaled

def forecast_with_exog(model, scaler, exog_scaled, seed_scaled, horizon):
    """
    Dự báo với mô hình GRU, sử dụng exog features để ảnh hưởng đến dự báo từng bước.
    """
    seq_len, n_feat = seed_scaled.shape
    seq = seed_scaled.copy()
    out_scaled = []

    for t in range(horizon):
        # Sử dụng exog cho bước tiếp theo
        exog_input = exog_scaled[t]  # lấy dòng t-th từ exog features
        # Tạo đầu vào mới cho mô hình (chuỗi seed + exog)
        next_input = np.append(seq[-seq_len:, 0], exog_input)  # (seq_len + exog)
        x = next_input.reshape(1, seq_len + 1, n_feat)  # reshape để phù hợp với input của GRU

        # Dự báo target (yhat_scaled)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]
        # Ghép bước tiếp theo vào chuỗi
        next_vec = np.empty((n_feat,), dtype=float)
        next_vec[0] = yhat_scaled  # target
        next_vec[1:] = exog_input  # exog (đã được scaled)
        seq = np.vstack([seq, next_vec])  # thêm giá trị mới vào chuỗi

        out_scaled.append(yhat_scaled)

    dummy = np.zeros((horizon, n_feat))
    dummy[:, 0] = np.array(out_scaled)
    inv = scaler.inverse_transform(dummy)[:, 0]
    return inv

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
df_hist = load_history(hist_path)
map_df  = load_station_cluster_map(map_path)

# Check if target column exists
if TARGET_COL not in df_hist.columns:
    st.error(f"Không tìm thấy cột {TARGET_COL} trong history.csv")
    st.stop()

# Đồng bộ kiểu station_id giữa 2 file
df_hist[ID_COL] = pd.to_numeric(df_hist[ID_COL], errors="raise")
map_df["station_id"] = pd.to_numeric(map_df["station_id"], errors="raise")

stations = sorted(df_hist[ID_COL].unique().tolist())
station_id = st.selectbox("Station", stations)

row = map_df.loc[map_df["station_id"] == station_id, "geo_cluster"]
if row.empty:
    st.error(f"Không tìm thấy geo_cluster cho station_id={station_id} trong station_to_cluster.csv")
    st.stop()
geo_cluster = int(row.iloc[0])
st.write(f"**Cluster:** `{geo_cluster}` • **Station:** `{station_id}`")

model, scaler, tail_scaled_opt, SEQ_LEN, N_FEAT = load_artifacts_for_cluster(geo_cluster)

df_feat = build_feature_matrix(df_hist)

# Nếu tail.npy đã scale đúng cách và shape khớp thì dùng, ngược lại tự dựng từ history
if tail_scaled_opt is not None and tail_scaled_opt.shape == (SEQ_LEN, N_FEAT):
    seed_scaled = tail_scaled_opt
else:
    seed_scaled = take_last_sequence_scaled(df_feat, station_id, SEQ_LEN, scaler)

# Lấy exog hiện tại làm template + override từ sidebar
last_row = (df_feat[df_feat[ID_COL] == station_id].tail(1)).iloc[0]
overrides = {
    "public_holiday": int(ph),
    "school_holiday": int(sh),
    "is_weekend": int(we),
    "Avg_Temp": float(t_avg),
    "Avg_Humidity": float(h_avg),
    "Avg_Wind": float(w_avg),
}

# Tạo exog features mới cho tương lai
future_exog = make_future_exog_overrides(last_row, horizon, overrides)

# Scale exog features cho tương lai
exog_future_scaled = scale_future_exog(future_exog, scaler, N_FEAT)

# ==========/ FORECAST ==========
yhat = forecast_with_exog(model, scaler, exog_future_scaled, seed_scaled, horizon)

# ==========/ PLOT ==========
hist_tail = df_hist[df_hist[ID_COL] == station_id].sort_values(TIME_COL).tail(SEQ_LEN).copy()
t0 = hist_tail[TIME_COL].iloc[-1]
freq = infer_freq(hist_tail[TIME_COL])
future_times = [t0 + (i+1)*freq for i in range(horizon)]

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

df_plot = pd.concat([df_plot_hist, df_plot_fcst], ignore_index=True)

chart = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("timestamp:T", title="Time"),
    y=alt.Y("value:Q", title="Demand (kWh)"),
    color=alt.Color("type:N", sort=["History","Forecast"])
).properties(width="container", height=380, title=f"Station {station_id} — GRU Forecast ({horizon} steps)")

st.altair_chart(chart, use_container_width=True)

# ==========/ EXPORT ==========
with st.expander("Export"):
    st.download_button(
        "Download Forecast CSV",
        data=df_plot_fcst.to_csv(index=False),
        file_name=f"forecast_station_{station_id}.csv",
        mime="text/csv"
    )
