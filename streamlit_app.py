# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from tensorflow.keras.models import load_model
import joblib

# ===================== PAGE SETUP =====================
st.set_page_config(page_title="EVAT — GRU Forecast by Cluster", page_icon="⚡", layout="wide")
st.title("⚡ EVAT — GRU Forecast per Cluster")
st.caption("Chọn cụm (0–4), điều chỉnh external factors → xem dự báo.")

TIME_COL     = "Date"
CLUSTER_COL  = "geo_cluster"
TARGET_COL   = "estimated_demand_kWh"
EXOG_COLS    = ["public_holiday","school_holiday","is_weekend",
                "Avg_Temp","Avg_Humidity","Avg_Wind"]

EXPECTED_FEATS = 1 + len(EXOG_COLS)  # 7

# ===================== UTILS =====================
@st.cache_data(show_spinner=False)
def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.sort_values([CLUSTER_COL, TIME_COL]).reset_index(drop=True)
    needed = [TIME_COL, CLUSTER_COL, TARGET_COL] + EXOG_COLS
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"`{path}` thiếu cột: {miss}")
    df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce")
    if df[CLUSTER_COL].isna().any():
        raise ValueError(f"Có NaN ở `{CLUSTER_COL}` trong {path}.")
    if (df[CLUSTER_COL] < 0).any():
        raise ValueError(f"Có giá trị âm ở `{CLUSTER_COL}` (vd -1).")
    return df

def cluster_dir(cid: int) -> str:
    p1 = os.path.join("artifacts", "clusters", str(cid))
    p2 = os.path.join("artifacts", f"cluster_{cid}")
    return p1 if os.path.isdir(p1) else p2

def has_artifacts(cid: int) -> bool:
    cdir = cluster_dir(cid)
    return os.path.exists(os.path.join(cdir, "model_gru.keras")) and \
           os.path.exists(os.path.join(cdir, "scaler_all.joblib"))

def artifact_version_key(geo_cluster: int) -> float:
    """Dùng mtime làm key để cache tự refresh khi bạn cập nhật file."""
    cdir = cluster_dir(geo_cluster)
    mtimes = []
    for p in ["model_gru.keras", "scaler_all.joblib", "tail.npy"]:
        f = os.path.join(cdir, p)
        if os.path.exists(f):
            mtimes.append(os.path.getmtime(f))
    return max(mtimes) if mtimes else 0.0

@st.cache_resource(show_spinner=False)
def load_artifacts_for_cluster(geo_cluster: int, version_key: float):
    """version_key chỉ để bẻ cache; không dùng bên trong."""
    cdir = cluster_dir(geo_cluster)
    mpath = os.path.join(cdir, "model_gru.keras")
    spath = os.path.join(cdir, "scaler_all.joblib")
    tpath = os.path.join(cdir, "tail.npy")
    if not (os.path.exists(mpath) and os.path.exists(spath)):
        raise FileNotFoundError(f"Thiếu model/scaler ở {cdir}")
    model = load_model(mpath)
    scaler = joblib.load(spath)
    tail_scaled = np.load(tpath) if os.path.exists(tpath) else None
    seq_len = model.input_shape[1]
    n_feat  = model.input_shape[2]
    return model, scaler, tail_scaled, seq_len, n_feat

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[[TIME_COL, CLUSTER_COL, TARGET_COL] + EXOG_COLS].copy()

def _scale_matrix_like_training(mat: np.ndarray, scaler) -> np.ndarray:
    """Ưu tiên scaler 7 cột; fallback legacy 1-cột nếu cần."""
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in == mat.shape[1]:
        return scaler.transform(mat)
    # legacy 1-cột
    h, w = mat.shape
    flat = mat.reshape(-1, 1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(h, w)

def _inverse_vector_like_training(vec: np.ndarray, scaler) -> np.ndarray:
    """Inverse cho TARGET theo kiểu scaler đã fit."""
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in and n_in > 1:
        return (vec - scaler.min_[0]) / scaler.scale_[0]
    flat = vec.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(-1)

def make_future_exog_overrides(base_row: pd.Series, horizon: int, overrides: dict) -> pd.DataFrame:
    rows = []
    for _ in range(horizon):
        r = {c: base_row.get(c, np.nan) for c in EXOG_COLS}
        r.update(overrides)
        rows.append(r)
    return pd.DataFrame(rows)

def scale_future_exog(future_exog_df: pd.DataFrame, scaler, n_feat: int) -> np.ndarray:
    """Scale EXOG (H,6) giống lúc train (7-cột) hoặc fallback legacy."""
    ex = future_exog_df[EXOG_COLS].to_numpy().astype(np.float32)
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in and n_in >= EXPECTED_FEATS:
        s = scaler.scale_[1:1+len(EXOG_COLS)]
        m = scaler.min_[1:1+len(EXOG_COLS)]
        return ex * s + m  # MinMax: X_scaled = X * scale_ + min_
    # legacy
    h, w = ex.shape
    flat = ex.reshape(-1, 1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(h, w)

def recursive_forecast(model, scaler, seed_scaled: np.ndarray, exog_future_scaled: np.ndarray, horizon: int) -> np.ndarray:
    """Dự báo kiểu 1-bước (legacy)."""
    seq_len, n_feat = seed_scaled.shape
    seq = seed_scaled.copy()
    out_scaled = []
    for t in range(horizon):
        x = seq[-seq_len:].reshape(1, seq_len, n_feat)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]
        next_vec = np.empty((n_feat,), dtype=np.float32)
        next_vec[0] = yhat_scaled
        next_vec[1:] = exog_future_scaled[t]
        seq = np.vstack([seq, next_vec])
        out_scaled.append(yhat_scaled)
    yhat_scaled_arr = np.array(out_scaled, dtype=np.float32)
    return _inverse_vector_like_training(yhat_scaled_arr, scaler)

def infer_freq(ts: pd.Series) -> pd.Timedelta:
    diffs = ts.diff()
    if diffs.notna().any():
        return diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()
    return pd.Timedelta(hours=1)

# ===================== SIDEBAR =====================
st.sidebar.subheader("Data path")
hist_path = st.sidebar.text_input("cluster_history.csv", "cluster_history.csv")

st.sidebar.subheader("External factors (override)")
ph = st.sidebar.selectbox("Public holiday", [0, 1], index=0)
sh = st.sidebar.selectbox("School holiday", [0, 1], index=0)
we = st.sidebar.selectbox("Weekend", [0, 1], index=0)
t_avg = st.sidebar.slider("Avg_Temp (°C)", -5.0, 45.0, 24.0, 0.5)
h_avg = st.sidebar.slider("Avg_Humidity (%)", 0.0, 100.0, 60.0, 1.0)
w_avg = st.sidebar.slider("Avg_Wind (m/s)", 0.0, 20.0, 3.0, 0.2)

# ===================== LOAD DATA =====================
df_hist = load_history(hist_path)
with st.expander("👀 Xem toàn bộ cluster_history.csv"):
    st.dataframe(df_hist, use_container_width=True)

if TARGET_COL not in df_hist.columns:
    st.error(f"Không tìm thấy cột {TARGET_COL} trong {hist_path}")
    st.stop()

# Chỉ cho phép 0..4 nếu dữ liệu có & đủ artifacts
allowed = {0, 1, 2, 3, 4}
present = set(df_hist[CLUSTER_COL].unique().tolist()) & allowed
clusters_present = sorted([c for c in present if has_artifacts(c)])
if not clusters_present:
    st.error("Không thấy artifact cho cụm nào trong [0..4].")
    st.stop()

geo_cluster = st.selectbox("Cluster (0–4)", clusters_present)

# ===================== LOAD ARTIFACTS =====================
ver_key = artifact_version_key(int(geo_cluster))
model, scaler, tail_scaled_opt, SEQ_LEN, N_FEAT = load_artifacts_for_cluster(int(geo_cluster), ver_key)

# Kiểm tra phù hợp scaler/model
n_in = getattr(scaler, "n_features_in_", None)
st.caption(f"🔧 Scaler features: {n_in}")
if n_in is not None and n_in != EXPECTED_FEATS:
    st.error(f"Scaler có n_features_in_={n_in} nhưng app mong đợi {EXPECTED_FEATS}.")
    st.stop()

if N_FEAT != EXPECTED_FEATS:
    st.error(f"Model N_FEAT={N_FEAT} nhưng app mong đợi {EXPECTED_FEATS} "
             f"(1 target + {len(EXOG_COLS)} exog).")
    st.stop()

# Lấy horizon từ kiến trúc (Dense(H))
out_units = model.output_shape[-1] if isinstance(model.output_shape, tuple) else model.output_shape[0][-1]
is_direct_multi_output = out_units > 1
final_horizon = out_units if is_direct_multi_output else 14
st.caption(f"📏 Horizon: **{final_horizon}** bước.")

# ===================== SEED =====================
df_feat = build_feature_matrix(df_hist)
seed_raw = (
    df_feat[df_feat[CLUSTER_COL] == geo_cluster]
    .sort_values(TIME_COL)
    .tail(SEQ_LEN)
    .copy()
)
if len(seed_raw) < SEQ_LEN:
    st.error(f"Lịch sử cho cụm {geo_cluster} < SEQ_LEN={SEQ_LEN}.")
    st.stop()

# Override EXOG bằng sliders
seed_raw.loc[:, "public_holiday"] = int(ph)
seed_raw.loc[:, "school_holiday"]  = int(sh)
seed_raw.loc[:, "is_weekend"]      = int(we)
seed_raw.loc[:, "Avg_Temp"]        = float(t_avg)
seed_raw.loc[:, "Avg_Humidity"]    = float(h_avg)
seed_raw.loc[:, "Avg_Wind"]        = float(w_avg)

# Scale seed đúng kiểu scaler khi train (7 cột)
seed_mat = seed_raw[[TARGET_COL] + EXOG_COLS].to_numpy().astype(np.float32)
seed_scaled = _scale_matrix_like_training(seed_mat, scaler)

# ===================== FORECAST =====================
if is_direct_multi_output:
    x_in = seed_scaled.reshape(1, SEQ_LEN, EXPECTED_FEATS)
    yhat_scaled = model.predict(x_in, verbose=0).reshape(-1)  # (H,)
    yhat = _inverse_vector_like_training(yhat_scaled, scaler)
else:
    # Legacy 1-step
    last_row = seed_raw.tail(1).iloc[0]
    overrides = {
        "public_holiday": int(ph),
        "school_holiday": int(sh),
        "is_weekend": int(we),
        "Avg_Temp": float(t_avg),
        "Avg_Humidity": float(h_avg),
        "Avg_Wind": float(w_avg),
    }
    future_exog = make_future_exog_overrides(last_row, final_horizon, overrides)
    exog_future_scaled = scale_future_exog(future_exog, scaler, EXPECTED_FEATS)
    yhat = recursive_forecast(model, scaler, seed_scaled, exog_future_scaled, horizon=final_horizon)

# ===================== PLOT =====================
hist_tail = df_hist[df_hist[CLUSTER_COL] == geo_cluster].sort_values(TIME_COL).tail(SEQ_LEN).copy()
t0 = hist_tail[TIME_COL].iloc[-1]

# dùng bước thời gian của 2 điểm cuối cùng
if len(hist_tail) >= 2:
    freq = hist_tail[TIME_COL].iloc[-1] - hist_tail[TIME_COL].iloc[-2]
else:
    freq = pd.Timedelta(days=1)

future_times = [t0 + (i+1)*freq for i in range(final_horizon)]

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

# (tùy chọn) chèn điểm nối ở t0 để nhìn liền mạch hơn
df_plot_fcst = pd.concat([
    pd.DataFrame({"timestamp":[t0],
                  "value":[hist_tail[TARGET_COL].iloc[-1]],
                  "type":["Forecast"]}),
    df_plot_fcst
], ignore_index=True)

df_plot = pd.concat([df_plot_hist, df_plot_fcst], ignore_index=True)

chart = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("timestamp:T", title="Time"),
    y=alt.Y("value:Q", title="Demand (kWh)"),
    color=alt.Color("type:N", sort=["History","Forecast"])
).properties(width="container", height=380,
             title=f"Cluster {geo_cluster} — GRU Forecast ({final_horizon} steps)")
st.altair_chart(chart, use_container_width=True)


# ===================== EXPORT =====================
with st.expander("Export"):
    st.download_button(
        "Download Forecast CSV",
        data=df_plot_fcst.to_csv(index=False),
        file_name=f"forecast_cluster_{geo_cluster}.csv",
        mime="text/csv"
    )

# ===================== DEBUG (tùy chọn) =====================
with st.expander("🔍 Debug scaler"):
    st.write("n_features_in_:", getattr(scaler, "n_features_in_", None))
    s = getattr(scaler, "scale_", None); m = getattr(scaler, "min_", None)
    if s is not None and m is not None:
        st.write("scale_.shape:", s.shape, "min_.shape:", m.shape)

