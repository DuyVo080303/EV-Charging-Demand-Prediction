import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import timedelta
from tensorflow.keras.models import load_model
import joblib

# ========== PAGE SETUP ==========
TIME_COL     = "Date"
CLUSTER_COL  = "geo_cluster"                 # dùng cluster, không còn station_id
TARGET_COL   = "estimated_demand_kWh"
EXOG_COLS    = ["public_holiday","school_holiday","is_weekend",
                "Avg_Temp","Avg_Humidity","Avg_Wind"]

CLUSTER_ARTIFACT_ROOT = "artifacts"
st.set_page_config(page_title="EVAT — GRU Forecast by Cluster", page_icon="⚡", layout="wide")
st.title("⚡ EVAT — GRU Forecast per Cluster")
st.caption("Select a cluster, adjust external factors → get a **forecast line chart**.")

# ========== UTILITY FUNCTIONS ==========
@st.cache_data(show_spinner=False)
def load_history(path: str) -> pd.DataFrame:
    """
    Yêu cầu cột: Date, geo_cluster, estimated_demand_kWh, cùng các EXOG_COLS.
    """
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.sort_values([CLUSTER_COL, TIME_COL]).reset_index(drop=True)
    needed = [TIME_COL, CLUSTER_COL, TARGET_COL] + EXOG_COLS
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"`{path}` thiếu cột: {miss}")
    # ép kiểu geo_cluster và loại NaN/âm
    df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce")
    if df[CLUSTER_COL].isna().any():
        raise ValueError(f"Có NaN ở `{CLUSTER_COL}` trong {path}. Vui lòng làm sạch dữ liệu.")
    if (df[CLUSTER_COL] < 0).any():
        raise ValueError(f"Có giá trị âm ở `{CLUSTER_COL}` (ví dụ -1). Vui lòng lọc bỏ.")
    return df

def cluster_dir_candidates(cid: int) -> list:
    cid = int(cid)
    return [
        os.path.join("artifacts", "clusters", str(cid)),   # cấu trúc mới
        os.path.join("artifacts", f"cluster_{cid}"),       # phòng cấu trúc cũ
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
            st.write("✅ Using cluster artifacts from:", cdir)
            try:
                st.write("Contents:", os.listdir(cdir))
            except Exception:
                pass
            model = load_model(mpath)
            scaler = joblib.load(spath)
            tail_scaled = np.load(tpath) if os.path.exists(tpath) else None
            # trả về SEQ_LEN, N_FEAT từ input shape model
            return model, scaler, tail_scaled, model.input_shape[1], model.input_shape[2]

    st.write("❌ Could not find artifacts for cluster:", geo_cluster)
    st.write("CWD:", os.getcwd())
    for cdir, mpath, spath, _ in tried:
        st.write("Tried:", cdir,
                 "| model exists?", os.path.exists(mpath),
                 "| scaler exists?", os.path.exists(spath))
        if os.path.exists(cdir):
            try:
                st.write("Dir contents:", os.listdir(cdir))
            except Exception:
                pass
    raise FileNotFoundError(
        f"Không tìm thấy model/scaler cho cụm {geo_cluster}. "
        f"Yêu cầu các file 'model_gru.keras', 'scaler_all.joblib' (và tùy chọn 'tail.npy') "
        f"trong: {', '.join(cluster_dir_candidates(geo_cluster))}"
    )

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [TIME_COL, CLUSTER_COL, TARGET_COL] + EXOG_COLS
    return df[cols].copy()

def _scale_matrix_like_training(mat: np.ndarray, scaler) -> np.ndarray:
    """
    Nếu scaler đã fit theo cột (7 cột) -> transform trực tiếp (T,7).
    Nếu scaler 1-cột (flatten) -> giữ nguyên cách cũ để tránh lệch phân phối.
    """
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in == mat.shape[1]:           # ví dụ 7 cột: [target] + 6 exog
        return scaler.transform(mat)
    # --- fallback: scaler 1-cột (cách cũ) ---
    h, w = mat.shape
    flat = mat.reshape(-1, 1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(h, w)

def _inverse_vector_like_training(vec: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse target theo đúng kiểu scaler đã fit.
    - Feature-wise (7 cột): dùng min_/scale_ của cột target (index 0).
    - 1-cột (flatten): dùng inverse_transform như cũ.
    """
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in and n_in > 1:
        # MinMax inverse cho cột 0: X = (X_scaled - min_[0]) / scale_[0]
        return (vec - scaler.min_[0]) / scaler.scale_[0]
    # --- fallback: scaler 1-cột ---
    flat = vec.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(-1)

def take_last_sequence_scaled(df_feat: pd.DataFrame, geo_cluster, seq_len: int, scaler) -> np.ndarray:
    """
    Lấy SEQ_LEN cuối của cụm → (seq_len, n_feat), rồi scale theo cách flatten-1-cột.
    """
    d = df_feat[df_feat[CLUSTER_COL] == geo_cluster].tail(seq_len)
    if len(d) < seq_len:
        raise ValueError(f"Lịch sử cho cụm {geo_cluster} < SEQ_LEN={seq_len}.")
    mat = d[[TARGET_COL] + EXOG_COLS].to_numpy().astype(float)
    return _scale_matrix_like_training(mat, scaler)

def make_future_exog_overrides(base_row: pd.Series, horizon: int, overrides: dict) -> pd.DataFrame:
    rows = []
    for _ in range(horizon):
        r = {c: base_row.get(c, np.nan) for c in EXOG_COLS}
        r.update(overrides)
        rows.append(r)
    return pd.DataFrame(rows)

def scale_future_exog(future_exog_df: pd.DataFrame, scaler, n_feat: int) -> np.ndarray:
    ex = future_exog_df[EXOG_COLS].to_numpy().astype(float)
    ex_scaled = _scale_matrix_like_training(ex, scaler)
    return ex_scaled

def recursive_forecast(model, scaler, seed_scaled: np.ndarray, exog_future_scaled: np.ndarray, horizon: int) -> np.ndarray:
    seq_len, n_feat = seed_scaled.shape
    seq = seed_scaled.copy()
    out_scaled = []
    for t in range(horizon):
        x = seq[-seq_len:].reshape(1, seq_len, n_feat)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]
        next_vec = np.empty((n_feat,), dtype=float)
        next_vec[0] = yhat_scaled
        next_vec[1:] = exog_future_scaled[t]
        seq = np.vstack([seq, next_vec])
        out_scaled.append(yhat_scaled)
    yhat_scaled_arr = np.array(out_scaled, dtype=float)
    inv = _inverse_vector_like_training(yhat_scaled_arr, scaler)
    return inv

def infer_freq(ts: pd.Series) -> pd.Timedelta:
    diffs = ts.diff()
    if diffs.notna().any():
        return diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()
    return pd.Timedelta(hours=1)

# ==========/ SIDEBAR ==========
st.sidebar.subheader("Data path")
hist_path = st.sidebar.text_input("cluster_history.csv", "cluster_history.csv")

st.sidebar.subheader("External factors (override)")
ph = st.sidebar.selectbox("Public holiday", [0, 1], index=0)
sh = st.sidebar.selectbox("School holiday", [0, 1], index=0)
we = st.sidebar.selectbox("Weekend", [0, 1], index=0)
t_avg = st.sidebar.slider("Avg_Temp (°C)", -5.0, 45.0, 24.0, 0.5)
h_avg = st.sidebar.slider("Avg_Humidity (%)", 0.0, 100.0, 60.0, 1.0)
w_avg = st.sidebar.slider("Avg_Wind (m/s)", 0.0, 20.0, 3.0, 0.2)

# ==========/ LOAD ==========
hist_path = "cluster_history.csv"

df_hist = load_history(hist_path)

with st.expander("👀 Xem toàn bộ cluster_history.csv"):
    st.dataframe(df_hist, use_container_width=True)

if TARGET_COL not in df_hist.columns:
    st.error(f"Không tìm thấy cột {TARGET_COL} trong {hist_path}")
    st.stop()

allowed_clusters = [0, 1, 2, 3, 4]
clusters = [c for c in allowed_clusters if c in df_hist[CLUSTER_COL].unique()]
geo_cluster = st.selectbox("Cluster", clusters)

# Artifacts theo CỤM
model, scaler, tail_scaled_opt, SEQ_LEN, N_FEAT = load_artifacts_for_cluster(int(geo_cluster))

# Kiểm tra scaler & n_feat
n_in = getattr(scaler, "n_features_in_", None)
expected_feats = 1 + len(EXOG_COLS)
if n_in is not None and n_in != 1:
    st.error(f"Scaler cho cụm {geo_cluster} có n_features_in_={n_in}, "
             f"trong khi pipeline dùng scaler 1-cột. Hãy export scaler đúng.")
    st.stop()
if N_FEAT != expected_feats:
    st.error(f"Model N_FEAT={N_FEAT} nhưng app mong đợi {expected_feats} "
             f"(1 target + {len(EXOG_COLS)} exog). Kiểm tra lại model cụm.")
    st.stop()

# Nhận dạng loại model theo output shape và ẤN ĐỊNH HORIZON (không cần nhập)
out_units = (model.output_shape[-1] if isinstance(model.output_shape, tuple)
             else model.output_shape[0][-1])
is_direct_multi_output = out_units > 1  # ví dụ = 14 theo code train của bạn

if is_direct_multi_output:
    final_horizon = out_units
    st.caption(f"📏 Horizon cố định theo mô hình: **{final_horizon}** bước.")
else:
    # Nếu model 1-bước, ta ấn định mặc định 14 bước để giữ hành vi quen thuộc.
    final_horizon = 14  # <-- đổi số này nếu bạn muốn mặc định khác
    st.caption(f"📏 Model 1-bước: dùng horizon mặc định **{final_horizon}** (không có ô nhập).")

# ==========/ SEED ==========
# ==========/ SEED ==========
df_feat = build_feature_matrix(df_hist)

# Lấy 50 bước cuối của cụm làm seed (giữ nguyên target history)
seed_raw = (
    df_feat[df_feat[CLUSTER_COL] == geo_cluster]
    .sort_values(TIME_COL)
    .tail(SEQ_LEN)
    .copy()
)
if len(seed_raw) < SEQ_LEN:
    st.error(f"Lịch sử cho cụm {geo_cluster} < SEQ_LEN={SEQ_LEN}.")
    st.stop()

# 👉 GHI ĐÈ EXOG TRONG CỬA SỔ BẰNG GIÁ TRỊ USER CHỌN
seed_raw.loc[:, "public_holiday"] = int(ph)
seed_raw.loc[:, "school_holiday"] = int(sh)
seed_raw.loc[:, "is_weekend"] = int(we)
seed_raw.loc[:, "Avg_Temp"] = float(t_avg)
seed_raw.loc[:, "Avg_Humidity"] = float(h_avg)
seed_raw.loc[:, "Avg_Wind"] = float(w_avg)

# Scale seed theo cách flatten-1-cột (đúng pipeline train)
seed_mat = seed_raw[[TARGET_COL] + EXOG_COLS].to_numpy().astype(float)
seed_scaled = _scale_matrix_like_training(seed_mat, scaler)

# ==========/ FORECAST ==========
if is_direct_multi_output:
    # Dự báo trực tiếp H bước từ seed đã override EXOG
    x_in = seed_scaled.reshape(1, SEQ_LEN, N_FEAT)
    yhat_scaled = model.predict(x_in, verbose=0).reshape(-1)      # (H,)
    yhat = _inverse_vector_like_training(yhat_scaled, scaler)     # về kWh
else:
    # Model 1-bước (ít gặp trong code train của bạn) - vẫn hỗ trợ
    # tạo exog_future lặp lại đúng các giá trị user để nhất quán
    overrides = {
        "public_holiday": int(ph),
        "school_holiday": int(sh),
        "is_weekend": int(we),
        "Avg_Temp": float(t_avg),
        "Avg_Humidity": float(h_avg),
        "Avg_Wind": float(w_avg),
    }
    last_row = seed_raw.tail(1).iloc[0]
    future_exog = make_future_exog_overrides(last_row, final_horizon, overrides)
    exog_future_scaled = scale_future_exog(future_exog, scaler, N_FEAT)
    yhat = recursive_forecast(model, scaler, seed_scaled, exog_future_scaled, horizon=final_horizon)

# ==========/ PLOT ==========
hist_tail = df_hist[df_hist[CLUSTER_COL] == geo_cluster].sort_values(TIME_COL).tail(SEQ_LEN).copy()
t0 = hist_tail[TIME_COL].iloc[-1]
freq = infer_freq(hist_tail[TIME_COL])
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
df_plot = pd.concat([df_plot_hist, df_plot_fcst], ignore_index=True)

chart = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("timestamp:T", title="Time"),
    y=alt.Y("value:Q", title="Demand (kWh)"),
    color=alt.Color("type:N", sort=["History","Forecast"])
).properties(width="container", height=380,
             title=f"Cluster {geo_cluster} — GRU Forecast ({final_horizon} steps)")
st.altair_chart(chart, use_container_width=True)

# ==========/ EXPORT ==========
with st.expander("Export"):
    st.download_button(
        "Download Forecast CSV",
        data=df_plot_fcst.to_csv(index=False),
        file_name=f"forecast_cluster_{geo_cluster}.csv",
        mime="text/csv"
    )
