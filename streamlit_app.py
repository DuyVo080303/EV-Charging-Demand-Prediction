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

def _scale_matrix_like_training(mat: np.ndarray, scaler) -> np.ndarray:
    """
    Scale theo cách bạn đã train:
    - Flatten toàn bộ ma trận về (-1, 1)
    - scaler.transform(...)
    - Reshape lại kích thước ban đầu
    """
    h, w = mat.shape
    flat = mat.reshape(-1, 1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(h, w)

def _inverse_vector_like_training(vec: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse cho một vector (horizon,) theo cách bạn đã train (scaler 1 cột).
    """
    flat = vec.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(-1)

def take_last_sequence_scaled(df_feat: pd.DataFrame, station_id, seq_len: int, scaler) -> np.ndarray:
    """
    Lấy SEQ_LEN cuối cùng của station → (seq_len, n_feat),
    rồi scale theo cách flatten 1-cột (giống lúc training).
    """
    d = df_feat[df_feat[ID_COL] == station_id].tail(seq_len)
    if len(d) < seq_len:
        raise ValueError(f"Lịch sử cho station {station_id} < SEQ_LEN={seq_len}.")
    # (seq_len, n_feat) với n_feat = 1 + len(EXOG_COLS)
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
    """
    Trả về exog tương lai đã scale theo đúng cách flatten-1-cột.
    Kết quả shape: (H, n_feat-1) tương ứng các EXOG.
    """
    ex = future_exog_df[EXOG_COLS].to_numpy().astype(float)   # (H, len(EXOG))
    ex_scaled = _scale_matrix_like_training(ex, scaler)       # scale từng phần tử theo scaler 1 cột
    return ex_scaled  # (H, len(EXOG)) == (H, n_feat-1)

def recursive_forecast(model, scaler, seed_scaled: np.ndarray, exog_future_scaled: np.ndarray, horizon: int) -> np.ndarray:
    """
    seed_scaled: (seq_len, n_feat) đã scale (theo cách flatten-1-cột).
    exog_future_scaled: (H, n_feat-1) đã scale (flatten-1-cột).
    Trả về yhat (H,) đã inverse theo scaler 1 cột.
    """
    seq_len, n_feat = seed_scaled.shape
    seq = seed_scaled.copy()
    out_scaled = []

    for t in range(horizon):
        x = seq[-seq_len:].reshape(1, seq_len, n_feat)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]   # giá trị TARGET đã ở “không gian scaled”
        # ghép bước kế tiếp vào chuỗi
        next_vec = np.empty((n_feat,), dtype=float)
        next_vec[0] = yhat_scaled
        next_vec[1:] = exog_future_scaled[t]
        seq = np.vstack([seq, next_vec])
        out_scaled.append(yhat_scaled)

    # Inverse target theo đúng cách scaler 1 cột
    yhat_scaled_arr = np.array(out_scaled, dtype=float)        # (H,)
    inv = _inverse_vector_like_training(yhat_scaled_arr, scaler)  # (H,)
    return inv

def infer_freq(ts: pd.Series) -> pd.Timedelta:
    diffs = ts.diff()
    if diffs.notna().any():
        return diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()
    return pd.Timedelta(hours=1)

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
# ==========/ LOAD ==========
hist_path = "history.csv"
map_path  = "station_to_cluster.csv"

df_hist = load_history(hist_path)
map_df  = load_station_cluster_map(map_path)

# Đồng bộ kiểu station_id giữa 2 file
try:
    df_hist[ID_COL] = pd.to_numeric(df_hist[ID_COL], errors="raise")
    map_df["station_id"] = pd.to_numeric(map_df["station_id"], errors="raise")
except Exception:
    df_hist[ID_COL] = df_hist[ID_COL].astype(str)
    map_df["station_id"] = map_df["station_id"].astype(str)

stations = sorted(df_hist[ID_COL].unique().tolist())
station_id = st.selectbox("Station", stations)

# Lấy geo_cluster (bắt buộc tồn tại, >=0, duy nhất)
row = map_df.loc[map_df["station_id"] == station_id, "geo_cluster"]
if row.empty:
    st.error(f"Không tìm thấy geo_cluster cho station_id={station_id} trong station_to_cluster.csv")
    st.stop()
geo_cluster = int(row.iloc[0])
st.write(f"**Cluster:** `{geo_cluster}` • **Station:** `{station_id}`")

# Chỉ dùng artifacts theo CỤM
model, scaler, tail_scaled_opt, SEQ_LEN, N_FEAT = load_artifacts_for_cluster(geo_cluster)

# --- Kiểm tra khớp cấu hình với scaler 1-cột (flatten) ---
n_in = getattr(scaler, "n_features_in_", None)
expected_feats = 1 + len(EXOG_COLS)
if n_in is not None and n_in != 1:
    st.error(f"Scaler của cụm {geo_cluster} có n_features_in_={n_in}, "
             f"nhưng pipeline training của bạn dùng scaler 1-cột. Hãy export scaler đúng pipeline.")
    st.stop()
if N_FEAT != expected_feats:
    st.error(f"Model N_FEAT={N_FEAT} nhưng app mong đợi {expected_feats} "
             f"(1 target + {len(EXOG_COLS)} exog). Kiểm tra lại kiến trúc/model cụm.")
    st.stop()

# ==========/ SEED ==========
df_feat = build_feature_matrix(df_hist)

# Nếu tail.npy đã scale đúng cách và shape khớp thì dùng, ngược lại tự dựng từ history
if tail_scaled_opt is not None and tail_scaled_opt.shape == (SEQ_LEN, N_FEAT):
    seed_scaled = tail_scaled_opt
else:
    if tail_scaled_opt is not None and tail_scaled_opt.shape != (SEQ_LEN, N_FEAT):
        st.info("`tail.npy` không khớp shape model hoặc không đúng kiểu scale → sẽ tự dựng seed từ history.")
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
future_exog = make_future_exog_overrides(last_row, horizon, overrides)

# Scale EXOG tương lai theo cách 1-cột (đã sửa trong hàm)
exog_future_scaled = scale_future_exog(future_exog, scaler, N_FEAT)

# ==========/ FORECAST ==========
yhat = recursive_forecast(model, scaler, seed_scaled, exog_future_scaled, horizon=horizon)

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
).properties(width="container", height=380,
             title=f"Station {station_id} — GRU Forecast ({horizon} steps)")
st.altair_chart(chart, use_container_width=True)

# ==========/ EXPORT ==========
with st.expander("Export"):
    st.download_button(
        "Download Forecast CSV",
        data=df_plot_fcst.to_csv(index=False),
        file_name=f"forecast_station_{station_id}.csv",
        mime="text/csv"
    )
