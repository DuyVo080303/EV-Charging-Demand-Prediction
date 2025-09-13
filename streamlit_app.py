import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import timedelta
from tensorflow.keras.models import load_model

# ========== PAGE SETUP ==========
TIME_COL = "Date"
ID_COL = "station_id"
TARGET_COL = "estimated_demand_kWh"
EXOG_COLS = ["public_holiday","school_holiday","is_weekend",
             "Avg_Temp","Avg_Humidity","Avg_Wind"]

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
    m["geo_cluster"] = pd.to_numeric(m["geo_cluster"], errors="coerce")
    if m["geo_cluster"].isna().any():
        bad = m[m["geo_cluster"].isna()]["station_id"].tolist()[:10]
        raise ValueError(f"Có NaN ở geo_cluster (ví dụ station_id: {bad})")
    if (m["geo_cluster"] < 0).any():
        bad = m[m["geo_cluster"] < 0]["station_id"].tolist()[:10]
        raise ValueError(f"Có giá trị âm (-1) ở geo_cluster (ví dụ station_id: {bad})")
    dup = m.duplicated("station_id", keep=False)
    if dup.any():
        raise ValueError("station_to_cluster.csv có station_id trùng lặp.")
    return m[["station_id", "geo_cluster"]]

def cluster_dir_candidates(cid: int) -> list:
    cid = int(cid)
    return [
        os.path.join("artifacts", "clusters", str(cid)),
        os.path.join("artifacts", f"cluster_{cid}"),
    ]

def load_minmax_params(cdir: str):
    jpath = os.path.join(cdir, "minmax_params.json")
    if not os.path.exists(jpath):
        return None
    with open(jpath, "r", encoding="utf-8") as f:
        params = json.load(f)
    cols = params["columns"]
    vmin = np.asarray(params["min"], dtype=float)
    vmax = np.asarray(params["max"], dtype=float)
    return {"columns": cols, "min": vmin, "max": vmax}

@st.cache_resource(show_spinner=False)
def load_artifacts_for_cluster(geo_cluster: int):
    tried = []
    for cdir in cluster_dir_candidates(geo_cluster):
        mpath = os.path.join(cdir, "model_gru.keras")
        tpath = os.path.join(cdir, "tail.npy")
        tried.append((cdir, mpath, tpath))
        if os.path.exists(mpath):
            st.write("✅ Using cluster artifacts from:", cdir)
            try:
                st.write("Contents:", os.listdir(cdir))
            except Exception:
                pass
            model = load_model(mpath)
            tail_scaled = np.load(tpath) if os.path.exists(tpath) else None
            mm = load_minmax_params(cdir)
            if mm is None:
                raise FileNotFoundError(
                    f"Thiếu 'minmax_params.json' cho cụm {geo_cluster}. "
                    f"Hãy xuất min/max theo thứ tự cột [{TARGET_COL}] + EXOG_COLS trong thư mục: {cdir}"
                )
            return {
                "model": model,
                "minmax": mm,
                "tail": tail_scaled,
                "SEQ_LEN": model.input_shape[1],
                "N_FEAT": model.input_shape[2],
                "cdir": cdir,
            }

    st.write("❌ Could not find artifacts for cluster:", geo_cluster)
    st.write("CWD:", os.getcwd())
    for cdir, mpath, _ in tried:
        st.write("Tried:", cdir, "| model exists?", os.path.exists(mpath))
        if os.path.exists(cdir):
            try:
                st.write("Dir contents:", os.listdir(cdir))
            except Exception:
                pass
    raise FileNotFoundError(
        f"Không tìm thấy model cho cụm {geo_cluster}. "
        f"Yêu cầu 'model_gru.keras' và 'minmax_params.json' trong một trong các thư mục: "
        + ", ".join(cluster_dir_candidates(geo_cluster))
    )

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [TIME_COL, ID_COL, TARGET_COL] + EXOG_COLS
    return df[cols].copy()

# ===== Min–Max per-feature (KHÔNG FLATTEN) =====
def _scale_matrix_per_feature(mat: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    # mat shape: (n_rows, n_feat) theo thứ tự [TARGET] + EXOG_COLS
    scale = (vmax - vmin)
    scale[scale == 0] = 1.0
    out = (mat - vmin) / scale
    return np.clip(out, 0.0, 1.0)

def _inverse_target_per_feature(vec_scaled: np.ndarray, vmin_target: float, vmax_target: float) -> np.ndarray:
    scale = (vmax_target - vmin_target)
    if scale == 0:
        scale = 1.0
    return vec_scaled * scale + vmin_target

def take_last_sequence_scaled_minmax(df_feat: pd.DataFrame, station_id, seq_len: int, mm) -> np.ndarray:
    d = df_feat[df_feat[ID_COL] == station_id].tail(seq_len)
    if len(d) < seq_len:
        raise ValueError(f"Lịch sử cho station {station_id} < SEQ_LEN={seq_len}.")
    cols = [TARGET_COL] + EXOG_COLS
    mat = d[cols].to_numpy(dtype=float)  # (seq_len, n_feat)
    return _scale_matrix_per_feature(mat, mm["min"], mm["max"])

def scale_future_exog_minmax(future_exog_df: pd.DataFrame, mm) -> np.ndarray:
    ex = future_exog_df[EXOG_COLS].to_numpy(dtype=float)      # (H, len(EXOG))
    return _scale_matrix_per_feature(ex, mm["min"][1:], mm["max"][1:])  # bỏ TARGET

def recursive_forecast_minmax(model, seed_scaled: np.ndarray, exog_future_scaled: np.ndarray, horizon: int) -> np.ndarray:
    """
    seed_scaled: (seq_len, n_feat) đã scale per-feature.
    exog_future_scaled: (H, n_feat-1) cho EXOG (đã scale).
    Trả về yhat_scaled (H,) trong miền [0,1].
    """
    seq_len, n_feat = seed_scaled.shape
    seq = seed_scaled.copy()
    out_scaled = []

    for t in range(horizon):
        x = seq[-seq_len:].reshape(1, seq_len, n_feat)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]   # y scaled
        next_vec = np.empty((n_feat,), dtype=float)
        next_vec[0] = yhat_scaled
        next_vec[1:] = exog_future_scaled[t]
        seq = np.vstack([seq, next_vec])
        out_scaled.append(yhat_scaled)

    return np.asarray(out_scaled, dtype=float)

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
df_hist = load_history(hist_path)
map_df  = load_station_cluster_map(map_path)

with st.expander("👀 Xem toàn bộ history.csv"):
    st.dataframe(df_hist, use_container_width=True)

if TARGET_COL not in df_hist.columns:
    st.error(f"Không tìm thấy cột {TARGET_COL} trong history.csv")
    st.stop()

# Đồng bộ kiểu station_id giữa 2 file
try:
    df_hist[ID_COL] = pd.to_numeric(df_hist[ID_COL], errors="raise")
    map_df["station_id"] = pd.to_numeric(map_df["station_id"], errors="raise")
except Exception:
    df_hist[ID_COL] = df_hist[ID_COL].astype(str)
    map_df["station_id"] = map_df["station_id"].astype(str)

stations = sorted(df_hist[ID_COL].unique().tolist())
station_id = st.selectbox("Station", stations)

# Lấy geo_cluster
row = map_df.loc[map_df["station_id"] == station_id, "geo_cluster"]
if row.empty:
    st.error(f"Không tìm thấy geo_cluster cho station_id={station_id} trong station_to_cluster.csv")
    st.stop()
geo_cluster = int(row.iloc[0])
st.write(f"**Cluster:** `{geo_cluster}` • **Station:** `{station_id}`")

# Load artifacts (model + minmax)
art = load_artifacts_for_cluster(geo_cluster)
model = art["model"]
mm = art["minmax"]
SEQ_LEN = art["SEQ_LEN"]
N_FEAT = art["N_FEAT"]

expected_feats = 1 + len(EXOG_COLS)
if N_FEAT != expected_feats:
    st.error(f"Model N_FEAT={N_FEAT} nhưng app mong đợi {expected_feats} "
             f"(1 target + {len(EXOG_COLS)} exog). Kiểm tra lại kiến trúc/model cụm.")
    st.stop()

# ==========/ SEED ==========
df_feat = build_feature_matrix(df_hist)

# Nếu tail.npy đã scale per-feature và shape khớp thì dùng; không thì tự dựng
tail_scaled_opt = art["tail"]
if tail_scaled_opt is not None and tail_scaled_opt.shape == (SEQ_LEN, N_FEAT):
    seed_scaled = tail_scaled_opt
else:
    if tail_scaled_opt is not None and tail_scaled_opt.shape != (SEQ_LEN, N_FEAT):
        st.info("`tail.npy` không khớp shape model → sẽ tự dựng seed từ history.")
    seed_scaled = take_last_sequence_scaled_minmax(df_feat, station_id, SEQ_LEN, mm)

# Tạo EXOG tương lai từ last_row + override (user có thể nhập 1 hàng vẫn OK)
last_row = (df_feat[df_feat[ID_COL] == station_id].tail(1)).iloc[0]
overrides = {
    "public_holiday": int(ph),
    "school_holiday": int(sh),
    "is_weekend": int(we),
    "Avg_Temp": float(t_avg),
    "Avg_Humidity": float(h_avg),
    "Avg_Wind": float(w_avg),
}
def make_future_exog_overrides(base_row: pd.Series, horizon: int, overrides: dict) -> pd.DataFrame:
    rows = []
    for _ in range(horizon):
        r = {c: base_row.get(c, np.nan) for c in EXOG_COLS}
        r.update(overrides)
        rows.append(r)
    return pd.DataFrame(rows)

future_exog = make_future_exog_overrides(last_row, horizon, overrides)
exog_future_scaled = scale_future_exog_minmax(future_exog, mm)

# ==========/ FORECAST ==========
yhat_scaled = recursive_forecast_minmax(model, seed_scaled, exog_future_scaled, horizon=horizon)
yhat = _inverse_target_per_feature(yhat_scaled, mm["min"][0], mm["max"][0])

# ==========/ PLOT ==========
hist_tail = df_hist[df_hist[ID_COL] == station_id].sort_values(TIME_COL).tail(SEQ_LEN).copy()
t0 = hist_tail[TIME_COL].iloc[-1]
freq = infer_freq(hist_tail[TIME_COL])
future_times = [t0 + (i+1)*freq for i in range(horizon)]

# Lọc history theo station
df_plot_hist = df_hist[df_hist[ID_COL] == station_id][[TIME_COL, TARGET_COL]].copy()
df_plot_hist.columns = ["timestamp", "value"]
df_plot_hist["type"] = "History"

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
