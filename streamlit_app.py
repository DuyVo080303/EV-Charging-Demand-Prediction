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

CLUSTER_ARTIFACT_ROOT = "artifacts"  
st.set_page_config(page_title="EVAT — GRU Forecast by Cluster", page_icon="⚡", layout="wide")
st.title("⚡ EVAT — GRU Forecast per Cluster")
st.caption("Select a cluster, adjust external factors → get a **forecast line chart**.")

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
    return m[["station_id", "geo_cluster"]]

def cluster_dir_candidates(cid: int) -> list:
    cid = int(cid)
    return [
        os.path.join("artifacts", "clusters", str(cid)),
        os.path.join("artifacts", f"cluster_{cid}"),
    ]

@st.cache_resource(show_spinner=False)
def load_artifacts_for_cluster(geo_cluster: int):
    for cdir in cluster_dir_candidates(geo_cluster):
        mpath = os.path.join(cdir, "model_gru.keras")
        spath = os.path.join(cdir, "scaler_all.joblib")
        tpath = os.path.join(cdir, "tail.npy")
        if os.path.exists(mpath) and os.path.exists(spath):
            model = load_model(mpath)
            scaler = joblib.load(spath)
            tail_scaled = np.load(tpath) if os.path.exists(tpath) else None
            return model, scaler, tail_scaled, model.input_shape[1], model.input_shape[2]
    raise FileNotFoundError(f"Model/scaler not found for cluster {geo_cluster}.")

def build_cluster_feature_matrix(df: pd.DataFrame, cluster_stations: list) -> pd.DataFrame:
    # Trung bình demand và exog theo cluster
    df_cluster = df[df[ID_COL].isin(cluster_stations)].copy()
    df_agg = df_cluster.groupby(TIME_COL)[[TARGET_COL]+EXOG_COLS].mean().reset_index()
    return df_agg

def scale_future_exog(future_exog_df: pd.DataFrame, scaler, n_feat: int) -> np.ndarray:
    ex = future_exog_df[EXOG_COLS].to_numpy().astype(float)
    flat = ex.reshape(-1,1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(ex.shape)

def make_future_exog_overrides(base_row: pd.Series, horizon: int, overrides: dict) -> pd.DataFrame:
    rows = []
    for _ in range(horizon):
        r = {c: base_row.get(c, np.nan) for c in EXOG_COLS}
        r.update(overrides)
        rows.append(r)
    return pd.DataFrame(rows)

def _inverse_vector_like_training(vec: np.ndarray, scaler) -> np.ndarray:
    flat = vec.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(-1)

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
    return diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()

# ==========/ SIDEBAR ==========
st.sidebar.subheader("Data paths")
hist_path = st.sidebar.text_input("history.csv", "history.csv")
map_path  = st.sidebar.text_input("station_to_cluster.csv", "station_to_cluster.csv")

horizon = 14

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

clusters = sorted(map_df["geo_cluster"].unique())
geo_cluster = st.selectbox("Cluster", clusters)
st.write(f"**Cluster:** `{geo_cluster}`")

cluster_stations = map_df.loc[map_df["geo_cluster"]==geo_cluster, "station_id"].tolist()
model, scaler, tail_scaled_opt, SEQ_LEN, N_FEAT = load_artifacts_for_cluster(geo_cluster)

df_feat = build_cluster_feature_matrix(df_hist, cluster_stations)

# Seed cho forecast (dùng tail của cluster nếu có)
if tail_scaled_opt is not None and tail_scaled_opt.shape == (SEQ_LEN, N_FEAT):
    seed_scaled = tail_scaled_opt
else:
    # Lấy SEQ_LEN gần nhất của cluster
    seed_scaled = df_feat[[TARGET_COL]+EXOG_COLS].tail(SEQ_LEN).to_numpy()
    # Scale
    flat = seed_scaled.reshape(-1,1)
    seed_scaled = scaler.transform(flat).reshape(SEQ_LEN, N_FEAT)

last_row = df_feat.tail(1).iloc[0]
overrides = {
    "public_holiday": int(ph),
    "school_holiday": int(sh),
    "is_weekend": int(we),
    "Avg_Temp": float(t_avg),
    "Avg_Humidity": float(h_avg),
    "Avg_Wind": float(w_avg),
}
future_exog = make_future_exog_overrides(last_row, horizon, overrides)
exog_future_scaled = scale_future_exog(future_exog, scaler, N_FEAT)

# ==========/ FORECAST ==========
yhat = recursive_forecast(model, scaler, seed_scaled, exog_future_scaled, horizon=horizon)

# ==========/ PLOT ==========
hist_tail = df_feat.tail(SEQ_LEN).copy()
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
).properties(width="container", height=380, title=f"Cluster {geo_cluster} — GRU Forecast ({horizon} steps)")

st.altair_chart
