import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import timedelta
from tensorflow.keras.models import load_model
import joblib

# ========== PAGE SETUP ==========
TIME_COL = "datetime"
ID_COL = "station_id"
TARGET_COL = "estimated_demand_kWh"
EXOG_COLS = ["public_holiday","school_holiday","is_weekend",
             "Avg_Temp","Avg_Humidity","Avg_Wind"]

GLOBAL_MODEL_PATH = "model_gru.keras"         # bạn đã cung cấp
GLOBAL_SCALER_PATH = "scaler_all.joblib"      # bạn đã cung cấp
GLOBAL_TAIL_PATH = "tail.npy"                 # optional (seed đã scale)
CLUSTER_ARTIFACT_ROOT = "artifacts"  
st.set_page_config(page_title="EVAT — GRU Forecast by Station", page_icon="⚡", layout="wide")
st.title("⚡ EVAT — GRU Forecast per Cluster / Station")
st.caption("Select a station, adjust external factors → get a **forecast line chart**.")

