import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="APS Failure Triage", layout="centered")
st.title("Scania APS Failure Triage")
st.write("Upload a CSV of raw sensor readings (same schema as the APS dataset) to get an APS / not-APS call per truck.")

with open("xgb_aps_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
imputer = artifact["imputer"]
threshold = artifact["threshold"]
columns = artifact["columns"]
high_missing_cols = artifact["high_missing_cols"]
mod_missing_cols = artifact["mod_missing_cols"]

uploaded = st.file_uploader("Upload sensor CSV", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded, na_values="na", low_memory=False)
    df = df.drop(columns=[c for c in high_missing_cols if c in df.columns])
    for c in mod_missing_cols:
        df[f"{c}_missing"] = df[c].isna().astype(int) if c in df.columns else 1
    df = df.reindex(columns=columns, fill_value=np.nan)
    df_imp = pd.DataFrame(imputer.transform(df), columns=columns)

    proba = model.predict_proba(df_imp)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame({"aps_probability": proba, "prediction": np.where(pred==1, "pos (APS)", "neg (other)")})
    st.dataframe(out)
    st.caption(f"Decision threshold in use: {threshold:.2f} (cost-optimized, FP cost=10, FN cost=500)")
