# -*- coding: utf-8 -*-
"""
SIOS - Smart Inventory Optimization System  |  Streamlit UI
Upload a demand dataset (or use the bundled sample) -> the app engineers features,
trains a GBDT with Holidays & Events, and shows interactive forecast charts.

Run:  C:/Temp/sios_venv/Scripts/streamlit run C:/Temp/sios/app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="SIOS Demand Forecasting", page_icon="📦", layout="wide")

NAVY="#1f3b5c"; GREEN="#548235"; ORANGE="#c55a11"; TEAL="#2a9d8f"; GREY="#9aa7b4"

# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def wape(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return 100*np.sum(np.abs(y-p))/np.sum(np.abs(y)) if np.sum(np.abs(y)) else np.nan

def mae(y,p): return float(np.mean(np.abs(np.asarray(y,float)-np.asarray(p,float))))
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))

@st.cache_data(show_spinner=True)
def engineer(df, holidays):
    df = df.copy()
    df["P"] = pd.PeriodIndex(df["Period"], freq="M")
    df = df.sort_values(["Item","Site","P"]).reset_index(drop=True)
    g = df.groupby(["Item","Site"], sort=False)["Demand"]
    for L in (1,2,3,12):
        df[f"lag_{L}"] = g.shift(L)
    df["roll_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    df["roll_mean_6"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
    df["roll_std_3"]  = g.transform(lambda s: s.shift(1).rolling(3).std())
    if holidays is not None:
        keep=[c for c in ["Period","Num_Holidays","Event_Intensity","Is_Event",
                          "Is_Holiday_Season","Is_Summer_Season"] if c in holidays.columns]
        df = df.merge(holidays[keep], on="Period", how="left")
    df["Quarter"] = df["P"].dt.quarter
    df["month_sin"] = np.sin(2*np.pi*df["Month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["Month"]/12)
    for c in ["Item","Site"]:
        df[c+"_code"] = df[c].astype("category").cat.codes
    if "Category" in df.columns:
        df["Category_code"] = df["Category"].astype("category").cat.codes
    df = df.dropna(subset=[f"lag_{L}" for L in (1,2,3,12)]+["roll_mean_6"]).reset_index(drop=True)
    return df

@st.cache_resource(show_spinner=True)
def train(df_key, df, feats, cat_feats, n_test):
    all_p = sorted(df["P"].unique()); test_p=set(all_p[-n_test:])
    tr=df[~df["P"].isin(test_p)]; te=df[df["P"].isin(test_p)]
    cat_idx=[feats.index(c) for c in cat_feats if c in feats]
    m=HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, categorical_features=cat_idx, random_state=42,
        early_stopping=True, validation_fraction=0.1)
    m.fit(tr[feats].values, tr["Demand"].values)
    te=te.copy(); te["pred"]=np.clip(m.predict(te[feats].values),0,None)
    return m, tr, te, sorted(str(p) for p in test_p)

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📦 SIOS")
st.sidebar.markdown("**Smart Inventory Optimization System**")
st.sidebar.markdown("---")
up = st.sidebar.file_uploader("Upload demand panel (CSV)", type="csv",
        help="Needs columns: Period (YYYY-MM), Item, Site, Demand. Optional: Category.")
up_h = st.sidebar.file_uploader("Upload Holidays & Events (CSV, optional)", type="csv")
n_test = st.sidebar.slider("Hold-out months (for testing)", 3, 12, 6)
use_hol = st.sidebar.checkbox("Use Holidays & Events features", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("No upload? The bundled 20k automotive sample is used.")

# ---------------------------------------------------------------- load data
if up is not None:
    df = load_csv(up); src="your upload"
else:
    df = load_csv(os.path.join(HERE,"sios_demand_panel.csv")); src="bundled sample (automotive, 20k rows)"

if up_h is not None:
    hol = load_csv(up_h)
elif os.path.exists(os.path.join(HERE,"sios_holidays.csv")):
    hol = load_csv(os.path.join(HERE,"sios_holidays.csv"))
else:
    hol = None

need={"Period","Item","Site","Demand"}
if not need.issubset(df.columns):
    st.error(f"Dataset must contain columns: {sorted(need)}. Found: {list(df.columns)}")
    st.stop()
if "Month" not in df.columns:
    df["Month"]=pd.PeriodIndex(df["Period"],freq="M").month
if "Year" not in df.columns:
    df["Year"]=pd.PeriodIndex(df["Period"],freq="M").year

# ---------------------------------------------------------------- header + KPIs
st.title("Smart Inventory Optimization System — Demand Forecasting")
st.markdown(f"Forecasting demand at the **Item × Warehouse × Month** grain · data source: *{src}*")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Items", df.Item.nunique())
c3.metric("Warehouses", df.Site.nunique())
c4.metric("Months", df.Period.nunique())
c5.metric("Total demand", f"{int(df.Demand.sum()):,}")

# ---------------------------------------------------------------- features + train
HOLIDAY_FEATS=["Num_Holidays","Event_Intensity","Is_Event","Is_Holiday_Season","Is_Summer_Season"]
fe = engineer(df, hol if use_hol else None)
BASE=["Month","Quarter","Year","month_sin","month_cos","lag_1","lag_2","lag_3","lag_12",
      "roll_mean_3","roll_mean_6","roll_std_3","Item_code","Site_code"]
if "Category_code" in fe.columns: BASE.append("Category_code")
CATF=[c for c in ["Item_code","Site_code","Category_code"] if c in fe.columns]
FEATS = BASE + [f for f in HOLIDAY_FEATS if (use_hol and f in fe.columns)]

with st.spinner("Engineering features and training the GBDT model…"):
    model, tr, te, test_months = train(f"{src}-{use_hol}-{n_test}-{len(fe)}", fe, FEATS, CATF, n_test)

ov_w = wape(te["Demand"], te["pred"])
sn_w = wape(te["Demand"], te["lag_12"])

# forecast-error spread per item-warehouse (drives the uncertainty band + safety stock)
te = te.copy(); te["resid"] = te["Demand"] - te["pred"]
global_sigma = float(te["resid"].std()) or 1.0
sigma_map = te.groupby(["Item","Site"])["resid"].std().to_dict()
def series_sigma(item, site):
    s = sigma_map.get((item, site))
    return global_sigma if (s is None or np.isnan(s)) else float(s)

tab1, tabInv, tab2, tab3, tab4 = st.tabs(
    ["🔮 Forecast Explorer","📦 Inventory & Reorder","📊 Model Performance","🧠 Feature Importance","📋 Data"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.subheader("Forecast vs Actual")
    cc1,cc2 = st.columns(2)
    item = cc1.selectbox("Item", sorted(fe.Item.unique()))
    site = cc2.selectbox("Warehouse", sorted(fe[fe.Item==item].Site.unique()))
    s = fe[(fe.Item==item)&(fe.Site==site)].sort_values("P")
    t = te[(te.Item==item)&(te.Site==site)].sort_values("P")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=s.Period, y=s.Demand, mode="lines+markers", name="Actual",
                             line=dict(color=NAVY,width=2)))
    if len(t):
        sig = series_sigma(item, site); band = 1.2816*sig   # ~P10–P90
        up_b = (t.pred + band); lo_b = (t.pred - band).clip(lower=0)
        fig.add_trace(go.Scatter(x=list(t.Period)+list(t.Period[::-1]),
                                 y=list(up_b)+list(lo_b[::-1]), fill="toself",
                                 fillcolor="rgba(197,90,17,0.15)", line=dict(width=0),
                                 name="forecast range (P10–P90)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t.Period, y=t.pred, mode="lines+markers", name="GBDT forecast",
                                 line=dict(color=ORANGE,width=3)))
        fig.add_vline(x=t.Period.iloc[0], line_dash="dash", line_color="grey")
    fig.update_layout(height=420, margin=dict(t=10,b=10), legend=dict(orientation="h"),
                      yaxis_title="Demand (units)", xaxis_title="Month")
    st.plotly_chart(fig, width="stretch")
    if len(t):
        m1,m2,m3=st.columns(3)
        m1.metric("This series WAPE", f"{wape(t.Demand,t.pred):.1f}%")
        m2.metric("MAE", f"{mae(t.Demand,t.pred):.1f}")
        m3.metric("Avg monthly demand", f"{s.Demand.mean():.0f}")
        if "Archetype" in s.columns:
            st.caption(f"Demand archetype for this item: **{s.Archetype.iloc[0]}**")

    st.markdown("#### Next-period forecast for every Item × Warehouse (hold-out average)")
    nxt = (te.groupby(["Item","Site"]).apply(
            lambda gr: pd.Series({"avg_actual": round(gr.Demand.mean()),
                                  "avg_forecast": round(gr.pred.mean()),
                                  "WAPE_%": round(wape(gr.Demand, gr.pred),1)}),
            include_groups=False).reset_index())
    st.dataframe(nxt, width="stretch", height=260)
    st.download_button("⬇️ Download all forecasts (CSV)",
        te[["Period","Item","Site","Demand","pred"]].rename(columns={"pred":"Forecast"}).to_csv(index=False),
        "sios_forecasts.csv","text/csv")

# ---------------------------------------------------------------- INVENTORY TAB
with tabInv:
    st.subheader("From forecast to inventory policy")
    st.caption("The forecast drives the safety stock and reorder point for every item-warehouse. "
               "Move the sliders to explore the service-level / inventory trade-off.")
    cA, cB = st.columns(2)
    lead = cA.slider("Supplier lead time (months)", 1, 6, 2)
    svc  = cB.slider("Target service level (%)", 80, 99, 95)
    z = float(norm.ppf(svc/100.0))

    rec = te.groupby(["Item","Site"]).agg(avg_forecast=("pred","mean")).reset_index()
    sig = te.groupby(["Item","Site"])["resid"].std().reset_index().rename(columns={"resid":"sigma"})
    rec = rec.merge(sig, on=["Item","Site"], how="left")
    rec["sigma"] = rec["sigma"].fillna(global_sigma)
    rec["safety_stock"]  = (z * rec["sigma"] * np.sqrt(lead)).round().clip(lower=0)
    rec["reorder_point"] = (rec["avg_forecast"]*lead + rec["safety_stock"]).round()
    rec["avg_forecast"]  = rec["avg_forecast"].round()
    rec = rec.sort_values("reorder_point", ascending=False)

    k1,k2,k3 = st.columns(3)
    k1.metric("Service-level factor (z)", f"{z:.2f}")
    k2.metric("Total safety stock (units)", f"{int(rec.safety_stock.sum()):,}")
    k3.metric("Avg reorder point", f"{rec.reorder_point.mean():.0f}")

    st.markdown("**Reorder recommendations** (formula: ROP = forecast × lead time + z·σ·√lead time)")
    st.dataframe(rec[["Item","Site","avg_forecast","sigma","safety_stock","reorder_point"]]
                 .rename(columns={"avg_forecast":"avg_monthly_forecast","sigma":"forecast_error_sd"}),
                 width="stretch", height=300)
    st.download_button("⬇️ Download reorder policy (CSV)", rec.to_csv(index=False),
                       "sios_reorder_policy.csv","text/csv")

    st.markdown("#### What-if: total safety stock vs. service level")
    levels = list(range(80,100))
    tot = [float((norm.ppf(l/100.0)*rec["sigma"]*np.sqrt(lead)).clip(lower=0).sum()) for l in levels]
    figc = go.Figure(go.Scatter(x=levels, y=tot, mode="lines+markers", line=dict(color=TEAL,width=2)))
    figc.add_vline(x=svc, line_dash="dash", line_color=ORANGE)
    figc.update_layout(height=320, xaxis_title="Service level (%)",
                       yaxis_title="Total safety stock (units)", margin=dict(t=10,b=10))
    st.plotly_chart(figc, width="stretch")
    st.info("Higher service levels cost disproportionately more buffer stock. Because a better forecast "
            "shrinks the error spread (σ), SIOS pushes this whole curve **down** — the same service level "
            "for less inventory, or a higher service level for the same inventory.")

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Baselines vs GBDT (hold-out test set)")
    rows={
        "Naive (last month)":      ("lag_1",),
        "Seasonal-naive (lag 12)": ("lag_12",),
        "Moving average (3m)":     ("roll_mean_3",),
    }
    data=[]
    for name,(col,) in rows.items():
        data.append([name, round(mae(te.Demand,te[col]),2), round(rmse(te.Demand,te[col]),2), round(wape(te.Demand,te[col]),2)])
    data.append(["GBDT (+ holidays)" if use_hol else "GBDT", round(mae(te.Demand,te.pred),2),
                 round(rmse(te.Demand,te.pred),2), round(ov_w,2)])
    perf=pd.DataFrame(data, columns=["Model","MAE","RMSE","WAPE_%"])
    cL,cR=st.columns([1.1,1])
    with cL:
        st.dataframe(perf, width="stretch", hide_index=True)
        st.metric("GBDT vs Seasonal-naive (WAPE)", f"{ov_w:.1f}%", f"{ov_w-sn_w:.1f} pp", delta_color="inverse")
    with cR:
        fig=go.Figure(go.Bar(x=perf["WAPE_%"], y=perf["Model"], orientation="h",
                             marker_color=[GREEN if "GBDT" in m else GREY for m in perf.Model]))
        fig.update_layout(height=320, xaxis_title="WAPE % (lower is better)", margin=dict(t=10,b=10))
        st.plotly_chart(fig, width="stretch")

    if "Archetype" in fe.columns:
        st.markdown("#### WAPE by demand archetype")
        arch=[]
        for a,grp in te.groupby("Archetype"):
            arch.append([a, len(grp), round(wape(grp.Demand,grp.lag_12),1), round(wape(grp.Demand,grp.pred),1)])
        adf=pd.DataFrame(arch, columns=["Archetype","n","Seasonal-naive WAPE_%","GBDT WAPE_%"]).sort_values("GBDT WAPE_%")
        fig=go.Figure()
        fig.add_trace(go.Bar(x=adf.Archetype, y=adf["Seasonal-naive WAPE_%"], name="Seasonal-naive", marker_color=GREY))
        fig.add_trace(go.Bar(x=adf.Archetype, y=adf["GBDT WAPE_%"], name="GBDT", marker_color=GREEN))
        fig.update_layout(height=380, barmode="group", yaxis_title="WAPE %", margin=dict(t=10,b=10))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("What drives the forecast?")
    st.caption("Permutation importance on a sample of the hold-out set (higher = more important).")
    samp=te.sample(min(1500,len(te)), random_state=0)
    pi=permutation_importance(model, samp[FEATS].values, samp["Demand"].values,
                              n_repeats=4, random_state=0, scoring="neg_mean_absolute_error")
    imp=pd.Series(pi.importances_mean, index=FEATS).sort_values().tail(14)
    fig=go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                  marker_color=[GREEN if f in HOLIDAY_FEATS else NAVY for f in imp.index]))
    fig.update_layout(height=460, xaxis_title="Increase in MAE when shuffled", margin=dict(t=10,b=10))
    st.plotly_chart(fig, width="stretch")
    st.info("Holiday / event features are shown in **green**. When present, they rank among the "
            "top drivers — quantifying the value of the external Holidays & Events feed.")

# ---------------------------------------------------------------- TAB 4
with tab4:
    st.subheader("Dataset preview")
    st.dataframe(df.head(500), width="stretch", height=360)
    cc1,cc2=st.columns(2)
    with cc1:
        st.markdown("**Demand by item (top 15)**")
        st.bar_chart(df.groupby("Item").Demand.sum().sort_values(ascending=False).head(15))
    with cc2:
        st.markdown("**Total demand by month**")
        st.line_chart(df.groupby("Period").Demand.sum())
