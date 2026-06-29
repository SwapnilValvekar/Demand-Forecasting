# -*- coding: utf-8 -*-
"""
SIOS demand-forecasting pipeline:
  1. load 20k panel + holidays calendar
  2. preprocessing + feature engineering (calendar, holiday, lags, rolling)
  3. walk-forward (out-of-time) train/test split
  4. baselines (naive, seasonal-naive, moving average) + GBDT
  5. GBDT WITH vs WITHOUT holiday features  ->  proves the Holidays & Events thesis
  6. evaluation: MAE, RMSE, WAPE, MASE (overall + by archetype) + charts
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))
FIGREPORT = os.path.join(HERE, "results"); os.makedirs(FIGREPORT, exist_ok=True)
BLUE="#1f4e79"; ORANGE="#c55a11"; GREEN="#548235"; TEAL="#2a9d8f"
plt.rcParams.update({"figure.dpi":150,"font.size":10,"axes.grid":True,"grid.alpha":0.3})

panel = pd.read_csv(os.path.join(HERE,"sios_demand_panel.csv"))
hol = pd.read_csv(os.path.join(HERE,"sios_holidays.csv"))
panel["P"] = pd.PeriodIndex(panel["Period"], freq="M")
panel = panel.sort_values(["Item","Site","P"]).reset_index(drop=True)

# ---------- feature engineering ----------
g = panel.groupby(["Item","Site"], sort=False)["Demand"]
for L in (1,2,3,12):
    panel[f"lag_{L}"] = g.shift(L)
# rolling features computed strictly on past values, within each item-site series
panel["roll_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
panel["roll_mean_6"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
panel["roll_std_3"]  = g.transform(lambda s: s.shift(1).rolling(3).std())

panel = panel.merge(hol[["Period","Num_Holidays","Event_Intensity","Is_Event","Is_Holiday_Season","Is_Summer_Season"]],
                    on="Period", how="left")
panel["Quarter"] = panel["P"].dt.quarter
panel["month_sin"] = np.sin(2*np.pi*panel["Month"]/12)
panel["month_cos"] = np.cos(2*np.pi*panel["Month"]/12)

# encode categoricals as integer codes (native categorical support in HistGBR)
cat_cols = ["Item","Site","Category"] + (["Climate","Region"] if "Climate" in panel.columns else [])
for c in cat_cols:
    panel[c+"_code"] = panel[c].astype("category").cat.codes

panel = panel.dropna(subset=[f"lag_{L}" for L in (1,2,3,12)]+["roll_mean_6"]).reset_index(drop=True)

HOLIDAY_FEATS = ["Num_Holidays","Event_Intensity","Is_Event","Is_Holiday_Season","Is_Summer_Season"]
GEO_FEATS = [c+"_code" for c in ["Climate","Region"] if c+"_code" in panel.columns]
BASE_FEATS = ["Month","Quarter","Year","month_sin","month_cos",
              "lag_1","lag_2","lag_3","lag_12","roll_mean_3","roll_mean_6","roll_std_3",
              "Promo_Flag","Unit_Price","Item_code","Site_code","Category_code"] + GEO_FEATS
ALL_FEATS = BASE_FEATS + HOLIDAY_FEATS
CAT_FEATS = ["Item_code","Site_code","Category_code"] + GEO_FEATS

# ---------- out-of-time split: last 6 months = test ----------
all_p = sorted(panel["P"].unique())
test_p = set(all_p[-6:])
train = panel[~panel["P"].isin(test_p)].copy()
test  = panel[panel["P"].isin(test_p)].copy()
print(f"Train rows: {len(train):,} | Test rows: {len(test):,} | "
      f"test months: {min(test_p)}..{max(test_p)}")

ytr, yte = train["Demand"].values, test["Demand"].values

def metrics(y, p, mae_naive=None):
    y=np.asarray(y,float); p=np.asarray(p,float)
    mae=np.mean(np.abs(y-p)); rmse=np.sqrt(np.mean((y-p)**2))
    wape=100*np.sum(np.abs(y-p))/np.sum(np.abs(y)) if np.sum(np.abs(y)) else np.nan
    out={"MAE":round(mae,2),"RMSE":round(rmse,2),"WAPE_%":round(wape,2)}
    if mae_naive: out["MASE"]=round(mae/mae_naive,3)
    return out

# ---------- baselines ----------
mae_snaive = np.mean(np.abs(yte - test["lag_12"].values))   # seasonal-naive scale for MASE
results={}
results["Naive (last month)"]      = metrics(yte, test["lag_1"].values,  mae_snaive)
results["Seasonal-naive (lag 12)"] = metrics(yte, test["lag_12"].values, mae_snaive)
results["Moving average (3m)"]     = metrics(yte, test["roll_mean_3"].values, mae_snaive)

def fit_gbdt(feats):
    cat_idx=[feats.index(c) for c in CAT_FEATS if c in feats]
    m=HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=None,
        max_leaf_nodes=31, l2_regularization=1.0, categorical_features=cat_idx,
        random_state=42, early_stopping=True, validation_fraction=0.1)
    m.fit(train[feats].values, ytr)
    return m

# GBDT without holidays
m_noh = fit_gbdt(BASE_FEATS)
pred_noh = np.clip(m_noh.predict(test[BASE_FEATS].values),0,None)
results["GBDT (no holidays)"] = metrics(yte, pred_noh, mae_snaive)

# GBDT with holidays
m_full = fit_gbdt(ALL_FEATS)
pred_full = np.clip(m_full.predict(test[ALL_FEATS].values),0,None)
results["GBDT (+ holidays)"] = metrics(yte, pred_full, mae_snaive)
test["pred"] = pred_full

res_df = pd.DataFrame(results).T
res_df.to_csv(os.path.join(FIGREPORT,"model_metrics.csv"))
print("\n================ MODEL COMPARISON (test set) ================")
print(res_df.to_string())

# holiday lift
wimp = (results["GBDT (no holidays)"]["WAPE_%"]-results["GBDT (+ holidays)"]["WAPE_%"])
print(f"\nHoliday-feature WAPE improvement: {wimp:.2f} percentage points "
      f"({100*wimp/results['GBDT (no holidays)']['WAPE_%']:.1f}% relative)")

# ---------- by archetype ----------
rows=[]
for arche,grp in test.groupby("Archetype"):
    y=grp["Demand"].values
    rows.append({"Archetype":arche,"n":len(grp),
                 "WAPE_snaive":round(100*np.sum(np.abs(y-grp['lag_12']))/np.sum(np.abs(y)),1),
                 "WAPE_GBDT":round(100*np.sum(np.abs(y-grp['pred']))/np.sum(np.abs(y)),1)})
arch_df=pd.DataFrame(rows).sort_values("WAPE_GBDT")
arch_df["improvement_pp"]=(arch_df["WAPE_snaive"]-arch_df["WAPE_GBDT"]).round(1)
arch_df.to_csv(os.path.join(FIGREPORT,"metrics_by_archetype.csv"),index=False)
print("\n============ WAPE by archetype: seasonal-naive vs GBDT ============")
print(arch_df.to_string(index=False))

# ---------- charts ----------
# 1) model comparison (WAPE)
fig,ax=plt.subplots(figsize=(7.5,3.8))
names=list(results.keys()); wapes=[results[n]["WAPE_%"] for n in names]
cols=[GREEN if "+ holidays" in n else (TEAL if "GBDT" in n else "#9aa7b4") for n in names]
ax.barh(names,wapes,color=cols)
for i,w in enumerate(wapes): ax.text(w+0.3,i,f"{w}%",va="center",fontsize=8)
ax.set_xlabel("WAPE (%) — lower is better"); ax.set_title("Forecast Accuracy: Baselines vs GBDT"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(os.path.join(FIGREPORT,"fig_model_compare.png"))
fig.savefig(os.path.join(FIGREPORT,"fig_model_compare.png")); plt.close()

# 2) by-archetype
fig,ax=plt.subplots(figsize=(8,4))
x=np.arange(len(arch_df)); w=0.38
ax.bar(x-w/2,arch_df["WAPE_snaive"],w,label="Seasonal-naive",color="#9aa7b4")
ax.bar(x+w/2,arch_df["WAPE_GBDT"],w,label="GBDT (+holidays)",color=GREEN)
ax.set_xticks(x); ax.set_xticklabels(arch_df["Archetype"],rotation=35,ha="right",fontsize=8)
ax.set_ylabel("WAPE (%)"); ax.set_title("Where GBDT Helps Most: WAPE by Demand Archetype"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGREPORT,"fig_archetype.png"))
fig.savefig(os.path.join(FIGREPORT,"fig_archetype.png")); plt.close()

# 3) permutation importance (sample for speed)
samp = test.sample(min(2000,len(test)), random_state=0)
pi = permutation_importance(m_full, samp[ALL_FEATS].values, samp["Demand"].values,
                            n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=ALL_FEATS).sort_values().tail(12)
fig,ax=plt.subplots(figsize=(7.5,4.2))
cols=[GREEN if f in HOLIDAY_FEATS else BLUE for f in imp.index]
ax.barh(imp.index,imp.values,color=cols)
ax.set_xlabel("Permutation importance (increase in MAE)"); ax.set_title("GBDT Feature Importance (holiday features in green)")
fig.tight_layout(); fig.savefig(os.path.join(FIGREPORT,"fig_importance.png"))
fig.savefig(os.path.join(FIGREPORT,"fig_importance.png")); plt.close()

# 4) forecast vs actual for two representative series
def plot_series(item, site, ax):
    s = panel[(panel.Item==item)&(panel.Site==site)].sort_values("P")
    ax.plot(range(len(s)), s["Demand"], marker="o", ms=3, lw=1.5, color=BLUE, label="Actual")
    tt = test[(test.Item==item)&(test.Site==site)].sort_values("P")
    idx=[list(s["Period"]).index(p) for p in tt["Period"]]
    ax.plot(idx, tt["pred"], marker="s", ms=4, lw=1.8, color=ORANGE, label="GBDT forecast")
    ax.axvline(len(s)-len(tt)-0.5, ls="--", color="grey", lw=1)
    ax.set_title(f"{item} @ {site}", fontsize=9); ax.legend(fontsize=7)
fig,axes=plt.subplots(2,1,figsize=(8,5.2))
plot_series("Floor Mat Set","Detroit Central",axes[0])      # holiday-driven
plot_series("Car Battery 12V","Chicago North",axes[1])      # seasonal-winter
for a in axes: a.set_ylabel("Demand"); a.set_xlabel("Month index")
fig.suptitle("Forecast vs Actual (dashed line = train/test split)",fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGREPORT,"fig_forecast_examples.png"))
fig.savefig(os.path.join(FIGREPORT,"fig_forecast_examples.png")); plt.close()

summary={"train_rows":int(len(train)),"test_rows":int(len(test)),
         "test_months":[str(min(test_p)),str(max(test_p))],
         "results":results,"holiday_wape_improvement_pp":round(wimp,2),
         "n_items":int(panel.Item.nunique()),"n_sites":int(panel.Site.nunique())}
json.dump(summary, open(os.path.join(FIGREPORT,"model_summary.json"),"w"), indent=2)
print("\nCharts saved: fig_model_compare, fig_archetype, fig_importance, fig_forecast_examples")
print("DONE")
