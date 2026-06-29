# -*- coding: utf-8 -*-
"""Rich matplotlib distribution / EDA charts on the 20k modelling dataset."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results"); os.makedirs(FIG, exist_ok=True)
BLUE="#1f4e79"; ORANGE="#c55a11"; GREEN="#548235"; TEAL="#2a9d8f"; NAVY="#1f3b5c"
plt.rcParams.update({"figure.dpi":150,"font.size":10,"axes.grid":True,"grid.alpha":0.3})

df = pd.read_csv(os.path.join(HERE,"sios_demand_panel.csv"))
hol = pd.read_csv(os.path.join(HERE,"sios_holidays.csv"))
df["P"]=pd.PeriodIndex(df["Period"],freq="M")
df=df.sort_values(["Item","Site","P"])
g=df.groupby(["Item","Site"],sort=False)["Demand"]
for L in (1,3,12): df[f"lag_{L}"]=g.shift(L)
df["roll_mean_3"]=g.transform(lambda s:s.shift(1).rolling(3).mean())
df["roll_mean_6"]=g.transform(lambda s:s.shift(1).rolling(6).mean())
df=df.merge(hol[["Period","Num_Holidays","Event_Intensity"]],on="Period",how="left")

# 1) Demand distribution (histogram) — shows the right-skew typical of demand data
fig,ax=plt.subplots(figsize=(8,3.6))
ax.hist(df["Demand"],bins=60,color=BLUE,edgecolor="white")
ax.axvline(df["Demand"].mean(),ls="--",color=ORANGE,lw=1.6,label=f"mean = {df['Demand'].mean():.0f}")
ax.axvline(df["Demand"].median(),ls="--",color=GREEN,lw=1.6,label=f"median = {df['Demand'].median():.0f}")
ax.set_title("Distribution of Monthly Demand (all item-warehouse-months)")
ax.set_xlabel("Demand (units)"); ax.set_ylabel("Frequency"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_dist_hist.png")); plt.close()

# 2) Boxplot of demand by archetype — visualises the eight planted patterns' spread
order=["stable","trend_up","trend_down","seasonal_winter","seasonal_summer","promo","holiday","new_product","intermittent"]
order=[a for a in order if a in df.Archetype.unique()]
data=[df.loc[df.Archetype==a,"Demand"].values for a in order]
fig,ax=plt.subplots(figsize=(9,4))
bp=ax.boxplot(data,vert=True,patch_artist=True,showfliers=False,widths=0.6)
for patch in bp["boxes"]: patch.set_facecolor("#cfe0f1"); patch.set_edgecolor(NAVY)
for med in bp["medians"]: med.set_color(ORANGE); med.set_linewidth(2)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order,rotation=30,ha="right",fontsize=9)
ax.set_title("Monthly Demand Distribution by Demand Archetype"); ax.set_ylabel("Demand (units)")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_box_archetype.png")); plt.close()

# 3) Correlation heatmap of model features
cols=["Demand","lag_1","lag_3","lag_12","roll_mean_3","roll_mean_6","Event_Intensity","Num_Holidays","Month"]
corr=df[cols].corr()
fig,ax=plt.subplots(figsize=(7.6,6.4))
im=ax.imshow(corr.values,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha="right",fontsize=8.5)
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols,fontsize=8.5)
for i in range(len(cols)):
    for j in range(len(cols)):
        v=corr.values[i,j]
        ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7.5,
                color="white" if abs(v)>0.55 else "black")
ax.set_title("Feature Correlation Heatmap"); ax.grid(False)
fig.colorbar(im,fraction=0.046,pad=0.04); fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_corr_heatmap.png")); plt.close()

# 4) Seasonality heatmap: average demand by month-of-year x category
piv=df.pivot_table(index="Category",columns="Month",values="Demand",aggfunc="mean")
fig,ax=plt.subplots(figsize=(9,4.2))
im=ax.imshow(piv.values,cmap="YlOrRd",aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],fontsize=8.5)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=8.5)
ax.set_title("Seasonality: Average Demand by Month and Product Category"); ax.grid(False)
fig.colorbar(im,fraction=0.03,pad=0.02,label="Avg demand"); fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_season_heatmap.png")); plt.close()

# 5) Total demand by category
cat=df.groupby("Category")["Demand"].sum().sort_values(ascending=False)
fig,ax=plt.subplots(figsize=(8,3.6))
ax.bar(cat.index,cat.values,color=TEAL)
ax.set_title("Total Demand by Product Category"); ax.set_ylabel("Units"); plt.xticks(rotation=25,ha="right",fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_cat_demand.png")); plt.close()

# 6) Predictive power of seasonality: Demand vs lag-12 scatter
s=df.dropna(subset=["lag_12"]).sample(min(4000,len(df)),random_state=1)
fig,ax=plt.subplots(figsize=(6.4,5.2))
ax.scatter(s["lag_12"],s["Demand"],s=6,alpha=0.25,color=BLUE)
lim=max(s["lag_12"].max(),s["Demand"].max())
ax.plot([0,lim],[0,lim],ls="--",color=ORANGE,lw=1.5,label="y = x (perfect seasonal repeat)")
r=np.corrcoef(s["lag_12"],s["Demand"])[0,1]
ax.set_title(f"Demand vs. Same Month Last Year (lag-12)\ncorrelation = {r:.2f}")
ax.set_xlabel("Demand 12 months ago"); ax.set_ylabel("Demand now"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_lag_scatter.png")); plt.close()

print("DIST CHARTS:", [f for f in sorted(os.listdir(FIG)) if f.startswith(("fig_dist","fig_box","fig_corr","fig_season","fig_cat","fig_lag"))])
print("OK")
