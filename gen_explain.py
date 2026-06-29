# -*- coding: utf-8 -*-
"""Charts that explain HOW the SIOS dataset was constructed."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results"); os.makedirs(FIG, exist_ok=True)
BLUE="#1f4e79"; ORANGE="#c55a11"; GREEN="#548235"; TEAL="#2a9d8f"; NAVY="#1f3b5c"; GREY="#9aa7b4"
LBLUE="#dce6f1"; LORG="#fde9d9"; LGRN="#e2efda"; LGREY="#eef1f4"
plt.rcParams.update({"figure.dpi":150,"font.size":10,"axes.grid":True,"grid.alpha":0.3})

df = pd.read_csv(os.path.join(HERE,"sios_demand_panel.csv"))
hol = pd.read_csv(os.path.join(HERE,"sios_holidays.csv"))
months_idx = sorted(df["Period"].unique())

# ---------- 1) CLIMATE EFFECT: same item, cold vs hot warehouse ----------
def monthly_avg(item, site):
    d = df[(df.Item==item)&(df.Site==site)]
    return d.groupby("Month")["Demand"].mean().reindex(range(1,13))
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
mlabels=["J","F","M","A","M","J","J","A","S","O","N","D"]
# winter item
axes[0].plot(range(1,13),monthly_avg("Car Battery 12V","Detroit Central"),marker="o",color=BLUE,lw=2,label="Detroit (cold)")
axes[0].plot(range(1,13),monthly_avg("Car Battery 12V","Phoenix West"),marker="s",color=ORANGE,lw=2,label="Phoenix (hot)")
axes[0].set_title("Car Battery (winter item)\nbig winter peak in cold areas, flat in hot",fontsize=10)
axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(mlabels); axes[0].set_ylabel("Avg demand"); axes[0].legend(fontsize=8)
# summer item
axes[1].plot(range(1,13),monthly_avg("Cabin Air Filter","Phoenix West"),marker="o",color=ORANGE,lw=2,label="Phoenix (hot)")
axes[1].plot(range(1,13),monthly_avg("Cabin Air Filter","Detroit Central"),marker="s",color=BLUE,lw=2,label="Detroit (cold)")
axes[1].set_title("Cabin Air Filter (summer item)\nbig summer peak in hot areas, flat in cold",fontsize=10)
axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(mlabels); axes[1].legend(fontsize=8)
fig.suptitle("Climate-Driven Demand: the model learns 'what area → what demand'",fontsize=12,fontweight="bold",color=NAVY)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_climate_effect.png")); plt.close()

# ---------- 2) EVENT CALENDAR (year-varying) ----------
fig,ax=plt.subplots(figsize=(9.5,3.2))
ev=hol.set_index("Period")["Event_Intensity"]
cols=[ORANGE if v>0 else GREY for v in ev.values]
ax.bar(range(len(ev)),ev.values,color=cols)
ax.set_xticks(range(0,len(ev),3)); ax.set_xticklabels([ev.index[i] for i in range(0,len(ev),3)],rotation=45,ha="right",fontsize=8)
ax.set_title("Holidays & Events Calendar: event intensity by month (note the year-to-year drift)")
ax.set_ylabel("Event intensity")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_event_calendar.png")); plt.close()

# ---------- 3) ARCHETYPE GALLERY (one example series each) ----------
examples=[("stable","Oil Filter","Dallas Hub"),("trend_up","LED Headlight Bulb","Dallas Hub"),
          ("trend_down","Halogen Headlight","Dallas Hub"),("seasonal_winter","Car Battery 12V","Detroit Central"),
          ("seasonal_summer","Cabin Air Filter","Phoenix West"),("intermittent","Water Pump","Dallas Hub"),
          ("promo","Side Mirror","Dallas Hub"),("holiday","Floor Mat Set","Dallas Hub"),
          ("new_product","Tire Pressure Sensor","Dallas Hub")]
fig,axes=plt.subplots(3,3,figsize=(10,7))
for ax,(arche,item,site) in zip(axes.ravel(),examples):
    d=df[(df.Item==item)&(df.Site==site)].sort_values("Period")
    ax.plot(range(len(d)),d["Demand"].values,color=BLUE,lw=1.4)
    ax.fill_between(range(len(d)),d["Demand"].values,color=BLUE,alpha=0.12)
    ax.set_title(f"{arche}\n({item})",fontsize=8.5); ax.tick_params(labelsize=6); ax.set_xticks([])
fig.suptitle("The Eight Demand Archetypes Embedded in the Dataset (one example each)",fontsize=12,fontweight="bold",color=NAVY)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_archetype_gallery.png")); plt.close()

# ---------- 4) DATA-GENERATION FLOW DIAGRAM ----------
def box(ax,x,y,w,h,text,fc=LBLUE,fs=9,bold=True,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=2.2",lw=1.5,
                 edgecolor=NAVY,facecolor=fc,mutation_aspect=1,zorder=z))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color="#13293d",
            fontweight="bold" if bold else "normal",linespacing=1.35,zorder=z+1)
def arr(ax,p1,p2):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=18,lw=1.8,color=GREY,shrinkA=2,shrinkB=2,zorder=3))
fig,ax=plt.subplots(figsize=(10,4.6)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
ax.text(50,97,"How the Dataset Was Constructed",ha="center",va="top",fontsize=13,fontweight="bold",color=NAVY)
# three inputs on the left
box(ax,2,68,26,18,"40 Items x 8 archetypes\n(stable, trend, seasonal,\nintermittent, promo,\nholiday, new-product)",fc=LBLUE,fs=8)
box(ax,2,41,26,18,"10 Warehouses x climate\n(cold / hot / temperate / wet)\nin different regions",fc=LGRN,fs=8)
box(ax,2,14,26,18,"Holidays & Events calendar\n(year-varying event\nintensity per month)",fc=LORG,fs=8)
# generator
box(ax,40,41,22,18,"Demand generator\nper Item x Site\nx Month",fc=LGREY,fs=9)
# output
box(ax,72,52,26,16,"20,000-row panel\nItem x Site x Month\n(True Demand target)",fc=LBLUE,fs=8.5)
box(ax,72,24,26,16,"Train GBDT\n+ evaluate\n(MAE/RMSE/WAPE)",fc=LGRN,fs=8.5)
for y in (77,50,23): arr(ax,(28,y),(40,50))
arr(ax,(62,50),(72,60)); arr(ax,(85,52),(85,40))
fig.savefig(os.path.join(FIG,"dia_datagen.png"),bbox_inches="tight"); plt.close()

print("GEN-EXPLAIN charts:",[f for f in sorted(os.listdir(FIG)) if f in
      ("fig_climate_effect.png","fig_event_calendar.png","fig_archetype_gallery.png","dia_datagen.png")])
print("OK")
