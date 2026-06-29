# -*- coding: utf-8 -*-
"""
SIOS synthetic data generator.
Creates a realistic automotive spare-parts (TRADED items) monthly demand panel
at the Item x Site x Month grain, embedding eight demand archetypes so that a
GBDT with holiday/calendar/lag features can demonstrably beat a naive/ARIMA baseline.

Outputs (in this folder):
  sios_demand_panel.csv   ~20,000 rows  (the modelling base table)
  sios_holidays.csv       monthly Holidays & Events calendar (external feature)
  sios_items.csv          item master with assigned archetype
"""
import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- dimensions ----------------
START = pd.Period("2021-01", freq="M")
N_MONTHS = 50                      # 2021-01 .. 2025-02
periods = [START + i for i in range(N_MONTHS)]

# (name, demand-scale, region, climate)  -- climate drives seasonal demand by area
SITES = [
    ("Detroit Central",   1.40, "Midwest",   "cold"),
    ("Atlanta South",     1.10, "Southeast", "hot"),
    ("Dallas Hub",        1.25, "South",     "hot"),
    ("Reno West",         0.80, "West",      "temperate"),
    ("Chicago North",     1.30, "Midwest",   "cold"),
    ("Phoenix West",      0.95, "Southwest", "hot"),
    ("Denver Central",    0.90, "Mountain",  "cold"),
    ("Seattle Pacific",   1.05, "Northwest", "wet"),
    ("Miami East",        1.00, "Southeast", "hot"),
    ("Columbus Midwest",  0.85, "Midwest",   "cold"),
]
# climate amplifies/dampens the seasonal swing of weather-sensitive items
CLIM_WINTER = {"cold": 1.7, "temperate": 1.0, "wet": 1.3, "hot": 0.4}   # batteries, antifreeze, wipers
CLIM_SUMMER = {"hot": 1.7, "temperate": 1.0, "wet": 0.85, "cold": 0.5}  # coolant, cabin/air filters, radiator

# (name, category, archetype, base_level)
ITEMS = [
    ("Brake Pads Set","Brakes","stable",420),
    ("Brake Disc Rotor","Brakes","stable",260),
    ("Brake Caliper","Brakes","intermittent",70),
    ("Brake Fluid (1L)","Fluids","seasonal_summer",300),
    ("Oil Filter","Filters","stable",900),
    ("Air Filter","Filters","seasonal_summer",520),
    ("Cabin Air Filter","Filters","seasonal_summer",480),
    ("Fuel Filter","Filters","stable",300),
    ("Alternator","Electrical","stable",150),
    ("Starter Motor","Electrical","intermittent",90),
    ("Car Battery 12V","Electrical","seasonal_winter",560),
    ("Spark Plug (4-pack)","Electrical","stable",640),
    ("Ignition Coil","Electrical","stable",210),
    ("LED Headlight Bulb","Electrical","trend_up",380),
    ("Halogen Headlight","Electrical","trend_down",300),
    ("Wiper Motor","Electrical","seasonal_winter",120),
    ("Engine Oil 5W-30 (5L)","Fluids","stable",1100),
    ("Transmission Fluid (1L)","Fluids","stable",340),
    ("Coolant Antifreeze (1L)","Fluids","seasonal_winter",520),
    ("Power Steering Fluid","Fluids","stable",180),
    ("Timing Belt Kit","Engine","stable",170),
    ("Serpentine Belt","Engine","stable",240),
    ("Water Pump","Engine","intermittent",95),
    ("Radiator","Engine","seasonal_summer",230),
    ("Thermostat","Engine","stable",160),
    ("Fuel Pump","Engine","intermittent",110),
    ("Clutch Kit","Drivetrain","intermittent",60),
    ("CV Joint Boot","Drivetrain","stable",200),
    ("Shock Absorber","Suspension","stable",260),
    ("Strut Assembly","Suspension","stable",190),
    ("Tie Rod End","Suspension","stable",230),
    ("Ball Joint","Suspension","stable",210),
    ("Control Arm","Suspension","intermittent",85),
    ("Wheel Bearing","Suspension","stable",240),
    ("Windshield Wiper Set","Exterior","seasonal_winter",700),
    ("Side Mirror","Exterior","promo",140),
    ("Door Handle","Exterior","promo",120),
    ("Floor Mat Set","Accessories","holiday",260),
    ("Tire Pressure Sensor","Electronics","new_product",150),
    ("Dash Camera","Accessories","holiday",110),
]

# ---------------- Holidays & Events calendar (external feature) ----------------
# Regular year-end peak (Black Friday / holidays) PLUS irregular promo/clearance
# events whose MONTH shifts from year to year. Because the timing varies, a plain
# calendar (Month / lag-12) cannot predict them - only this external feed can.
US_HOLIDAYS_PER_MONTH = {1:2,2:1,3:0,4:0,5:1,6:1,7:1,8:0,9:1,10:1,11:2,12:1}  # approx US federal
EVENTS = {   # period -> event/promotion intensity (note the year-to-year drift)
    "2021-11":0.60,"2021-12":0.50,
    "2022-05":0.55,"2022-11":0.60,"2022-12":0.50,
    "2023-03":0.55,"2023-07":0.45,"2023-11":0.60,"2023-12":0.50,
    "2024-02":0.50,"2024-09":0.55,"2024-11":0.60,"2024-12":0.50,
    "2025-01":0.50,
}
EVP = np.array([EVENTS.get(str(p), 0.0) for p in periods])   # aligned to periods

hol_rows = []
for i, p in enumerate(periods):
    m = p.month
    hol_rows.append({
        "Period": str(p), "Year": p.year, "Month": m,
        "Num_Holidays": US_HOLIDAYS_PER_MONTH[m],
        "Event_Intensity": round(float(EVP[i]), 3),
        "Is_Event": int(EVP[i] > 0),
        "Is_Holiday_Season": int(m in (11, 12)),
        "Is_Summer_Season": int(m in (6, 7, 8)),
    })
hol = pd.DataFrame(hol_rows)
hol.to_csv(os.path.join(OUT, "sios_holidays.csv"), index=False)

# ---------------- demand generators per archetype ----------------
def gen_series(arche, base, site_scale, climate="temperate"):
    T = N_MONTHS
    t = np.arange(T)
    months = np.array([p.month for p in periods])
    ev = EVP                                   # per-period event intensity (year-varying)
    lvl = base * site_scale
    noise = rng.normal(0, 0.08, T)

    if arche == "stable":
        s = lvl * (1 + 0.05*np.sin(2*np.pi*t/12) + noise)
    elif arche == "trend_up":
        s = lvl * (0.6 + 0.9*t/T) * (1 + noise)
    elif arche == "trend_down":
        s = lvl * (1.3 - 0.7*t/T) * (1 + noise)
    elif arche == "seasonal_winter":   # peak Dec-Feb; swing amplified in cold areas
        amp = 0.45 * CLIM_WINTER.get(climate, 1.0)
        seas = amp*np.cos(2*np.pi*(months-1)/12)        # max at month 1 (Jan)
        s = lvl * (1 + seas) * (1 + 0.6*noise)
    elif arche == "seasonal_summer":   # peak Jun-Aug; swing amplified in hot areas
        amp = 0.45 * CLIM_SUMMER.get(climate, 1.0)
        seas = amp*np.cos(2*np.pi*(months-7)/12)         # max at month 7 (Jul)
        s = lvl * (1 + seas) * (1 + 0.6*noise)
    elif arche == "intermittent":      # lumpy, many low/zero months
        occur = rng.random(T) < 0.45
        sizes = rng.poisson(max(lvl,1), T)
        s = np.where(occur, sizes, rng.poisson(max(lvl*0.05,0.3), T))
    elif arche == "promo":             # baseline + event-driven promo spikes
        s = lvl * (1 + 0.4*noise)
        promo = rng.random(T) < 0.10
        s = s * np.where(promo, rng.uniform(1.8, 3.0, T), 1.0)
        s = s * (1 + 1.4*ev)           # promotions cluster on event periods (year-varying)
    elif arche == "holiday":           # spikes only during events; timing drifts across years
        s = lvl * (0.55 + 2.4*ev) * (1 + 0.4*noise)
    elif arche == "new_product":       # zeros then logistic ramp (cold start)
        launch = 14
        ramp = 1/(1+np.exp(-(t-(launch+8))/3.0))
        s = lvl * ramp * (1 + 0.5*noise)
        s[t < launch] = 0
    else:
        s = lvl * (1 + noise)

    # small universal holiday lift for everything + floor at 0
    s = s * (1 + 0.10*ev)
    s = np.clip(s, 0, None)
    return np.round(s).astype(int), promo if arche=="promo" else (rng.random(T) < 0.0)

# ---------------- build panel ----------------
rows = []
for it_name, cat, arche, base in ITEMS:
    for site, scale, region, climate in SITES:
        demand, promo_flags = gen_series(arche, base, scale, climate)
        # capacity cap -> lost sales (service constraint), returns ~1-3%
        cap = np.round(demand * rng.uniform(0.96, 1.15, N_MONTHS)).astype(int)
        issue = np.minimum(demand, cap)
        lost = demand - issue
        returns = rng.binomial(np.maximum(issue,0), rng.uniform(0.005, 0.03))
        price = round(float(rng.uniform(8, 600)), 2)
        for i, p in enumerate(periods):
            rows.append({
                "Period": str(p), "Year": p.year, "Month": p.month,
                "Item": it_name, "Category": cat, "Archetype": arche,
                "Site": site, "Region": region, "Climate": climate,
                "Demand": int(demand[i]),          # TARGET (true demand)
                "Actual_Issue": int(issue[i]),
                "Lost_Sales": int(lost[i]),
                "Returns": int(returns[i]),
                "Promo_Flag": int(bool(promo_flags[i])),
                "Unit_Price": price,
            })

panel = pd.DataFrame(rows)
panel.to_csv(os.path.join(OUT, "sios_demand_panel.csv"), index=False)

items_df = pd.DataFrame([{"Item":n,"Category":c,"Archetype":a,"Base_Level":b} for n,c,a,b in ITEMS])
items_df.to_csv(os.path.join(OUT, "sios_items.csv"), index=False)

# ---------------- summary ----------------
print("PANEL ROWS:", len(panel))
print("Items:", panel.Item.nunique(), "| Sites:", panel.Site.nunique(), "| Months:", panel.Period.nunique())
print("Series (item x site):", panel.Item.nunique()*panel.Site.nunique())
print("Date range:", panel.Period.min(), "->", panel.Period.max())
print("Total demand:", int(panel.Demand.sum()), "| Total lost sales:", int(panel.Lost_Sales.sum()))
print("\nArchetype mix:")
print(items_df.Archetype.value_counts().to_string())
print("\nDemand by archetype (mean monthly per series):")
print(panel.groupby("Archetype").Demand.mean().round(1).to_string())
print("\nSaved: sios_demand_panel.csv, sios_holidays.csv, sios_items.csv")
