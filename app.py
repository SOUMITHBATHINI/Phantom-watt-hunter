"""
=============================================================================
Phantom Watt Hunter  -  Hackathon MVP (single file)
-----------------------------------------------------------------------------
Flags SUSPICIOUS electricity use: power drawn by a zone that is supposed to
be idle. Stack: Python, Streamlit, Pandas, Plotly. No database, API key,
Arduino or internet needed.

Run:  pip install streamlit pandas plotly
      streamlit run app.py

IMPORTANT: demo data is SIMULATED. "Suspicious" means "worth checking",
NOT "confirmed waste". Only a person inspecting the room can confirm waste.
=============================================================================
"""

import random
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. SETTINGS
# -----------------------------------------------------------------------------
INTERVAL_MIN = 30  # demo data: one reading every 30 minutes
COLUMNS = ["Timestamp", "Zone", "Power_W", "Expected_Status"]
STATUSES = ["Active", "Idle"]

# Accepted header names for uploaded CSV files (case-insensitive)
COLUMN_ALIASES = {
    "timestamp": "Timestamp", "time": "Timestamp",
    "zone": "Zone",
    "power_w": "Power_W", "power": "Power_W",
    "status": "Expected_Status", "expected_status": "Expected_Status",
}

# Demo zones: typical Active / Idle power (W), active hours, and the power range
# of a simulated "equipment left on" event (anomaly) for that zone.
ZONE_PROFILES = {
    "Computer Lab 101":     {"active_w": 3200, "idle_w": 180, "start": 9, "end": 18, "anomaly": (1450, 1850)},
    "Electronics Workshop": {"active_w": 2600, "idle_w": 120, "start": 9, "end": 18, "anomaly": (950, 1300)},
    "Faculty Cabins":       {"active_w": 1400, "idle_w": 90,  "start": 9, "end": 18, "anomaly": (500, 900)},
    "Auditorium & Stage":   {"active_w": 4200, "idle_w": 150, "start": 9, "end": 18, "anomaly": (1600, 2100)},
    "Central Library Wing":  {"active_w": 2100, "idle_w": 140, "start": 9, "end": 18, "anomaly": (700, 1100)},
}


# -----------------------------------------------------------------------------
# 2. SIMULATED DEMO DATA
# -----------------------------------------------------------------------------
def generate_demo_data(seed: int) -> pd.DataFrame:
    """One simulated day (48 half-hour readings) for each zone.

    Normal behaviour: Active hours use a lot of power, Idle hours use a small
    standby load. Then we inject random 'phantom load' events (equipment left
    on) into idle hours. At least 3 zones always get one, so the demo is never
    empty. The same seed always gives the same data.
    """
    rng = random.Random(seed)
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    slots = 24 * 60 // INTERVAL_MIN
    zones = list(ZONE_PROFILES)
    must_have = set(rng.sample(zones, 3))  # these zones definitely get an anomaly

    rows = []
    for zone in zones:
        p = ZONE_PROFILES[zone]
        status, power = [], []
        for i in range(slots):
            hour = i * INTERVAL_MIN / 60
            if p["start"] <= hour < p["end"]:
                status.append("Active")
                power.append(max(200.0, rng.gauss(p["active_w"], p["active_w"] * 0.12)))
            else:
                status.append("Idle")
                power.append(max(40.0, rng.gauss(p["idle_w"], p["idle_w"] * 0.15)))

        idle_slots = [i for i, s in enumerate(status) if s == "Idle"]
        n_events = 1 if zone in must_have else (1 if rng.random() < 0.25 else 0)
        for _ in range(n_events):
            start = rng.choice(idle_slots)
            length = rng.randint(2, 8)  # 1 to 4 hours
            for j in range(start, min(start + length, slots)):
                if status[j] == "Idle":  # never spill into active hours
                    power[j] = rng.uniform(*p["anomaly"])

        for i in range(slots):
            rows.append({
                "Timestamp": midnight + timedelta(minutes=INTERVAL_MIN * i),
                "Zone": zone,
                "Power_W": round(power[i], 1),
                "Expected_Status": status[i],
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. DATA CLEANING (missing / invalid data)
# -----------------------------------------------------------------------------
def clean_readings(raw: pd.DataFrame):
    """Validate a readings table. Returns (clean_df, notes).
    Raises ValueError if columns are missing or no valid rows remain."""
    notes = []
    df = raw.rename(columns={c: COLUMN_ALIASES.get(str(c).strip().lower(), c) for c in raw.columns})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. "
                         "Need: timestamp, zone, power_w, status")

    df = df[COLUMNS].copy()
    total = len(df)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Power_W"] = pd.to_numeric(df["Power_W"], errors="coerce")
    zone_missing = df["Zone"].isna()  # check BEFORE converting to text
    df["Zone"] = df["Zone"].astype(str).str.strip()
    df["Expected_Status"] = df["Expected_Status"].astype(str).str.strip().str.title()

    ok = (df["Timestamp"].notna() & df["Power_W"].notna() & (df["Power_W"] >= 0)
          & df["Expected_Status"].isin(STATUSES) & ~zone_missing
          & (df["Zone"] != "") & (df["Zone"].str.lower() != "nan"))
    df = df[ok]
    if total - len(df):
        notes.append(f"Removed {total - len(df)} invalid row(s) (bad time, missing/negative power, "
                     "unknown status or empty zone).")

    before = len(df)
    df = df.drop_duplicates(subset=["Timestamp", "Zone"])
    if before - len(df):
        notes.append(f"Removed {before - len(df)} duplicate reading(s).")
    if df.empty:
        raise ValueError("No valid readings left after cleaning.")
    return df.sort_values(["Zone", "Timestamp"]).reset_index(drop=True), notes


def get_interval_hours(df: pd.DataFrame) -> float:
    """Typical gap between readings in hours (median per zone). Defaults to 30 min."""
    default = INTERVAL_MIN / 60
    diffs = df.groupby("Zone")["Timestamp"].diff().dropna()
    if diffs.empty:
        return default
    hours = diffs.median().total_seconds() / 3600
    return hours if hours > 0 else default


# -----------------------------------------------------------------------------
# 4. DETECTION, ENERGY, COST AND SAVINGS
# -----------------------------------------------------------------------------
def analyse(df: pd.DataFrame, interval_h: float, tariff: float,
            threshold_w: float, recoverable_pct: float) -> pd.DataFrame:
    """Adds flags, energy, cost and estimated savings to every reading.

    RULE:    suspicious = status is 'Idle' AND power > threshold.
    ENERGY:  kWh = watts x hours / 1000
    COST:    Rs  = kWh x tariff
    SAVINGS (estimate, not a guarantee):
      1. For suspicious readings only, take the power ABOVE the zone's normal
         idle level (median of its non-suspicious idle readings).
      2. Assume only `recoverable_pct`% of that could really be saved, since
         some of it may be a legitimate load (servers, security lights...).
    """
    out = df.copy()
    out["Energy_kWh"] = out["Power_W"] * interval_h / 1000
    out["Cost_INR"] = out["Energy_kWh"] * tariff
    out["Is_Suspicious"] = (out["Expected_Status"] == "Idle") & (out["Power_W"] > threshold_w)

    normal_idle = out[(out["Expected_Status"] == "Idle") & (~out["Is_Suspicious"])]
    baseline = normal_idle.groupby("Zone")["Power_W"].median()
    out["Idle_Baseline_W"] = out["Zone"].map(baseline).fillna(threshold_w)  # conservative fallback

    extra_w = (out["Power_W"] - out["Idle_Baseline_W"]).clip(lower=0).where(out["Is_Suspicious"], 0.0)
    out["Unexpected_kWh"] = extra_w * interval_h / 1000
    out["Savings_kWh"] = out["Unexpected_kWh"] * recoverable_pct / 100
    out["Savings_INR"] = out["Savings_kWh"] * tariff
    return out


def explain_event(zone: str, avg_w: float, peak_w: float, minutes: float, threshold_w: float):
    """Returns (reason, recommended_action). Causes are POSSIBLE, never confirmed."""
    reason = (f"{zone} is marked Idle but drew about {avg_w:.0f} W on average (peak {peak_w:.0f} W) "
              f"against a {threshold_w:.0f} W idle limit, for {minutes:.0f} min.")
    if peak_w >= 1500:
        reason += " Possible cause: AC, heater or heavy equipment left running."
        action = f"Inspect {zone}: check the AC/thermostat schedule and main switch; switch off if the room is empty."
    elif peak_w >= 800:
        reason += " Possible cause: computers, bench equipment or lighting left on."
        action = f"Ask security or staff to check switches in {zone}; consider a scheduled shutdown."
    else:
        reason += " Possible cause: standby creep from chargers or peripherals."
        action = "Check power-strip master switches and unplug unneeded devices."
    if minutes >= 120:
        action += " Long duration - check as a priority."
    return reason, action


def build_alerts(df: pd.DataFrame, interval_h: float, threshold_w: float) -> pd.DataFrame:
    """Merges consecutive suspicious readings in a zone into ONE alert event."""
    cols = ["Start", "End", "Zone", "Duration (min)", "Avg power (W)", "Peak power (W)",
            "Severity", "Reason", "Recommended action", "Est. savings (kWh)",
            "Est. savings (Rs)", "Detection status"]
    if not df["Is_Suspicious"].any():
        return pd.DataFrame(columns=cols)

    d = df.sort_values(["Zone", "Timestamp"]).copy()
    # run number changes whenever the suspicious flag flips inside a zone
    d["run"] = d.groupby("Zone")["Is_Suspicious"].transform(lambda s: s.ne(s.shift()).cumsum())

    events = []
    for (zone, _), g in d[d["Is_Suspicious"]].groupby(["Zone", "run"]):
        minutes = len(g) * interval_h * 60
        avg_w, peak_w = g["Power_W"].mean(), g["Power_W"].max()
        unexpected = g["Unexpected_kWh"].sum()
        severity = "High" if unexpected >= 2 else ("Medium" if unexpected >= 0.5 else "Low")
        reason, action = explain_event(zone, avg_w, peak_w, minutes, threshold_w)
        events.append({
            "Start": g["Timestamp"].min(),
            "End": g["Timestamp"].max() + pd.Timedelta(hours=interval_h),
            "Zone": zone, "Duration (min)": round(minutes),
            "Avg power (W)": round(avg_w), "Peak power (W)": round(peak_w),
            "Severity": severity, "Reason": reason, "Recommended action": action,
            "Est. savings (kWh)": round(g["Savings_kWh"].sum(), 3),
            "Est. savings (Rs)": round(g["Savings_INR"].sum(), 2),
            "Detection status": "Suspicious - needs verification",
        })
    return pd.DataFrame(events, columns=cols).sort_values("Start").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 5. DASHBOARD
# -----------------------------------------------------------------------------
CSS = """
<style>
div[data-testid="stMetric"] {background:#f8f9fa; border:1px solid #e2e8f0; padding:14px 18px;
    border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.06);}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#0f172a !important;}
.sim-banner {background:#eff6ff; border-left:5px solid #3b82f6; padding:12px 16px;
    border-radius:4px; margin-bottom:16px; font-size:0.95rem; color:#1e3a8a;}
.upl-banner {background:#fefce8; border-left:5px solid #eab308; padding:12px 16px;
    border-radius:4px; margin-bottom:16px; font-size:0.95rem; color:#713f12;}
</style>
"""


def load_data():
    """Returns (df, is_simulated, notes). Uses an uploaded CSV if valid, else demo data."""
    uploaded = st.sidebar.file_uploader(
        "Optional: upload your own CSV", type=["csv"],
        help="Columns: timestamp, zone, power_w, status (Active/Idle)")
    if uploaded is not None:
        try:
            df, notes = clean_readings(pd.read_csv(uploaded))
            return df, False, notes
        except Exception as error:  # bad file -> message + fall back to demo
            st.sidebar.error(f"Could not use the uploaded file: {error}")
    df, notes = clean_readings(st.session_state["demo_df"])
    return df, True, notes


def main():
    st.set_page_config(page_title="Phantom Watt Hunter", page_icon="⚡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if "seed" not in st.session_state:
        st.session_state["seed"] = 42
        st.session_state["demo_df"] = generate_demo_data(42)

    # ---------------- Sidebar ----------------
    st.sidebar.title("⚙️ Audit Parameters")
    tariff = st.sidebar.number_input("Electricity tariff (₹ per kWh)", 0.0, 100.0, 8.5, 0.25)
    threshold_w = st.sidebar.slider("Max allowed idle power (W)", 100, 1500, 400, 50,
                                    help="An Idle zone drawing MORE than this is flagged as suspicious.")
    recoverable_pct = st.sidebar.slider("Assumed recoverable share (%)", 10, 100, 60, 5,
                                        help="Share of the unexpected energy we assume could really be saved.")
    st.sidebar.divider()
    if st.sidebar.button("🔄 Generate new demo dataset"):
        st.session_state["seed"] = random.randint(100, 99999)
        st.session_state["demo_df"] = generate_demo_data(st.session_state["seed"])
    st.sidebar.caption(f"Demo seed: `{st.session_state['seed']}`")

    df, is_simulated, notes = load_data()
    for note in notes:
        st.sidebar.warning(note)
    all_zones = sorted(df["Zone"].unique())
    chosen = st.sidebar.multiselect("Zones to show", all_zones, default=all_zones)
    df = df[df["Zone"].isin(chosen)]

    # ---------------- Header ----------------
    st.title("⚡ Phantom Watt Hunter")
    st.markdown("##### Idle-hours electricity audit dashboard")
    if is_simulated:
        st.markdown('<div class="sim-banner"><strong>🧪 SIMULATED DATA:</strong> readings are generated by the '
                    'app (not measured). <strong>Suspicious ≠ confirmed waste:</strong> a flag only means an '
                    'idle zone drew more than your limit. Someone must inspect the room to confirm.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="upl-banner"><strong>📂 UPLOADED DATA:</strong> not verified by this app. '
                    '<strong>Suspicious ≠ confirmed waste</strong> - inspect before acting.</div>',
                    unsafe_allow_html=True)
    if df.empty:
        st.error("No zones selected. Pick at least one zone in the sidebar.")
        return

    interval_h = get_interval_hours(df)
    data = analyse(df, interval_h, tariff, threshold_w, recoverable_pct)
    alerts = build_alerts(data, interval_h, threshold_w)

    # ---------------- KPI cards ----------------
    total_kwh = data["Energy_kWh"].sum()
    savings_kwh = data["Savings_kWh"].sum()
    waste_pct = savings_kwh / total_kwh * 100 if total_kwh > 0 else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total energy", f"{total_kwh:,.1f} kWh")
    c2.metric("Estimated cost", f"₹ {data['Cost_INR'].sum():,.0f}", help=f"Energy × ₹{tariff:.2f}/kWh")
    c3.metric("Suspicious events", f"{len(alerts)}",
              delta=f"{int(data['Is_Suspicious'].sum())} flagged readings", delta_color="off",
              help="Consecutive suspicious readings in a zone count as one event.")
    c4.metric("Suspicious-reading energy",
              f"{data.loc[data['Is_Suspicious'], 'Energy_kWh'].sum():,.1f} kWh",
              help="All energy in flagged readings. NOT the same as savings.")
    c5.metric("Est. potential savings", f"₹ {data['Savings_INR'].sum():,.0f}",
              delta=f"{waste_pct:.1f}% of energy", delta_color="off",
              help=f"Only power above normal idle level, × {recoverable_pct}% assumed recoverable.")
    st.divider()

    tab_time, tab_zone, tab_alert, tab_data, tab_info = st.tabs(
        ["📈 Load timeline", "🏢 Zone comparison", "🚨 Alerts", "📄 Readings & export", "ℹ️ How it works"])

    # ---------------- Tab: timeline ----------------
    with tab_time:
        st.subheader("Power over time vs. idle threshold")
        fig = px.line(data, x="Timestamp", y="Power_W", color="Zone", template="plotly_white",
                      labels={"Power_W": "Power (W)", "Timestamp": "Time"})
        flagged = data[data["Is_Suspicious"]]
        if not flagged.empty:
            fig.add_trace(go.Scatter(
                x=flagged["Timestamp"], y=flagged["Power_W"], mode="markers",
                name="⚠️ Suspicious reading",
                marker={"symbol": "triangle-up", "size": 11, "color": "#dc2626",
                        "line": {"width": 1, "color": "#7f1d1d"}},
                text=flagged["Zone"] + ": " + flagged["Power_W"].round(0).astype(int).astype(str) + " W (Idle)",
                hoverinfo="text"))
        fig.add_hline(y=threshold_w, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Idle threshold ({threshold_w} W)", annotation_position="bottom right")
        fig.update_layout(height=450, margin={"l": 20, "r": 20, "t": 30, "b": 20},
                          legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "yanchor": "bottom"})
        st.plotly_chart(fig)

    # ---------------- Tab: zones ----------------
    with tab_zone:
        st.subheader("Zone comparison")
        left, right = st.columns(2)
        with left:
            st.markdown("**Energy by zone: normal vs. suspicious (kWh)**")
            by_zone = (data.assign(Type=data["Is_Suspicious"].map({True: "Suspicious", False: "Normal"}))
                       .groupby(["Zone", "Type"], as_index=False)["Energy_kWh"].sum())
            fig2 = px.bar(by_zone, x="Zone", y="Energy_kWh", color="Type", barmode="stack",
                          template="plotly_white",
                          color_discrete_map={"Normal": "#3b82f6", "Suspicious": "#ef4444"},
                          labels={"Energy_kWh": "Energy (kWh)"})
            fig2.update_layout(height=380, margin={"l": 20, "r": 20, "t": 30, "b": 20},
                               legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "yanchor": "bottom"})
            st.plotly_chart(fig2)
        with right:
            st.markdown("**Share of estimated savings by zone (₹)**")
            per_zone = data.groupby("Zone", as_index=False)["Savings_INR"].sum()
            if per_zone["Savings_INR"].sum() > 0:
                fig3 = px.pie(per_zone, names="Zone", values="Savings_INR", hole=0.45,
                              template="plotly_white", color_discrete_sequence=px.colors.sequential.Reds_r)
                fig3.update_traces(textposition="inside", textinfo="percent+label")
                fig3.update_layout(height=380, margin={"l": 20, "r": 20, "t": 30, "b": 20}, showlegend=False)
                st.plotly_chart(fig3)
            else:
                st.info("No estimated savings at the current threshold.")

        summary = data.groupby("Zone").agg(
            Energy_kWh=("Energy_kWh", "sum"), Cost_Rs=("Cost_INR", "sum"),
            Suspicious_readings=("Is_Suspicious", "sum"), Est_savings_Rs=("Savings_INR", "sum")).round(2)
        summary.insert(2, "Suspicious_events", alerts["Zone"].value_counts().reindex(summary.index).fillna(0).astype(int))
        st.dataframe(summary.reset_index())

    # ---------------- Tab: alerts ----------------
    with tab_alert:
        st.subheader(f"🚨 Alerts ({len(alerts)} suspicious events)")
        if alerts.empty:
            st.success("No suspicious idle consumption at the current threshold.")
        else:
            table = alerts.drop(columns=["Reason", "Recommended action"]).copy()
            table["Start"] = table["Start"].dt.strftime("%I:%M %p")
            table["End"] = table["End"].dt.strftime("%I:%M %p")
            st.dataframe(table, hide_index=True)
            st.download_button("⬇️ Download alerts (CSV)", alerts.to_csv(index=False).encode("utf-8"),
                               file_name="phantom_watt_alerts.csv", mime="text/csv")

            st.markdown("**Details for the biggest events** (top 10 by estimated savings)")
            for _, a in alerts.sort_values("Est. savings (Rs)", ascending=False).head(10).iterrows():
                title = (f"[{a['Severity']}] {a['Zone']} · {a['Start']:%I:%M %p}–{a['End']:%I:%M %p} · "
                         f"avg {a['Avg power (W)']} W")
                with st.expander(title):
                    x, y = st.columns([1, 2])
                    x.write(f"**Duration:** {a['Duration (min)']} min")
                    x.write(f"**Peak power:** {a['Peak power (W)']} W")
                    x.write(f"**Idle limit:** {threshold_w} W")
                    x.write(f"**Est. savings:** ₹ {a['Est. savings (Rs)']:.2f}")
                    y.markdown(f"**Why flagged:** {a['Reason']}")
                    y.markdown(f"**Recommended action:** {a['Recommended action']}")
                    y.caption("Status: suspicious, NOT confirmed waste.")

    # ---------------- Tab: readings ----------------
    with tab_data:
        st.subheader("Readings")
        choice = st.selectbox("Show", ["All readings", "Suspicious only", "Normal only"])
        view = data
        if choice == "Suspicious only":
            view = data[data["Is_Suspicious"]]
        elif choice == "Normal only":
            view = data[~data["Is_Suspicious"]]
        view = view.sort_values(["Timestamp", "Zone"])
        st.dataframe(pd.DataFrame({
            "Timestamp": view["Timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
            "Zone": view["Zone"], "Power (W)": view["Power_W"],
            "Expected activity": view["Expected_Status"],
            "Energy (kWh)": view["Energy_kWh"].round(3),
            "Detection": view["Is_Suspicious"].map({True: "⚠️ Suspicious", False: "OK"}),
        }), hide_index=True)
        st.download_button("⬇️ Download readings (CSV)",
                           data.drop(columns=["Idle_Baseline_W"]).to_csv(index=False).encode("utf-8"),
                           file_name="phantom_watt_readings.csv", mime="text/csv")

    # ---------------- Tab: how it works ----------------
    with tab_info:
        st.markdown(f"""
- **Energy (kWh)** = power (W) × hours ÷ 1000. Each reading covers **{interval_h * 60:.0f} min**.
- **Cost (₹)** = energy (kWh) × tariff (₹/kWh).
- **Suspicious reading** = status is *Idle* **and** power > **{threshold_w} W**.
- **Event** = consecutive suspicious readings in one zone, merged into one alert.
  Severity from unexpected energy: High ≥ 2 kWh, Medium ≥ 0.5 kWh, else Low.
- **Estimated savings** = (power above the zone's *normal idle level*) × time × tariff × **{recoverable_pct}%**.
  Only suspicious readings count, and only the extra power. The percentage is an assumption.
- **Suspicious ≠ confirmed waste.** "Possible causes" in alerts are guesses from the power level only.
""")


if __name__ == "__main__":
    main()
