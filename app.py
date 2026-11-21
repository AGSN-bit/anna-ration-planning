"""ANNA: a local, synthetic-data decision-support demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anna.data import ITEMS, generate_batches, generate_history, get_item
from anna.forecasting import evaluate_holdout, forecast
from anna.inventory import allocate_fefo, inventory_summary, recommend_order, simulate


st.set_page_config(page_title="ANNA | Ration Planning Demo", page_icon="📦", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-family: Inter, "Segoe UI", sans-serif; color:#172b3a; }
.stApp { background:#f7f9fc; }
.block-container { padding-top:1.6rem; padding-bottom:2.5rem; max-width:1440px; }
h1, h2, h3, h4 { color:#142b3d; letter-spacing:-.025em; }
h2 { font-size:1.45rem; }
h3 { font-size:1.18rem; }
.hero { background:linear-gradient(112deg,#102b46 0%,#164c66 57%,#166b66 100%); color:#fff; padding:1.8rem 2rem; border-radius:18px; margin-bottom:1rem; box-shadow:0 10px 28px rgba(16,43,70,.14); }
.hero h1 { font-size:2.1rem; line-height:1.15; margin:.32rem 0 .45rem; color:#fff; }
.hero p { max-width:850px; margin:0; color:#e7f1f5; font-size:1.02rem; line-height:1.55; }
.eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:750; color:#a9eadb; }
.note { background:#eaf3f7; border:1px solid #bdd3df; border-left:5px solid #146b70; padding:.9rem 1.1rem; border-radius:10px; color:#193648; margin:.75rem 0 1.4rem; line-height:1.5; }
.vision-card { border:1px solid #cbd8e2; border-radius:14px; padding:1.1rem; background:#fff; min-height:155px; box-shadow:0 3px 12px rgba(18,43,61,.045); }
.vision-card h4 { margin:.6rem 0 .35rem; color:#18374b; }
.vision-card p { color:#40576a; line-height:1.5; margin:0; }
.vision-step { color:#075e59; font-weight:750; font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #d3dee7; padding:14px 16px; border-radius:13px; box-shadow:0 2px 8px rgba(18,43,61,.035); }
div[data-testid="stMetricLabel"] p { color:#4a6071 !important; font-size:.82rem; font-weight:650; }
div[data-testid="stMetricValue"] { color:#102b46 !important; font-weight:750; }
[data-testid="stSidebar"] { background:#edf2f6; border-right:1px solid #d6e0e8; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#102b46; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#344b5d; }
[data-testid="stTabs"] [role="tablist"] { gap:.35rem; border-bottom:1px solid #d2dde6; }
[data-testid="stTabs"] button[role="tab"] { color:#40586a; font-weight:650; padding:.65rem .9rem; border-radius:9px 9px 0 0; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#075e59; background:#e5f1ef; }
[data-testid="stAlert"] { border-radius:11px; }
div[data-testid="stDataFrame"] { border:1px solid #d4dee7; border-radius:11px; overflow:hidden; }
button[kind="secondary"], .stDownloadButton button { border:1px solid #9aafbf; color:#173b51; font-weight:650; border-radius:9px; }
button[kind="secondary"]:hover, .stDownloadButton button:hover { border-color:#075e59; color:#075e59; }
hr { border-color:#d9e2e9; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ANNA")
    st.caption("Predict ahead. Supply with precision.")
    st.divider()
    st.markdown("### Demo controls")
    seed = st.number_input("Synthetic data seed", min_value=1, max_value=99999, value=42, step=1)
    selected_item = st.selectbox("Ration category", [item.name for item in ITEMS])
    model = st.selectbox("Forecast method", ["Moving average", "Random Forest"])
    horizon = st.select_slider("Planning horizon", options=[7, 14, 30], value=14, format_func=lambda x: f"{x} days")
    st.markdown("### What-if scenario")
    delay = st.slider("Additional delivery delay", 0, 30, 7, 1, format="%d days")
    demand_multiplier = st.slider("Demand change", 0.8, 1.5, 1.15, 0.05, format="%.2f×")
    interruption = st.slider("Supply interruption", 0, 30, 0, 1, format="%d days")
    st.caption("Scenario ranges are demonstration controls, not real operating parameters.")

history = generate_history(int(seed))
batches_all = generate_batches(int(seed), history)
item = get_item(selected_item)
series_df = history[history.item == selected_item].sort_values("date").reset_index(drop=True)
batches = batches_all[batches_all.item == selected_item].copy().reset_index(drop=True)
dates = series_df.date
values = series_df.daily_consumption
as_of = pd.Timestamp(dates.iloc[-1]).normalize()
future = forecast(values, dates, 30, model, int(seed))
selected_forecast = future[:horizon]
lead_time = 7
# Size the illustrative order against the selected scenario's demand and effective lead time.
scenario_demand = selected_forecast * float(demand_multiplier)
scenario_demand[:min(int(interruption), len(scenario_demand))] *= 1.15
order = recommend_order(scenario_demand, batches, as_of,
                        lead_time=lead_time + int(delay), horizon=horizon)
summary = inventory_summary(batches, as_of)
scenario_no_order = simulate(batches, selected_forecast, as_of, lead_time, int(delay), float(demand_multiplier), int(interruption))
scenario_with_order = simulate(batches, selected_forecast, as_of, lead_time, int(delay), float(demand_multiplier), int(interruption), order["order_qty"])
baseline = simulate(batches, selected_forecast, as_of, lead_time, 0, 1.0, 0)
metrics = evaluate_holdout(values, dates, model, int(seed))

st.markdown(f"""
<div class="hero"><div class="eyebrow">Academic simulation · Synthetic data only</div>
<h1>ANNA ration planning demo</h1>
<p>A small prototype showing how forecasts, batch expiry and delivery disruption can shape a human-reviewed replenishment decision.</p></div>
""", unsafe_allow_html=True)
st.markdown("<div class='note'><b>Simulation boundary:</b> This demo uses generated data and illustrative assumptions. It is not an Army system, does not use operational data, and cannot place orders. Review every recommendation.</div>", unsafe_allow_html=True)

summary_col, reason_col = st.columns([1.65, 1])
with summary_col:
    st.subheader(f"{selected_item} outlook")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Usable stock", f"{summary['usable']:,.0f} {item.unit}")
    k2.metric(f"Forecast · {horizon}d", f"{sum(selected_forecast):,.0f} {item.unit}")
    k3.metric("Near expiry · 30d", f"{summary['near_expiry']:,.0f} {item.unit}")
    k4.metric("Suggested order", f"{order['order_qty']:,.0f} {item.unit}",
              delta="Review required", delta_color="off")
with reason_col:
    st.subheader("Why this suggestion?")
    st.markdown(f"- Current usable stock: **{summary['usable']:,.1f} {item.unit}**")
    st.markdown(f"- Forecast demand over {horizon} days: **{order['horizon_demand']:,.1f} {item.unit}**")
    st.markdown(f"- Safety stock allowance: **{order['safety_stock']:,.1f} {item.unit}**")
    st.markdown(f"- Simplified reorder point: **{order['reorder_point']:,.1f} {item.unit}**")
    st.caption(f"Order sizing uses the selected demand scenario and an illustrative {lead_time + int(delay)}-day effective lead time.")
    if order["needs_order"]:
        st.warning("Planning rule suggests an order for human review.")
    else:
        st.success("Usable stock is above the simplified reorder point.")

tab_overview, tab_forecast, tab_stock, tab_scenario, tab_vision = st.tabs(
    ["Overview", "Forecast", "Inventory & FEFO", "What-if simulator", "Bigger vision"]
)

with tab_overview:
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("#### Stock path under the selected scenario")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=baseline.date, y=baseline.closing_stock, name="Baseline · no disruption", line=dict(color="#27877b", width=3)))
        fig.add_trace(go.Scatter(x=scenario_no_order.date, y=scenario_no_order.closing_stock, name="What-if · no new order", line=dict(color="#d78237", width=3)))
        fig.add_trace(go.Scatter(x=scenario_with_order.date, y=scenario_with_order.closing_stock, name="What-if · suggested order", line=dict(color="#4266a8", width=3, dash="dash")))
        fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", font=dict(color="#263d4d", family="Inter, Segoe UI, sans-serif"), height=350, margin=dict(l=8,r=8,t=15,b=8), yaxis_title=f"Closing stock ({item.unit})", xaxis_title="Simulated date", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Scenario impact")
        no_order_short = float(scenario_no_order.shortage.sum())
        with_order_short = float(scenario_with_order.shortage.sum())
        st.metric("Shortage without proposed order", f"{no_order_short:,.1f} {item.unit}")
        st.metric("Shortage with proposed order", f"{with_order_short:,.1f} {item.unit}", delta=f"{no_order_short-with_order_short:,.1f} avoided in simulation")
        st.caption(f"Scenario: +{delay} day delivery delay · {demand_multiplier:.2f}× demand · {interruption} day interruption. Projection is based on the selected forecast and stated assumptions.")
    per_item = []
    for spec in ITEMS:
        sub = batches_all[batches_all.item == spec.name]
        s = inventory_summary(sub, as_of)
        per_item.append({"Item": spec.name, "Usable stock": f"{s['usable']:,.0f} {spec.unit}",
                         "Near expiry": f"{s['near_expiry']:,.0f} {spec.unit}", "Expiry assumption": f"{spec.shelf_life_days} days*"})
    st.markdown("#### Depot snapshot · synthetic inventory")
    st.dataframe(pd.DataFrame(per_item), hide_index=True, use_container_width=True)

with tab_forecast:
    st.markdown(f"#### Historical demand and {model.lower()} forecast")
    history_tail = series_df.tail(120)
    future_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon, freq="D")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_tail.date, y=history_tail.daily_consumption, name="Synthetic history", line=dict(color="#61758a", width=1.6)))
    fig.add_trace(go.Scatter(x=future_dates, y=selected_forecast, name="Forecast", line=dict(color="#27877b", width=3), mode="lines+markers"))
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", font=dict(color="#263d4d", family="Inter, Segoe UI, sans-serif"), height=360, margin=dict(l=8,r=8,t=20,b=8), yaxis_title=f"Daily demand ({item.unit})", xaxis_title="Date", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Chronological holdout evaluation")
    st.caption("Metrics below use the last 28 days as a holdout; the model is fitted only on earlier dates. They are calculated from this synthetic series, not entered as sample performance.")
    a,b,c = st.columns(3)
    a.metric("MAE", f"{metrics['MAE']:.2f} {item.unit}/day")
    b.metric("RMSE", f"{metrics['RMSE']:.2f} {item.unit}/day")
    c.metric("WAPE", f"{metrics['WAPE']:.1f}%")
    st.info("The moving-average method is the transparent benchmark. Random Forest is an optional prototype model; one synthetic holdout is not evidence of real-world forecasting performance.")
    export_forecast = pd.DataFrame({"forecast_date": future_dates, "item": selected_item, "model": model,
                                    "forecast_quantity": selected_forecast, "unit": item.unit, "synthetic": True})
    st.download_button("Download forecast CSV", export_forecast.to_csv(index=False).encode(), "anna_forecast_simulated.csv", "text/csv")

with tab_stock:
    st.markdown("#### Batch-level inventory")
    view = batches.copy()
    view["days_to_expiry"] = (view.expiry_date - as_of).dt.days
    view["status"] = np.where(view.days_to_expiry < 0, "Expired", np.where(view.days_to_expiry <= 30, "Near expiry", "Usable"))
    st.dataframe(view[["batch_id", "quantity", "received_date", "expiry_date", "days_to_expiry", "status"]].rename(columns={"quantity": f"Quantity ({item.unit})"}), hide_index=True, use_container_width=True)
    st.markdown("#### FEFO pick preview")
    preview, _, unfilled_pick = allocate_fefo(batches, order["mean_daily"] * min(7, horizon), as_of)
    preview["available"] = batches["quantity"]
    preview["planned_pick"] = preview["allocated"]
    preview["remaining_after_preview"] = preview["quantity"]
    st.dataframe(preview[["batch_id", "expiry_date", "available", "planned_pick", "remaining_after_preview"]].rename(columns={"available": f"Available ({item.unit})", "planned_pick": f"FEFO pick · 7d ({item.unit})", "remaining_after_preview": f"Remaining ({item.unit})"}), hide_index=True, use_container_width=True)
    if unfilled_pick > 0.01:
        st.warning(f"Synthetic usable stock cannot cover the seven-day preview by {unfilled_pick:,.1f} {item.unit}.")
    st.caption("FEFO means First Expired, First Out. The pick preview allocates the next seven forecast-demand days to the earliest-expiring batches first. Expiry values are illustrative synthetic assumptions.")

with tab_scenario:
    st.markdown("#### Compare baseline and disruption")
    comp = pd.DataFrame([
        {"Case": "Baseline", "Delay (days)": 0, "Demand multiplier": "1.00×", "Interruption (days)": 0,
         "Shortage": float(baseline.shortage.sum()), "Ending stock": float(baseline.closing_stock.iloc[-1])},
        {"Case": "Selected what-if · no new order", "Delay (days)": int(delay), "Demand multiplier": f"{demand_multiplier:.2f}×", "Interruption (days)": int(interruption),
         "Shortage": float(scenario_no_order.shortage.sum()), "Ending stock": float(scenario_no_order.closing_stock.iloc[-1])},
        {"Case": "Selected what-if · suggested order", "Delay (days)": int(delay), "Demand multiplier": f"{demand_multiplier:.2f}×", "Interruption (days)": int(interruption),
         "Shortage": float(scenario_with_order.shortage.sum()), "Ending stock": float(scenario_with_order.closing_stock.iloc[-1])},
    ])
    comp["Shortage"] = comp["Shortage"].map(lambda x: f"{x:,.1f} {item.unit}")
    comp["Ending stock"] = comp["Ending stock"].map(lambda x: f"{x:,.1f} {item.unit}")
    st.dataframe(comp, hide_index=True, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scenario_no_order.date, y=scenario_no_order.shortage, name="Daily shortage · no order", fill="tozeroy", line=dict(color="#d66b5d")))
    fig.add_trace(go.Scatter(x=scenario_with_order.date, y=scenario_with_order.shortage, name="Daily shortage · proposed order", line=dict(color="#4266a8", width=2)))
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", font=dict(color="#263d4d", family="Inter, Segoe UI, sans-serif"), height=320, margin=dict(l=8,r=8,t=20,b=8), yaxis_title=f"Unmet demand ({item.unit})", xaxis_title="Simulated date", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The simulator applies the selected demand multiplier and delivery delay to a copied synthetic scenario. An interruption increases modeled demand pressure during the selected initial period. This is a teaching scenario, not an operational forecast.")
    st.download_button("Download scenario results CSV", scenario_with_order.assign(item=selected_item, scenario="simulated", synthetic=True).to_csv(index=False).encode(), "anna_scenario_simulated.csv", "text/csv")

with tab_vision:
    st.markdown("#### From a focused demo to a broader logistics vision")
    a,b,c = st.columns(3)
    with a:
        st.markdown("<div class='vision-card'><div class='vision-step'>Now · prototype</div><h4>Prove the core workflow</h4><p>One simulated depot, generated data, explainable forecasts, FEFO, replenishment math and what-if comparisons.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='vision-card'><div class='vision-step'>Next · research validation</div><h4>Test with approved evidence</h4><p>Compare policies across seeds and scenarios; only later consider authorized, de-identified historical data in shadow mode.</p></div>", unsafe_allow_html=True)
    with c:
        st.markdown("<div class='vision-card'><div class='vision-step'>Longer term · vision</div><h4>Explore a multi-depot platform</h4><p>Regional profiles, stronger resilience and approved integrations are future possibilities subject to sponsorship, security review and validation.</p></div>", unsafe_allow_html=True)
    st.markdown("#### What this prototype does not claim")
    st.markdown("- It does not represent current Army software, ration scales, operational parameters, or actual depot performance.\n- It does not prove that a forecast improves outcomes or that deployment saves money.\n- It does not send orders or replace human planning and approval.")

st.divider()
st.caption(f"Run seed {seed} · As-of date {as_of.date()} · *Shelf-life values are illustrative prototype assumptions, not verified storage specifications.")
