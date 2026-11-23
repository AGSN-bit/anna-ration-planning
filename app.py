"""ANNA logistics decision-support prototype using synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anna.data import ITEMS, generate_batches, generate_history, get_item
from anna.forecasting import evaluate_holdout, forecast
from anna.inventory import (
    allocate_fefo,
    apply_demand_scenario,
    inventory_summary,
    recommend_order,
    simulate_policy,
    summarize_simulation,
)


st.set_page_config(page_title="ANNA | Logistics Control Tower", page_icon="📦", layout="wide")
st.markdown(
    """
<style>
:root, [data-theme="light"], [data-theme="dark"] {
  color-scheme: light;
  --text-color: #183047;
  --background-color: #f5f8fb;
  --secondary-background-color: #eaf0f5;
  --primary-color: #0a8178;
}
html, body, .stApp { background:#f5f8fb; color:#183047; color-scheme:light; }
.block-container { max-width:1480px; padding:1.6rem 2rem 3rem; }
[data-testid="stSidebar"] { background:#eaf0f5; border-right:1px solid #d3dee7; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] label { color:#30495c !important; opacity:1 !important; }
h1,h2,h3,h4,[data-testid="stHeading"] { color:#122d46 !important; opacity:1 !important; letter-spacing:-.025em; }
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stCaptionContainer"] p { color:#30495c !important; opacity:1 !important; }
.stApp [data-testid="stMarkdownContainer"] strong { color:#17364b !important; }
.brand { background:linear-gradient(120deg,#102d48,#15506a 62%,#0b766f); color:#fff; padding:1.35rem 1.6rem; border-radius:17px; margin:0 0 1rem; box-shadow:0 9px 25px rgba(16,45,72,.12); }
.brand .eyebrow { color:#9ce5d7; font-weight:750; letter-spacing:.13em; text-transform:uppercase; font-size:.7rem; }
.brand h1 { color:#fff !important; font-size:1.9rem; margin:.35rem 0 .35rem; }
.brand p { color:#e3eef3 !important; margin:0; max-width:900px; line-height:1.5; }
.pill { display:inline-block; border-radius:999px; padding:.22rem .62rem; background:#d9f1ec; color:#075f58; font-size:.72rem; font-weight:750; letter-spacing:.04em; }
.section-kicker { color:#08766d; font-size:.72rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.2rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #d5e0e8; padding:13px 15px; border-radius:13px; box-shadow:0 2px 8px rgba(18,43,61,.035); }
div[data-testid="stMetricLabel"] p { color:#496174 !important; font-size:.8rem; font-weight:650; }
div[data-testid="stMetricValue"] { color:#122d46 !important; font-weight:760; }
div[data-testid="stMetricDelta"] { font-size:.78rem; }
[data-testid="stTabs"] button[role="tab"] { color:#40596b; font-weight:650; }
[data-testid="stAlert"] { border-radius:12px; }
div[data-testid="stDataFrame"] { border:1px solid #d5e0e8; border-radius:11px; overflow:hidden; }
.stButton button, .stFormSubmitButton button, .stDownloadButton button { border-radius:9px; font-weight:700; }
hr { border-color:#d7e1e8; }
@media (max-width: 800px) { .block-container { padding:.9rem 1rem 2rem; } .brand { padding:1rem 1.1rem; } .brand h1 { font-size:1.55rem; } }
</style>
""",
    unsafe_allow_html=True,
)


PAGE_NAMES = [
    "Overview",
    "Demand Forecast",
    "Inventory & FEFO",
    "Scenario Simulator",
    "Recommendations",
]
DEFAULT_SCENARIO = {
    "lead_time_days": 7,
    "additional_delay_days": 7,
    "demand_multiplier": 1.15,
    "interruption_days": 0,
    "horizon_days": 14,
}
SERVICE_FACTOR = 1.28  # Illustrative z-value, approximately 90% under normal assumptions.


def quantity(value: float, unit: str, digits: int = 0) -> str:
    return f"{value:,.{digits}f} {unit}"


def stock_chart(
    baseline: pd.DataFrame,
    no_order: pd.DataFrame,
    proposed: pd.DataFrame,
    unit: str,
    arrival_date: pd.Timestamp | pd.NaT,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=baseline.date, y=baseline.closing_stock, name="Baseline · no scenario", line={"color": "#64788a", "width": 2.2}))
    figure.add_trace(go.Scatter(x=no_order.date, y=no_order.closing_stock, name="Selected scenario · no order", line={"color": "#d27a35", "width": 2.8}))
    figure.add_trace(go.Scatter(x=proposed.date, y=proposed.closing_stock, name="Selected scenario · proposed order", line={"color": "#087d73", "width": 3.2}))
    if not pd.isna(arrival_date) and arrival_date <= proposed.date.max():
        figure.add_vline(x=arrival_date, line_width=1.5, line_dash="dot", line_color="#315ca8", annotation_text="Order arrival", annotation_position="top")
    figure.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
        font={"color": "#2e4659", "family": "Inter, Segoe UI, sans-serif"},
        height=360, margin={"l": 8, "r": 12, "t": 24, "b": 8},
        yaxis_title=f"Usable closing stock ({unit})", xaxis_title="Date",
        legend={"orientation": "h", "y": -0.24}, hovermode="x unified",
    )
    figure.update_yaxes(rangemode="tozero", gridcolor="#e8eef2")
    figure.update_xaxes(gridcolor="#f0f3f6")
    return figure


def demand_chart(
    dates: pd.DatetimeIndex,
    baseline: np.ndarray,
    adjusted: np.ndarray,
    unit: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dates, y=baseline, name="Baseline forecast", line={"color": "#17618a", "width": 2.8}))
    figure.add_trace(go.Scatter(x=dates, y=adjusted, name="Scenario-adjusted demand", line={"color": "#d27a35", "width": 2.8, "dash": "dash"}))
    figure.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
        font={"color": "#2e4659", "family": "Inter, Segoe UI, sans-serif"},
        height=360, margin={"l": 8, "r": 12, "t": 20, "b": 8},
        yaxis_title=f"Daily demand ({unit})", xaxis_title="Date",
        legend={"orientation": "h", "y": -0.24}, hovermode="x unified",
    )
    figure.update_yaxes(rangemode="tozero", gridcolor="#e8eef2")
    figure.update_xaxes(gridcolor="#f0f3f6")
    return figure


st.session_state.setdefault("active_scenario", DEFAULT_SCENARIO.copy())
active_scenario = dict(st.session_state.active_scenario)

with st.sidebar:
    st.markdown("## ANNA")
    st.caption("LOGISTICS DECISION SUPPORT")
    st.markdown("---")
    page = st.radio("Workspace", PAGE_NAMES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### Select a supply item")
    seed = st.number_input("Synthetic data seed", min_value=1, max_value=99_999, value=42, step=1)
    selected_item = st.selectbox("Ration category", [item.name for item in ITEMS])
    model = st.selectbox("Forecast method", ["Moving average", "Random Forest"])
    st.markdown("---")
    st.caption("Synthetic demonstration · no operational data or order transmission")


history = generate_history(int(seed))
batches_all = generate_batches(int(seed), history)
item = get_item(selected_item)
item_history = history.loc[history.item == selected_item].sort_values("date").reset_index(drop=True)
batches = batches_all.loc[batches_all.item == selected_item].copy().reset_index(drop=True)
dates = pd.DatetimeIndex(item_history.date)
values = item_history.daily_consumption.astype(float)
as_of = dates[-1].normalize()
baseline_90 = forecast(values, pd.Series(dates), 90, model, int(seed))
history_std = float(values.tail(90).std(ddof=1))
if not np.isfinite(history_std):
    history_std = 0.0

scenario = dict(active_scenario)
horizon_days = int(scenario["horizon_days"])
effective_lead_days = int(scenario["lead_time_days"] + scenario["additional_delay_days"])
baseline_forecast = baseline_90[:horizon_days]
scenario_forecast = apply_demand_scenario(
    baseline_forecast,
    float(scenario["demand_multiplier"]),
    int(scenario["interruption_days"]),
)
order_plan = recommend_order(
    scenario_forecast,
    batches,
    as_of,
    lead_time_days=effective_lead_days,
    horizon_days=horizon_days,
    daily_demand_std=history_std,
    service_factor=SERVICE_FACTOR,
    confirmed_incoming=0.0,
)
arrival_date = as_of + pd.Timedelta(days=effective_lead_days) if float(order_plan["order_qty"]) > 1e-6 else pd.NaT
baseline_run = simulate_policy(
    batches, baseline_forecast, as_of, horizon_days, effective_lead_days,
    additional_delay_days=0, order_qty=0.0,
    order_shelf_life_days=item.shelf_life_days,
)
no_order_run = simulate_policy(
    batches, scenario_forecast, as_of, horizon_days, int(scenario["lead_time_days"]),
    additional_delay_days=int(scenario["additional_delay_days"]), order_qty=0.0,
    order_shelf_life_days=item.shelf_life_days,
)
proposed_run = simulate_policy(
    batches, scenario_forecast, as_of, horizon_days, int(scenario["lead_time_days"]),
    additional_delay_days=int(scenario["additional_delay_days"]),
    order_qty=float(order_plan["order_qty"]), order_shelf_life_days=item.shelf_life_days,
)
baseline_summary = summarize_simulation(baseline_run)
no_order_summary = summarize_simulation(no_order_run)
proposed_summary = summarize_simulation(proposed_run)
current_stock = inventory_summary(batches, as_of)
mean_forecast = float(np.mean(baseline_forecast)) if len(baseline_forecast) else 0.0
days_cover = current_stock["usable"] / mean_forecast if mean_forecast > 0 else float("inf")
order_improvement = float(no_order_summary["shortage"] - proposed_summary["shortage"])

st.markdown(
    f"""
<div class="brand"><div class="eyebrow">ANNA · Depot decision workspace</div>
<h1>{page}</h1>
<p>{selected_item} · {item.unit} · As of {as_of.strftime('%d %b %Y')} &nbsp; <span class="pill">SYNTHETIC SIMULATION</span></p></div>
""",
    unsafe_allow_html=True,
)
st.info("Illustrative synthetic data only. This prototype is not an operational Army system and does not create or transmit orders.", icon="ℹ️")


if page == "Overview":
    st.markdown('<div class="section-kicker">Current position</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable stock", quantity(current_stock["usable"], item.unit))
    c2.metric("Days of cover", f"{days_cover:.1f} days" if np.isfinite(days_cover) else "—")
    c3.metric(f"Baseline demand · {horizon_days}d", quantity(float(baseline_forecast.sum()), item.unit))
    c4.metric("Expiring within 30 days", quantity(current_stock["near_expiry"], item.unit))

    alert_col, action_col = st.columns([1.2, 1])
    with alert_col:
        st.subheader("Priority alerts")
        if current_stock["expired"] > 0:
            st.error(f"Expired stock excluded from available supply: {quantity(current_stock['expired'], item.unit, 1)}.")
        if no_order_summary["first_shortage_date"] is not None:
            st.warning(
                f"Selected scenario projects unmet demand starting {no_order_summary['first_shortage_date']:%d %b}: "
                f"{quantity(float(no_order_summary['shortage']), item.unit, 1)} over the {horizon_days}-day horizon."
            )
        else:
            st.success(f"No unmet demand is projected in the selected {horizon_days}-day scenario.")
        if current_stock["near_expiry"] > 0:
            st.warning(f"Prioritize review of {quantity(current_stock['near_expiry'], item.unit, 1)} due to expire within 30 days.")
        if bool(order_plan["needs_order"]):
            st.info(f"Planning rule indicates a human-reviewed replenishment of {quantity(float(order_plan['order_qty']), item.unit, 1)}.")
    with action_col:
        st.subheader("Recommended next action")
        if float(order_plan["order_qty"]) > 1e-6:
            st.markdown(f"### Review {quantity(float(order_plan['order_qty']), item.unit, 1)} for replenishment")
            st.write(f"Illustrative arrival: **{arrival_date:%d %b %Y}** · effective lead time {effective_lead_days} days.")
            if bool(proposed_summary["delivery_after_horizon"]):
                st.warning("The order arrives after this planning horizon, so it does not improve this projection.")
            elif order_improvement > 1e-6:
                st.success(f"Simulation demonstrates {quantity(order_improvement, item.unit, 1)} less unmet demand with the order under these assumptions.")
            else:
                st.info("The simulation does not demonstrate a shortage reduction with this proposed order.")
        else:
            st.markdown("### No order indicated by the current rule")
            st.write("Usable stock and scenario demand do not produce a positive order-up-to quantity in this illustrative calculation.")
        st.caption("Recommendation is a planning aid for review. No procurement or transport action is created.")

    st.subheader("Projected usable stock")
    st.plotly_chart(stock_chart(baseline_run, no_order_run, proposed_run, item.unit, arrival_date), width="stretch", config={"displayModeBar": False, "responsive": True})
    st.caption("Selected scenario demand and proposed-order cases use the same starting batches, forecast horizon, FEFO allocation, and expiry rules. The baseline is the unadjusted demand reference.")


elif page == "Demand Forecast":
    st.markdown('<div class="section-kicker">Demand outlook</div>', unsafe_allow_html=True)
    forecast_days = st.select_slider("Forecast view", options=[7, 14, 30, 60, 90], value=30, format_func=lambda value: f"{value} days")
    forecast_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    base = baseline_90[:forecast_days]
    adjusted = apply_demand_scenario(
        base,
        float(scenario["demand_multiplier"]),
        int(scenario["interruption_days"]),
    )
    f1, f2, f3 = st.columns(3)
    f1.metric("Baseline demand", quantity(float(base.sum()), item.unit, 1))
    f2.metric("Scenario-adjusted demand", quantity(float(adjusted.sum()), item.unit, 1))
    f3.metric("Scenario uplift", quantity(float(adjusted.sum() - base.sum()), item.unit, 1))
    st.plotly_chart(demand_chart(forecast_dates, base, adjusted, item.unit), width="stretch", config={"displayModeBar": False, "responsive": True})
    st.caption(
        f"Scenario overlay uses {scenario['demand_multiplier']:.2f}× demand and applies an additional 1.15× pressure for the first "
        f"{scenario['interruption_days']} day(s). These are illustrative controls configured in Scenario Simulator."
    )
    st.subheader("Chronological holdout evaluation")
    metrics = evaluate_holdout(values, pd.Series(dates), model, int(seed), holdout=28)
    st.caption("The most recent 28 synthetic observations are held out; training uses earlier dates only. These scores are not real-world performance evidence.")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE", f"{metrics['MAE']:.2f} {item.unit}/day")
    e2.metric("RMSE", f"{metrics['RMSE']:.2f} {item.unit}/day")
    e3.metric("WAPE", f"{metrics['WAPE']:.1f}%")
    st.download_button(
        "Download forecast CSV",
        pd.DataFrame({"date": forecast_dates, "baseline_forecast": base, "scenario_adjusted_demand": adjusted, "item": selected_item, "unit": item.unit, "synthetic": True}).to_csv(index=False).encode("utf-8"),
        "anna_demand_forecast_synthetic.csv", "text/csv",
    )


elif page == "Inventory & FEFO":
    st.markdown('<div class="section-kicker">Stock condition</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Usable", quantity(current_stock["usable"], item.unit, 1))
    s2.metric("Expired · unavailable", quantity(current_stock["expired"], item.unit, 1))
    s3.metric("Near expiry · 30 days", quantity(current_stock["near_expiry"], item.unit, 1))
    s4.metric("Selected scenario waste", quantity(float(proposed_summary["expired_waste"]), item.unit, 1))
    st.subheader("Batch register")
    batch_view = batches.copy()
    batch_view["days_to_expiry"] = (batch_view.expiry_date - as_of).dt.days
    batch_view["status"] = np.select(
        [batch_view.days_to_expiry < 0, batch_view.days_to_expiry <= 30],
        ["Expired · unavailable", "Near expiry"], default="Usable",
    )
    batch_view = batch_view.rename(columns={"batch_id": "Batch", "quantity": f"Quantity ({item.unit})", "received_date": "Received", "expiry_date": "Expiry", "days_to_expiry": "Days to expiry", "status": "Status"})
    st.dataframe(batch_view[["Batch", f"Quantity ({item.unit})", "Received", "Expiry", "Days to expiry", "Status"]], hide_index=True, width="stretch")

    st.subheader("Seven-day FEFO pick preview")
    requested = float(baseline_90[:7].sum())
    picked, served, unmet = allocate_fefo(batches, requested, as_of)
    picked["available"] = batches.quantity.to_numpy()
    picked["planned_pick"] = picked.allocated
    picked["remaining"] = picked.quantity
    picked_view = picked.rename(columns={"batch_id": "Batch", "expiry_date": "Expiry", "available": f"Available ({item.unit})", "planned_pick": f"FEFO pick ({item.unit})", "remaining": f"Remaining ({item.unit})"})
    st.dataframe(picked_view[["Batch", "Expiry", f"Available ({item.unit})", f"FEFO pick ({item.unit})", f"Remaining ({item.unit})"]], hide_index=True, width="stretch")
    if unmet > 1e-6:
        st.warning(f"Usable batches serve {quantity(served, item.unit, 1)} of the {quantity(requested, item.unit, 1)} preview demand; {quantity(unmet, item.unit, 1)} remains unmet.")
    st.caption("FEFO means First Expired, First Out. Expired batches are shown for visibility but excluded from pick allocation and usable stock.")

    st.subheader("Synthetic depot snapshot")
    depot_rows = []
    for spec in ITEMS:
        item_batches = batches_all.loc[batches_all.item == spec.name]
        item_summary = inventory_summary(item_batches, as_of)
        depot_rows.append({"Ration item": spec.name, "Usable stock": quantity(item_summary["usable"], spec.unit, 1), "Expired": quantity(item_summary["expired"], spec.unit, 1), "Near expiry": quantity(item_summary["near_expiry"], spec.unit, 1)})
    st.dataframe(pd.DataFrame(depot_rows), hide_index=True, width="stretch")


elif page == "Scenario Simulator":
    st.markdown('<div class="section-kicker">Advanced controls · illustrative only</div>', unsafe_allow_html=True)
    st.write("Choose assumptions, then run matched no-order and proposed-order policies. The only difference between those two runs is whether the calculated order enters the pipeline.")
    with st.form("scenario_form", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            horizon_choice = st.select_slider("Planning horizon", options=[7, 14, 30], value=int(scenario["horizon_days"]), format_func=lambda value: f"{value} days")
            nominal_lead = st.slider("Nominal transport lead time", 1, 21, int(scenario["lead_time_days"]), format="%d days")
        with c2:
            delay_choice = st.slider("Additional arrival delay", 0, 30, int(scenario["additional_delay_days"]), format="%d days")
            demand_choice = st.slider("Demand multiplier", 0.75, 1.75, float(scenario["demand_multiplier"]), 0.05, format="%.2f×")
        with c3:
            interruption_choice = st.slider("Interruption pressure duration", 0, 30, int(scenario["interruption_days"]), format="%d days")
            st.caption("The interruption proxy adds 15% demand pressure during the selected initial days; it does not model a real transport or supply event.")
        submitted = st.form_submit_button("Run matched scenario", type="primary", width="stretch")
    if submitted:
        st.session_state.active_scenario = {
            "lead_time_days": nominal_lead,
            "additional_delay_days": delay_choice,
            "demand_multiplier": demand_choice,
            "interruption_days": interruption_choice,
            "horizon_days": horizon_choice,
        }
        st.rerun()

    st.subheader("Scenario being compared")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Adjusted daily demand", f"{float(scenario_forecast.mean()):,.1f} {item.unit}/day")
    m2.metric("Planning horizon", f"{horizon_days} days")
    m3.metric("Order arrival", f"{arrival_date:%d %b %Y}" if float(order_plan["order_qty"]) > 0 else "No order")
    m4.metric("Proposed quantity", quantity(float(order_plan["order_qty"]), item.unit, 1))
    if bool(proposed_summary["delivery_after_horizon"]):
        st.warning(f"Order arrives on {arrival_date:%d %b %Y}, after this {horizon_days}-day planning horizon. It is not counted as received, and it cannot reduce this horizon's shortage.")
    if no_order_summary["first_shortage_date"] is not None:
        st.error(f"No-order scenario first runs short on {no_order_summary['first_shortage_date']:%d %b %Y}; total unmet demand is {quantity(float(no_order_summary['shortage']), item.unit, 1)}.")
    if order_improvement > 1e-6:
        st.success(f"Matched simulation shows {quantity(order_improvement, item.unit, 1)} less unmet demand after the proposed order arrives.")
    else:
        st.info("The proposed order does not reduce simulated shortage within this horizon. No shortage-reduction claim is made.")

    st.subheader("Policy comparison")
    policy_rows = []
    for name, metrics in [
        ("Baseline · unadjusted forecast, no order", baseline_summary),
        ("Selected scenario · no order", no_order_summary),
        ("Selected scenario · proposed order", proposed_summary),
    ]:
        policy_rows.append({
            "Policy": name,
            "Demand": quantity(float(metrics["total_demand"]), item.unit, 1),
            "Unmet demand": quantity(float(metrics["shortage"]), item.unit, 1),
            "Fill rate": f"{float(metrics['fill_rate_pct']):.1f}%",
            "Expired / waste": quantity(float(metrics["expired_waste"]), item.unit, 1),
            "Ending usable stock": quantity(float(metrics["ending_stock"]), item.unit, 1),
            "Order received": "Yes" if metrics["order_received"] else "No",
        })
    st.dataframe(pd.DataFrame(policy_rows), hide_index=True, width="stretch")
    st.caption("All policies start from the same batches and use the same FEFO and expiry rules. The matched no-order and proposed-order runs use identical scenario-adjusted demand, lead time, delay, and horizon; only the proposed run adds the calculated order.")
    st.plotly_chart(stock_chart(baseline_run, no_order_run, proposed_run, item.unit, arrival_date), width="stretch", config={"displayModeBar": False, "responsive": True})
    with st.expander("Daily simulation detail"):
        detail = proposed_run.copy()
        detail.attrs = {}
        st.dataframe(detail.rename(columns={"date": "Date", "demand": f"Demand ({item.unit})", "served": f"Served ({item.unit})", "shortage": f"Shortage ({item.unit})", "incoming": f"Incoming ({item.unit})", "expired_waste": f"Waste ({item.unit})", "closing_stock": f"Closing stock ({item.unit})", "usable_closing_stock": f"Usable closing stock ({item.unit})"}), hide_index=True, width="stretch")


elif page == "Recommendations":
    st.markdown('<div class="section-kicker">Explainable planning rule</div>', unsafe_allow_html=True)
    rec_col, basis_col = st.columns([1, 1.25])
    with rec_col:
        st.subheader("Suggested action")
        if float(order_plan["order_qty"]) > 1e-6:
            st.markdown(f"## Review {quantity(float(order_plan['order_qty']), item.unit, 1)}")
            st.write(f"Target arrival: **{arrival_date:%A, %d %B %Y}**")
            st.write(f"Effective lead time: **{effective_lead_days} days** (nominal {scenario['lead_time_days']} + additional delay {scenario['additional_delay_days']}).")
            if bool(proposed_summary["delivery_after_horizon"]):
                st.warning("This arrival falls after the selected planning horizon. The proposal cannot prevent shortages in the current projection.")
            if order_improvement > 1e-6:
                st.success(f"Simulated shortage reduction: {quantity(order_improvement, item.unit, 1)} under the selected assumptions.")
            else:
                st.info("The matched simulation did not show lower unmet demand with this order.")
        else:
            st.markdown("## No order indicated")
            st.write("The simplified order-up-to quantity is zero for the selected assumptions.")
        st.warning("Human review required. This synthetic calculation does not create or transmit a procurement request.")
    with basis_col:
        st.subheader("Calculation basis")
        st.markdown(
            "**Reorder point** = forecast demand during effective lead time + safety stock  \n"
            "**Safety stock** = 1.28 × historical daily-demand standard deviation × √(effective lead time)  \n"
            "**Target stock** = forecast demand over max(lead time, planning horizon) + safety stock  \n"
            "**Order quantity** = max(0, target stock − usable stock − confirmed incoming stock)"
        )
        details = [
            ("Scenario demand over planning horizon", quantity(float(order_plan["horizon_demand"]), item.unit, 1)),
            ("Demand through effective lead time", quantity(float(order_plan["lead_demand"]), item.unit, 1)),
            ("Historical daily-demand standard deviation", f"{history_std:.2f} {item.unit}/day"),
            ("Illustrative safety stock (z = 1.28)", quantity(float(order_plan["safety_stock"]), item.unit, 1)),
            ("Reorder point", quantity(float(order_plan["reorder_point"]), item.unit, 1)),
            (f"Target coverage ({order_plan['coverage_days']} days)", quantity(float(order_plan["target_stock"]), item.unit, 1)),
            ("Usable stock", quantity(float(order_plan["usable_stock"]), item.unit, 1)),
            ("Confirmed incoming stock assumed", quantity(float(order_plan["confirmed_incoming"]), item.unit, 1)),
        ]
        st.dataframe(pd.DataFrame(details, columns=["Input", "Calculated value"]), hide_index=True, width="stretch")
        st.caption("Safety stock uses recent synthetic daily-demand variability, not forecast residuals. z = 1.28 is an illustrative normal-demand service factor. No confirmed inbound shipments are present in this prototype.")

    st.subheader("Why this recommendation appears")
    why = [
        f"Usable stock is {quantity(float(order_plan['usable_stock']), item.unit, 1)}; expired stock is excluded.",
        f"The selected scenario forecasts {quantity(float(order_plan['horizon_demand']), item.unit, 1)} over {horizon_days} days.",
        f"The effective arrival lead time is {effective_lead_days} days, giving a reorder point of {quantity(float(order_plan['reorder_point']), item.unit, 1)}.",
        f"Order sizing targets {int(order_plan['coverage_days'])} days of forecast coverage plus safety stock.",
    ]
    for explanation in why:
        st.markdown(f"- {explanation}")
    st.caption("Expiry waste, fill rate, and unmet demand are recalculated in the policy simulation. The recommendation is not evidence of operational savings or real-world forecast performance.")


st.markdown("---")
st.caption("ANNA Prototype 0 · Generated data only · All shelf life, demand variability, lead times, and scenario inputs are illustrative.")
