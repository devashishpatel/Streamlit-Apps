import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian FIRE Calculator",
    page_icon="🔥",
    layout="wide"
)

# --- Indian Historical Market Data (1995-2023) ---
# For a real application, this data should be from a reliable, updated source.
# This is sample data for demonstration, using Nifty 50 TRI, Indian G-Sec 10Y, and CPI.
# Returns are nominal annual returns.
HISTORICAL_DATA_CSV = """
Year,Equity_Return,Debt_Return,Inflation
1995,0.063,0.135,0.102
1996,-0.082,0.141,0.090
1997,0.187,0.118,0.072
1998,0.161,0.122,0.132
1999,0.718,0.114,0.047
2000,-0.121,0.109,0.040
2001,-0.161,0.102,0.038
2002,0.038,0.081,0.043
2003,0.852,0.061,0.038
2004,0.134,0.063,0.038
2005,0.418,0.071,0.042
2006,0.461,0.079,0.058
2007,0.572,0.077,0.064
2008,-0.509,0.086,0.083
2009,0.781,0.065,0.109
2010,0.191,0.081,0.120
2011,-0.238,0.083,0.089
2012,0.302,0.082,0.093
2013,0.081,0.089,0.111
2014,0.329,0.103,0.067
2015,0.000,0.078,0.049
2016,0.044,0.138,0.049
2017,0.305,0.046,0.029
2018,0.046,0.079,0.047
2019,0.135,0.108,0.037
2020,0.163,0.097,0.066
2021,0.256,0.036,0.051
2022,0.057,0.073,0.067
2023,0.214,0.072,0.057
"""
historical_df = pd.read_csv(io.StringIO(HISTORICAL_DATA_CSV))
DATA_START_YEAR = historical_df['Year'].min()
DATA_END_YEAR = historical_df['Year'].max()
MAX_TIME_HORIZON = len(historical_df)

# --- VPW (Variable Percentage Withdrawal) Data Table ---
# Source: Bogleheads Wiki VPW method
VPW_DATA = {
    'Equity_Allocation': [100, 90, 80, 75, 70, 60, 50, 40, 30, 20, 10, 0],
    'Withdrawal_Rate': [7.0, 6.7, 6.4, 6.2, 6.1, 5.8, 5.5, 5.2, 4.9, 4.6, 4.3, 4.0]
}
vpw_df = pd.DataFrame(VPW_DATA).set_index('Equity_Allocation')

# --- Helper Functions ---
def format_inr(amount):
    """Formats a number into Indian Crore/Lakh format."""
    if pd.isna(amount) or amount == 0:
        return "₹ 0"
    if abs(amount) >= 1_00_00_000:
        return f"₹ {amount / 1_00_00_000:.2f} Cr"
    else:
        return f"₹ {amount / 1_00_000:.2f} L"

def get_vpw_rate(equity_allocation_pct):
    """Finds the closest VPW withdrawal rate from the table."""
    available_allocations = np.array(vpw_df.index)
    nearest_alloc = available_allocations[np.abs(available_allocations - equity_allocation_pct).argmin()]
    return vpw_df.loc[nearest_alloc, 'Withdrawal_Rate'] / 100.0


@st.cache_data(ttl=3600)
def run_historical_simulation(
    initial_portfolio, time_horizon, equity_allocation_pct, spending_plan,
    # Constant Spending params
    initial_annual_spending,
    # Guardrails params
    guardrails_spending_pct, upper_guardrail_pct, lower_guardrail_pct,
    spending_change_pct
):
    equity_alloc = equity_allocation_pct / 100.0
    debt_alloc = 1.0 - equity_alloc
    num_historical_years = len(historical_df)
    
    # This slice can become empty if time_horizon is too large, which is the intended logic
    simulation_start_years = historical_df['Year'].iloc[:-(time_horizon-1)] if time_horizon > 1 else historical_df['Year']

    results = []
    all_paths = []

    for start_year in simulation_start_years:
        portfolio_value = initial_portfolio
        
        if spending_plan == 'Guardrails':
            annual_spending = initial_portfolio * (guardrails_spending_pct / 100.0)
        else:
            annual_spending = initial_annual_spending

        path = [portfolio_value]
        is_failed = False
        
        start_index = historical_df[historical_df['Year'] == start_year].index[0]

        for year_num in range(time_horizon):
            if is_failed:
                path.append(0)
                continue

            current_historical_year_index = (start_index + year_num)
            # Stop if we run out of historical data
            if current_historical_year_index >= num_historical_years:
                # This case shouldn't be hit with the new start_year slicing, but is a safeguard
                is_failed = True # Or handle as an incomplete simulation
                path.append(path[-1]) # Keep last value
                continue
            
            market_data = historical_df.iloc[current_historical_year_index]

            if spending_plan == 'Variable Percentage Withdrawal (VPW)':
                vpw_rate = get_vpw_rate(equity_allocation_pct)
                withdrawal = portfolio_value * vpw_rate
            else:
                withdrawal = annual_spending

            portfolio_value -= withdrawal
            if portfolio_value <= 0:
                is_failed = True
                path.append(0)
                continue
            
            equity_return = market_data['Equity_Return']
            debt_return = market_data['Debt_Return']
            portfolio_return = (equity_alloc * equity_return) + (debt_alloc * debt_return)
            portfolio_value *= (1 + portfolio_return)

            inflation = market_data['Inflation']
            if spending_plan == 'Constant Spending':
                annual_spending *= (1 + inflation)
            elif spending_plan == 'Guardrails':
                current_wr = (annual_spending / portfolio_value) * 100.0 if portfolio_value > 0 else float('inf')
                if current_wr > upper_guardrail_pct:
                    annual_spending *= (1 - (spending_change_pct / 100.0))
                elif current_wr < lower_guardrail_pct:
                    annual_spending *= (1 + (spending_change_pct / 100.0))
                else:
                    annual_spending *= (1 + inflation)

            path.append(portfolio_value)

        final_value = path[-1]
        results.append({
            'Start Year': start_year,
            'End Portfolio Value': final_value,
            'Status': 'Failed' if is_failed else 'Success',
            'Years Lasted': time_horizon if not is_failed else next((i for i, v in enumerate(path) if v == 0), time_horizon) -1
        })
        all_paths.append(path)

    return pd.DataFrame(results), np.array(all_paths)


# --- UI ---
st.title("🔥 Indian Financial Independence Calculator")
st.markdown("This calculator uses  historical simulation methodology with Indian market data.")

with st.sidebar:
    st.header("1. Your Portfolio")
    initial_portfolio_lakhs = st.number_input("Portfolio Value (in Lakhs ₹)", min_value=1.0, value=100.0, step=10.0)
    initial_portfolio = initial_portfolio_lakhs * 1_00_000

    equity_allocation_pct = st.slider("Equity Allocation (%)", 0, 100, 75)
    st.markdown(f"**Debt Allocation:** {100 - equity_allocation_pct}%")

    st.header("2. Timeframe")
    time_horizon = st.slider("Retirement Duration (Years)", 10, 50, 20)

    st.header("3. Spending Plan")
    spending_plan = st.radio(
        "Select a spending model:",
        ('Constant Spending', 'Guardrails', 'Variable Percentage Withdrawal (VPW)'),
        captions=[
            "Withdraw a fixed, inflation-adjusted amount.",
            "Adjust spending only when crossing thresholds.",
            "Withdraw a variable % of the portfolio."
        ]
    )
    
    initial_annual_spending = 0
    guardrails_spending_pct = 0
    upper_guardrail_pct, lower_guardrail_pct, spending_change_pct = 0, 0, 0

    if spending_plan == 'Constant Spending':
        annual_spending_lakhs = st.number_input("Annual Spending (in Lakhs ₹)", min_value=1.0, value=4.0, step=0.5)
        initial_annual_spending = annual_spending_lakhs * 1_00_000
        wr = (initial_annual_spending / initial_portfolio) * 100
        st.markdown(f"**Initial Withdrawal Rate: `{wr:.2f}%`**")
    
    elif spending_plan == 'Guardrails':
        st.markdown("This model starts with a withdrawal rate and adjusts spending by a fixed % if the rate goes outside the guardrails.")
        guardrails_spending_pct = st.slider("Initial Withdrawal Rate (%)", 1.0, 10.0, 4.0, 0.5)
        lower_guardrail_pct = st.number_input("Lower Guardrail (WR %)", value=3.2, help="If WR drops below this, increase spending.")
        upper_guardrail_pct = st.number_input("Upper Guardrail (WR %)", value=4.8, help="If WR rises above this, cut spending.")
        spending_change_pct = st.slider("Spending Adjustment (%)", 5, 25, 10, help="By what % to increase/decrease spending when a guardrail is hit.")
    
    elif spending_plan == 'Variable Percentage Withdrawal (VPW)':
        vpw_rate_pct = get_vpw_rate(equity_allocation_pct) * 100
        st.info(f"For a **{equity_allocation_pct}%** equity allocation, the VPW model uses a withdrawal rate of **{vpw_rate_pct:.1f}%** of the *current* portfolio value each year.")
        st.markdown("This method automatically adjusts spending based on portfolio performance.")

# --- Main Page Results ---
st.header("Simulation Results")
results_df, paths = run_historical_simulation(
    initial_portfolio, time_horizon, equity_allocation_pct, spending_plan,
    initial_annual_spending,
    guardrails_spending_pct, upper_guardrail_pct, lower_guardrail_pct, spending_change_pct
)

# ----------------- THE FIX IS HERE -----------------
# Check if the results DataFrame is empty. If it is, show an error and stop.
if results_df.empty:
    st.error(
        f"**Simulation Cannot Run!**\n\n"
        f"The selected **Time Horizon ({time_horizon} years)** is too long for the available historical data "
        f"which has **{MAX_TIME_HORIZON} years** (from {DATA_START_YEAR} to {DATA_END_YEAR}).\n\n"
        f"Please choose a time horizon of **{MAX_TIME_HORIZON} years or less**."
    )
else:
    # --- This entire block only runs if the simulation was successful ---
    success_rate = (results_df['Status'] == 'Success').mean() * 100
    successful_runs = results_df[results_df['Status'] == 'Success']
    failed_runs = results_df[results_df['Status'] == 'Failed']

    col1, col2, col3 = st.columns(3)
    col1.metric("Success Rate", f"{success_rate:.1f}%")
    if not successful_runs.empty:
        median_final_value = successful_runs['End Portfolio Value'].median()
        col2.metric("Median Final Portfolio (Success)", format_inr(median_final_value))
    else:
        col2.metric("Median Final Portfolio (Success)", "N/A")

    if not failed_runs.empty:
        worst_case_years = failed_runs['Years Lasted'].min()
        col3.metric("Shortest Duration (Failure)", f"{worst_case_years} years")
    else:
        col3.metric("Shortest Duration", f"{time_horizon} years (No Failures)")

    fig = go.Figure()
    paths_in_lakhs = paths / 1_00_000

    for i, row in failed_runs.iterrows():
        path_index = results_df.index.get_loc(i)
        fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=paths_in_lakhs[path_index], mode='lines', line=dict(color='rgba(239, 83, 80, 0.4)'), showlegend=False))
    
    for i, row in successful_runs.iterrows():
        path_index = results_df.index.get_loc(i)
        fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=paths_in_lakhs[path_index], mode='lines', line=dict(color='rgba(33, 150, 243, 0.4)'), showlegend=False))

    if not successful_runs.empty:
        successful_paths = paths[successful_runs.index] / 1_00_000
        p10 = np.percentile(successful_paths, 10, axis=0)
        p50 = np.percentile(successful_paths, 50, axis=0)
        p90 = np.percentile(successful_paths, 90, axis=0)
        fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=p10, mode='lines', line=dict(color='#ff9800', width=2, dash='dash'), name='10th Percentile'))
        fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=p50, mode='lines', line=dict(color='#000000', width=3), name='Median'))
        fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=p90, mode='lines', line=dict(color='#4caf50', width=2, dash='dash'), name='90th Percentile'))

    fig.update_layout(
        title=f'Portfolio Trajectories Across {len(results_df)} Historical Periods',
        xaxis_title='Year in Retirement', yaxis_title='Portfolio Value (in Lakhs ₹)',
        yaxis_tickprefix='₹', yaxis_ticksuffix='L', legend_title_text='Successful Scenarios'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("Detailed Outcomes by Historical Start Year")
    st.markdown("This table shows the result of your retirement plan if it had started in each of the historical years below.")
    display_df = results_df.copy()
    display_df['End Portfolio Value'] = display_df['End Portfolio Value'].apply(format_inr)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ----------------- END OF THE FIX BLOCK -----------------


with st.expander("⚠️ Click here to understand the Methodology and Assumptions"):
    st.markdown(f"""
    This calculator **does not use random simulations (Monte Carlo)**. Instead, it uses a methodology called **Historical Simulation**.

    #### How it Works:
    1.  We use a dataset of actual market returns for Indian Equities (Nifty 50 TRI), Debt (10Y G-Sec), and Inflation (CPI) from **{DATA_START_YEAR} to {DATA_END_YEAR}**.
    2.  The simulation models your retirement as if it started in every possible year that allows for a full `{time_horizon}`-year period (e.g., starting in 1995, 1996, etc.).
    3.  For each simulation, it replays the sequence of actual market returns and inflation that followed that start year.
    4.  The **Success Rate** is the percentage of these historical starting periods in which your portfolio survived for the entire duration.

    #### Spending Models Explained:
    -   **Constant Spending:** You withdraw your initial amount, adjusted for the *actual historical inflation* of each year.
    -   **Guardrails:** Your spending is adjusted only if your withdrawal rate (annual spending / portfolio value) goes outside a pre-defined range. This adapts to market conditions.
    -   **Variable Percentage Withdrawal (VPW):** You withdraw a percentage of your *current portfolio balance* each year. This method is highly adaptive and significantly reduces the risk of ruin.

    **Disclaimer:** This is an educational tool based on historical data. Past performance is not a guarantee of future results. The model makes simplifying assumptions (e.g., annual rebalancing, no taxes). Consult a financial advisor for personalized advice.

    """)
