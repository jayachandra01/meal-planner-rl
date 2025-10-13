"""
Streamlit application for the ForkCast vegetarian meal planner.

This web interface allows users to train a reinforcement learning agent to
generate customised weekly meal plans.  Users can specify their daily
calorie target, weekly budget, number of meals per day and training
episodes.  After training the agent, the app displays a seven‑day meal plan
with four meals per day by default, along with a summary of total calories,
protein, fibre and cost.

The interface adopts a dark theme with red accents to evoke an appetising
yet functional aesthetic.  Colours and typography have been chosen to
provide sufficient contrast while maintaining visual appeal.  The name
“ForkCast” is prominently featured to reinforce the branding of the
application.
"""

import random
from typing import List

import pandas as pd
import streamlit as st

from meal_planner_rl import MealPlanEnv, QLearningAgent, MEALS


def train_agent(
    daily_calories: int,
    weekly_budget: float,
    episodes: int,
    meals_per_day: int,
) -> tuple[QLearningAgent, MealPlanEnv]:
    """Train a Q‑learning agent with the specified parameters.

    Returns both the trained agent and the environment used for training.
    """
    env = MealPlanEnv(
        MEALS,
        weekly_calorie_target=daily_calories * 7,
        weekly_cost_budget=weekly_budget,
        meals_per_day=meals_per_day,
    )
    agent = QLearningAgent(n_actions=env.n_actions)
    for _ in range(episodes):
        state = env.reset()
        for _ in range(env.max_days):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()
    return agent, env


def generate_plan(env: MealPlanEnv, agent: QLearningAgent) -> List[int]:
    """Generate a plan using the greedy policy (no exploration)."""
    state = env.reset()
    plan: List[int] = []
    for _ in range(env.max_days):
        qs = agent.q[state]
        max_val = max(qs)
        # choose the first action with maximal Q to make the plan deterministic
        action = next(i for i, v in enumerate(qs) if v == max_val)
        plan.append(action)
        state, _, done = env.step(action)
        if done:
            break
    return plan


def build_plan_dataframe(plan: List[int], meals_per_day: int) -> pd.DataFrame:
    """Convert a flat list of meal indices into a DataFrame grouped by day.

    The DataFrame has one row per day and one column per meal (Meal 1,
    Meal 2, …).  Cell values are the meal names from the catalogue.
    """
    num_days = len(plan) // meals_per_day
    rows: list[dict[str, str]] = []
    for day in range(num_days):
        start = day * meals_per_day
        end = start + meals_per_day
        row: dict[str, str] = {}
        for idx_in_day, action in enumerate(plan[start:end], start=1):
            meal = MEALS[action]
            row[f"Meal {idx_in_day}"] = meal["name"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.index = [f"Day {i + 1}" for i in range(num_days)]
    return df


def compute_summary(env: MealPlanEnv) -> dict:
    """Return a dictionary of summary statistics for the current episode."""
    return {
        "Total calories (kcal)": f"{env.cum_calories:.0f}",
        "Total protein (g)": f"{env.cum_protein}",
        "Total fibre (g)": f"{env.cum_fibre}",
        "Total cost (₹)": f"{env.cum_cost:.2f}",
    }


def apply_custom_styles() -> None:
    """Inject custom CSS to apply a dark theme with red accents."""
    st.markdown(
        """
        <style>
        /* Base background and text colours */
        body, .stApp {
            background-color: #0d0d0d;
            color: #f5f5f5;
        }
        /* Override default Streamlit colors for primary and secondary elements */
        .css-1cpxqw2 {  /* Title */
            color: #e63946 !important;
        }
        .css-18e3th9, .css-1d391kg {  /* sidebar and main container background */
            background-color: #121212 !important;
        }
        .css-5rimss {  /* card backgrounds */
            background-color: #1e1e1e !important;
            border-radius: 8px;
            padding: 12px;
        }
        .stButton>button {
            background-color: #e63946 !important;
            color: white !important;
            border: None;
            border-radius: 4px;
        }
        .stButton>button:hover {
            background-color: #b42b34 !important;
        }
        /* Table styling */
        .dataframe table {
            background-color: #1e1e1e;
            color: #f5f5f5;
        }
        .dataframe th {
            background-color: #e63946;
            color: white;
        }
        .dataframe td {
            background-color: #222222;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # Configure page
    st.set_page_config(
        page_title="ForkCast – Vegetarian Meal Planner",
        page_icon="🍴",
        layout="wide",
    )
    apply_custom_styles()

    # Sidebar inputs
    st.sidebar.title("ForkCast Settings")
    daily_cal = st.sidebar.number_input(
        "Daily calories (kcal)", min_value=800, max_value=4000, value=2000, step=50
    )
    weekly_budget = st.sidebar.number_input(
        "Weekly budget (₹)", min_value=10.0, max_value=100.0, value=30.0, step=1.0, format="%.2f"
    )
    meals_per_day = st.sidebar.slider(
        "Meals per day", min_value=1, max_value=6, value=4, step=1
    )
    episodes = st.sidebar.slider(
        "Training episodes", min_value=500, max_value=5000, value=2000, step=100
    )

    if "trained_agent" not in st.session_state:
        st.session_state.trained_agent = None
    if "trained_env" not in st.session_state:
        st.session_state.trained_env = None
    if "plan" not in st.session_state:
        st.session_state.plan = None

    st.markdown(
        "<h1 style='color:#e63946;margin-bottom:0;'>ForkCast</h1>"
        "<h3 style='margin-top:0;'>Vegetarian Meal Planner</h3>",
        unsafe_allow_html=True,
    )

    st.write(
        "Plan your meals for the week using reinforcement learning. Adjust your targets "
        "in the sidebar, train the agent, and view your customised weekly plan."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Train & Generate Plan"):
            with st.spinner("Training ForkCast agent…"):
                agent, env = train_agent(int(daily_cal), float(weekly_budget), int(episodes), int(meals_per_day))
                st.session_state.trained_agent = agent
                st.session_state.trained_env = env
                st.session_state.plan = generate_plan(env, agent)
            st.success("ForkCast agent trained and plan generated!")

    # Display plan if available
    if st.session_state.plan is not None:
        plan_indices = st.session_state.plan
        env = st.session_state.trained_env
        # Build DataFrame for display
        df_plan = build_plan_dataframe(plan_indices, env.meals_per_day)
        st.subheader("Your 7‑day meal plan")
        st.dataframe(df_plan, use_container_width=True, height=300)
        # Compute and show summary metrics
        summary = compute_summary(env)
        st.subheader("Summary")
        sum_cols = st.columns(len(summary))
        for (label, value), col in zip(summary.items(), sum_cols):
            col.metric(label, value)


if __name__ == "__main__":
    main()
