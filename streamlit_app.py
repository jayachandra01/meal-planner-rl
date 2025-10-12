
# -*- coding: utf-8 -*-
import streamlit as st
from collections import Counter, deque
import random

# Local imports
from meal_planner_rl import MealPlanEnv, QLearningAgent, MEALS

st.set_page_config(page_title="Healthy Meal Planner RL - Vegetarian", layout="wide")
st.title("Healthy Meal Planner RL - Vegetarian (Interactive)")
st.caption("Plan a 7-day vegetarian menu balancing nutrition, budget, calories and variety. "
           "Use AUTO for a one-click plan or INTERACTIVE to pick meals and give feedback.")

# --------------------------- Sidebar Controls ---------------------------
with st.sidebar:
    st.header("Settings")
    daily_cal = st.number_input("Daily calories", 1200, 4000, 2000, 50)
    budget = st.number_input("Weekly budget (Rs)", 10.0, 200.0, 30.0, 1.0)
    episodes = st.number_input("Training episodes", 100, 10000, 2000, 100)
    div_w = st.slider("Diversity weight", 0.0, 1.0, 0.2, 0.05)
    recent_window = st.slider("No-repeat window (days)", 0, 6, 2, 1)
    max_repeats = st.slider("Max repeats per week", 1, 7, 2, 1)
    feedback_weight = st.slider("Feedback weight (interactive)", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    train_btn = st.button("Train / Retrain Agent", type="primary")
    reset_btn = st.button("Reset Session (clear agent/env)")

def info_text_from_meal(m):
    return "Calories: " + str(m["calories"]) + " kcal - Protein: " + str(m["protein"]) + " g - " +            "Fibre: " + str(m["fibre"]) + " g - Cost: Rs " + str(m["cost"])

def block_text_from_meal(m):
    return "Calories: " + str(m["calories"]) + " kcal  \n" +            "Protein: " + str(m["protein"]) + " g  \n" +            "Fibre: " + str(m["fibre"]) + " g  \n" +            "Cost: Rs " + str(m["cost"])

# Session state init / reset
if "env" not in st.session_state or reset_btn:
    st.session_state.env = None
    st.session_state.agent = None
    st.session_state.trained = False
    st.session_state.day = 0
    st.session_state.recent = deque(maxlen=recent_window)
    st.session_state.counts = Counter()
    st.session_state.state = None
    st.session_state.plan = []
    st.session_state.finished = False

# Train/retrain
if train_btn:
    env = MealPlanEnv(
        MEALS,
        weekly_calorie_target=daily_cal*7,
        weekly_cost_budget=budget,
        diversity_weight=div_w,
        recent_window=recent_window,
        max_repeats=max_repeats,
    )
    agent = QLearningAgent(n_actions=env.n_actions)
    prog = st.progress(0.0, text="Training agent...")
    for i in range(int(episodes)):
        s = env.reset()
        for _ in range(env.max_days):
            a = agent.select_action(s)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns
            if done:
                break
        agent.decay_epsilon()
        if (i+1) % max(1, int(episodes)//100 if int(episodes) >= 100 else 1) == 0:
            prog.progress((i+1)/float(episodes), text="Training agent... " + str(i+1) + "/" + str(episodes))
    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.trained = True
    st.session_state.day = 0
    st.session_state.recent = deque(maxlen=recent_window)
    st.session_state.counts = Counter()
    st.session_state.state = env.reset()
    st.session_state.plan = []
    st.session_state.finished = False
    st.success("Agent trained. You can use AUTO or INTERACTIVE below.")

if not st.session_state.trained:
    st.info("Set your sliders and click Train / Retrain Agent to begin.")
    st.stop()

env = st.session_state.env
agent = st.session_state.agent

tab_auto, tab_inter = st.tabs(["Auto Plan", "Interactive Plan"])

# ------------------------------- AUTO ----------------------------------
with tab_auto:
    st.subheader("Auto Plan (one click)")
    if st.button("Generate Auto Plan"):
        s = env.reset()
        plan_idx = []
        for _ in range(env.max_days):
            qs = agent.q[s]
            mval = max(qs)
            best = [i for i,v in enumerate(qs) if v == mval]
            a = random.choice(best)
            plan_idx.append(a)
            s, _, done = env.step(a)
            if done:
                break

        st.write("Your 7-day vegetarian plan")
        for i, idx in enumerate(plan_idx, 1):
            m = MEALS[idx]
            st.markdown("Day " + str(i) + ": " + m["name"])
            st.markdown(info_text_from_meal(m))

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total calories", str(env.cum_calories) + " kcal")
        c2.metric("Total protein", str(env.cum_protein) + " g")
        c3.metric("Total fibre", str(env.cum_fibre) + " g")
        c4.metric("Total cost", "Rs " + "{:.2f}".format(env.cum_cost))
        st.caption("Tip: increase Diversity weight or lower Max repeats to avoid duplicates.")

# ---------------------------- INTERACTIVE -------------------------------
with tab_inter:
    st.subheader("Interactive Plan (your choices + feedback)")

    if st.session_state.state is None:
        st.session_state.state = env.reset()
        st.session_state.day = 0
        st.session_state.recent = deque(maxlen=recent_window)
        st.session_state.counts = Counter()
        st.session_state.plan = []
        st.session_state.finished = False

    if st.session_state.finished:
        st.success("Week complete!")
        st.write("Your interactive plan")
        for i, idx in enumerate(st.session_state.plan, 1):
            m = MEALS[idx]
            st.markdown("Day " + str(i) + ": " + m["name"])
            st.markdown(info_text_from_meal(m))
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total calories", str(env.cum_calories) + " kcal")
        c2.metric("Total protein", str(env.cum_protein) + " g")
        c3.metric("Total fibre", str(env.cum_fibre) + " g")
        c4.metric("Total cost", "Rs " + "{:.2f}".format(env.cum_cost))
        if st.button("Start New Week"):
            st.session_state.state = env.reset()
            st.session_state.day = 0
            st.session_state.recent = deque(maxlen=recent_window)
            st.session_state.counts = Counter()
            st.session_state.plan = []
            st.session_state.finished = False
        st.stop()

    day = st.session_state.day + 1
    st.write("Day " + str(day) + " — pick your meal")

    qs = agent.q[st.session_state.state]
    ranked = sorted(range(env.n_actions), key=lambda i: qs[i], reverse=True)
    options = [i for i in ranked if st.session_state.counts[i] < max_repeats and i not in st.session_state.recent][:3]
    if not options:
        options = ranked[:3]

    cols = st.columns(3)
    for k, idx in enumerate(options):
        m = MEALS[idx]
        with cols[k]:
            st.markdown(str(k+1) + ") " + m["name"])
            comps = env.reward_components_if(idx)
            st.caption("Q=" + "{:.2f}".format(qs[idx]) + " | R~" + "{:.2f}".format(comps["reward"]) +
                       " (nutr " + "{:.2f}".format(comps["nutrition_score"]) +
                       ", cost " + "{:.2f}".format(comps["cost_penalty"]) +
                       ", cal " + "{:.2f}".format(comps["calorie_penalty"]) +
                       ", div " + "{:.2f}".format(comps["diversity_penalty"]) + ")")
            st.write(block_text_from_meal(m))

    choice = st.radio("Choose your meal", [str(i+1) for i in range(len(options))], horizontal=True, index=0)

    if st.button("Confirm choice"):
        pick = options[int(choice)-1]
        ns, r, done = env.step(pick)
        st.success("You ate: " + MEALS[pick]["name"] + "  •  reward " + "{:.3f}".format(r))
        st.session_state.plan.append(pick)

        fb = st.radio("Rate it", ["Like", "Neutral", "Dislike"], horizontal=True, index=1)
        fb_map = {"Like": 1, "Neutral": 0, "Dislike": -1}
        fbv = fb_map.get(fb, 0)
        agent.update(st.session_state.state, pick, r + feedback_weight * fbv, ns, False)

        st.session_state.state = ns
        st.session_state.counts[pick] += 1
        st.session_state.recent.append(pick)
        st.session_state.day += 1
        if done:
            st.session_state.finished = True
        st.rerun()
