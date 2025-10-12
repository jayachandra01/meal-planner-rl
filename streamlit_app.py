
import streamlit as st
from collections import Counter, deque
import random

# Local imports
from meal_planner_rl import MealPlanEnv, QLearningAgent, MEALS

st.set_page_config(page_title="Healthy Meal Planner RL — Vegetarian", layout="wide")
st.title("🥗 Healthy Meal Planner RL — Vegetarian (Interactive)")
st.caption("Plan a 7‑day vegetarian menu balancing nutrition, budget, calories and variety. "
           "Use AUTO for a one‑click plan or INTERACTIVE to pick meals and give feedback.")

# --------------------------- Sidebar Controls ---------------------------
with st.sidebar:
    st.header("Settings")
    daily_cal = st.number_input("Daily calories", 1200, 4000, 2000, 50)
    budget = st.number_input("Weekly budget (₹)", 10.0, 200.0, 30.0, 1.0)
    episodes = st.number_input("Training episodes", 100, 10000, 2000, 100)
    div_w = st.slider("Diversity weight", 0.0, 1.0, 0.2, 0.05)
    recent_window = st.slider("No‑repeat window (days)", 0, 6, 2, 1)
    max_repeats = st.slider("Max repeats per week", 1, 7, 2, 1)
    feedback_weight = st.slider("Feedback weight (interactive)", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    train_btn = st.button("🚀 Train / Retrain Agent", type="primary")
    reset_btn = st.button("🔄 Reset Session (clear agent/env)")

if "env" not in st.session_state or reset_btn:
    st.session_state.env = None
    st.session_state.agent = None
    st.session_state.trained = False
    # interactive state
    st.session_state.day = 0
    st.session_state.recent = deque(maxlen=recent_window)
    st.session_state.counts = Counter()
    st.session_state.state = None
    st.session_state.plan = []
    st.session_state.finished = False

# Train/retrain action
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
            if done: break
        agent.decay_epsilon()
        if (i+1) % max(1, episodes//100) == 0:
            prog.progress((i+1)/episodes, text=f"Training agent... {i+1}/{episodes}")
    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.trained = True
    # reset interactive state
    st.session_state.day = 0
    st.session_state.recent = deque(maxlen=recent_window)
    st.session_state.counts = Counter()
    st.session_state.state = env.reset()
    st.session_state.plan = []
    st.session_state.finished = False
    st.success("Agent trained. You can use AUTO or INTERACTIVE below.")

if not st.session_state.trained:
    st.info("Set your sliders and click **Train / Retrain Agent** to begin.")
    st.stop()

env = st.session_state.env
agent = st.session_state.agent

tab_auto, tab_inter = st.tabs(["⚡ Auto Plan", "🧭 Interactive Plan"])

# ------------------------------- AUTO ----------------------------------
with tab_auto:
    st.subheader("Auto Plan (one click)")
    if st.button("Generate Auto Plan"):
        s = env.reset()
        plan_idx = []
        for _ in range(env.max_days):
            qs = agent.q[s]
            m = max(qs)
            best = [i for i,v in enumerate(qs) if v == m]
            a = random.choice(best)
            plan_idx.append(a)
            s, _, done = env.step(a)
            if done: break

        st.write("### Your 7‑day vegetarian plan")
        for i, idx in enumerate(plan_idx, 1):
            m = MEALS[idx]
            st.markdown(f"**Day {i}: {m['name']}**  
"
                        f"Calories: {m['calories']} kcal • Protein: {m['protein']} g • "
                        f"Fibre: {m['fibre']} g • Cost: ₹{m['cost']}")

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total calories", f"{env.cum_calories} kcal")
        c2.metric("Total protein", f"{env.cum_protein} g")
        c3.metric("Total fibre", f"{env.cum_fibre} g")
        c4.metric("Total cost", f"₹{env.cum_cost:.2f}")
        st.caption("Tip: increase Diversity weight or lower Max repeats to avoid duplicates.")

# ---------------------------- INTERACTIVE -------------------------------
with tab_inter:
    st.subheader("Interactive Plan (your choices + feedback)")

    # Initialize interactive state on first load after training
    if st.session_state.state is None:
        st.session_state.state = env.reset()
        st.session_state.day = 0
        st.session_state.recent = deque(maxlen=recent_window)
        st.session_state.counts = Counter()
        st.session_state.plan = []
        st.session_state.finished = False

    if st.session_state.finished:
        st.success("Week complete!")
        st.write("### Your interactive plan")
        for i, idx in enumerate(st.session_state.plan, 1):
            m = MEALS[idx]
            st.markdown(f"**Day {i}: {m['name']}**  
"
                        f"Calories: {m['calories']} kcal • Protein: {m['protein']} g • "
                        f"Fibre: {m['fibre']} g • Cost: ₹{m['cost']}")
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total calories", f"{env.cum_calories} kcal")
        c2.metric("Total protein", f"{env.cum_protein} g")
        c3.metric("Total fibre", f"{env.cum_fibre} g")
        c4.metric("Total cost", f"₹{env.cum_cost:.2f}")
        if st.button("Start New Week"):
            st.session_state.state = env.reset()
            st.session_state.day = 0
            st.session_state.recent = deque(maxlen=recent_window)
            st.session_state.counts = Counter()
            st.session_state.plan = []
            st.session_state.finished = False
        st.stop()

    day = st.session_state.day + 1
    st.write(f"### Day {day} — pick your meal")

    # Rank top options under constraints
    qs = agent.q[st.session_state.state]
    ranked = sorted(range(env.n_actions), key=lambda i: qs[i], reverse=True)
    options = [i for i in ranked if st.session_state.counts[i] < max_repeats and i not in st.session_state.recent][:3]
    if not options:
        options = ranked[:3]

    cols = st.columns(3)
    for k, idx in enumerate(options):
        m = MEALS[idx]
        with cols[k]:
            comps = env.reward_components_if(idx)
            st.markdown(f"**{k+1}) {m['name']}**")
            st.caption(f"Q={qs[idx]:.2f} | R≈{comps['reward']:.2f} "
                       f"(nutr {comps['nutrition_score']:.2f}, "
                       f"cost {comps['cost_penalty']:.2f}, "
                       f"cal {comps['calorie_penalty']:.2f}, "
                       f"div {comps['diversity_penalty']:.2f})")
            st.write(f"Calories: {m['calories']} kcal  
"
                     f"Protein: {m['protein']} g  
"
                     f"Fibre: {m['fibre']} g  
"
                     f"Cost: ₹{m['cost']}")

    choice = st.radio("Choose your meal", [f"{i+1}" for i in range(len(options))], horizontal=True, index=0)

    if st.button("Confirm choice"):
        pick = options[int(choice)-1]
        ns, r, done = env.step(pick)
        st.success(f"You ate: {MEALS[pick]['name']}  •  reward {r:.3f}")
        st.session_state.plan.append(pick)
        # allow feedback
        fb = st.radio("Rate it", ["👍 Like", "😐 Neutral", "👎 Dislike"], horizontal=True, index=1)
        fbv = {"👍 Like": 1, "😐 Neutral": 0, "👎 Dislike": -1}[fb]
        # small on-policy update with feedback
        agent.update(st.session_state.state, pick, r + feedback_weight * fbv, ns, False)
        # advance
        st.session_state.state = ns
        st.session_state.counts[pick] += 1
        st.session_state.recent.append(pick)
        st.session_state.day += 1
        if done:
            st.session_state.finished = True
        st.experimental_rerun()
