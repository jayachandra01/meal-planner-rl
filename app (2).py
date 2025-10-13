"""
Command‑line interface for the Healthy Meal Planner RL project.

This script trains a reinforcement learning agent to plan a weekly menu and
provides a simple interactive interface for users to generate meal plans under
their own calorie and budget constraints.  By default it uses the meal
catalogue defined in ``meal_planner_rl.py`` and trains a tabular Q‑learning
agent on a specified number of episodes.  Users can adjust the daily calorie
target, weekly budget and the number of training episodes via command line
arguments.

Example usage:

    python app.py --calories 2200 --budget 35 --episodes 3000

This trains an agent with a daily calorie target of 2200 kcal, a weekly
budget of ₹35 and 3 000 training episodes.  After training, the script
prints a recommended seven‑day meal plan along with nutrition and cost
summaries.
"""

import argparse
from meal_planner_rl import MealPlanEnv, QLearningAgent, MEALS


def build_environment(daily_calories: int, weekly_budget: float, meals_per_day: int) -> MealPlanEnv:
    """
    Construct an environment with custom calorie and budget targets.

    ``ForkCast`` plans a full week of meals.  You can specify how many meals
    you intend to eat each day (e.g. 1 for a single large meal or 4 for
    breakfast, lunch, snacks and dinner).  The environment converts the
    daily calorie target into a weekly target and passes the number of
    meals per day to the simulator.

    Args:
        daily_calories: Target calories per day.
        weekly_budget: Maximum cost allowed per week (₹).
        meals_per_day: Number of meals consumed per day.

    Returns:
        MealPlanEnv instance configured with the requested parameters.
    """
    # weekly target is daily target times number of days
    weekly_calories = daily_calories * 7
    return MealPlanEnv(
        MEALS,
        weekly_calorie_target=weekly_calories,
        weekly_cost_budget=weekly_budget,
        meals_per_day=meals_per_day,
    )


def train_q_agent(env: MealPlanEnv, episodes: int) -> QLearningAgent:
    """Train a Q‑learning agent on the given environment.

    Args:
        env: Instance of MealPlanEnv.
        episodes: Number of training episodes.

    Returns:
        A trained QLearningAgent.
    """
    agent = QLearningAgent(n_actions=env.n_actions)
    for episode in range(episodes):
        state = env.reset()
        for _ in range(env.max_days):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()
    return agent


def generate_weekly_plan(env: MealPlanEnv, agent: QLearningAgent) -> list[int]:
    """
    Generate a weekly meal plan using the learned policy.

    The agent chooses the highest‑valued action at each state without
    exploration.  The returned list contains one action index per environment
    step (``env.max_days`` elements).  To convert these indices into
    per‑day menus, slice the list into ``env.meals_per_day`` chunks.

    Args:
        env: Trained MealPlanEnv environment.  Note that cumulative sums will
            update as we generate the plan.
        agent: Trained QLearningAgent.

    Returns:
        A list of action indices corresponding to chosen meals for each step.
    """
    state = env.reset()
    plan: list[int] = []
    for _ in range(env.max_days):
        qs = agent.q[state]
        max_val = max(qs)
        # break ties deterministically by choosing first index
        action = next(i for i, v in enumerate(qs) if v == max_val)
        plan.append(action)
        state, _, done = env.step(action)
        if done:
            break
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthy Meal Planner RL")
    parser.add_argument(
        "--calories",
        type=int,
        default=2000,
        help="Daily calorie target (kcal)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=30.0,
        help="Weekly cost budget (₹)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1500,
        help="Number of training episodes for the Q‑learning agent",
    )

    parser.add_argument(
        "--meals-per-day",
        type=int,
        default=4,
        help="Number of meals consumed per day (e.g. 4 for breakfast, lunch, snacks and dinner)",
    )

    args = parser.parse_args()

    print("\nForkCast – Vegetarian Meal Planner (RL)")
    print("=======================================\n")
    print(
        f"Training agent with daily calories = {args.calories} kcal, weekly budget = ₹{args.budget}, "
        f"meals per day = {args.meals_per_day}, episodes = {args.episodes}...\n"
    )

    env = build_environment(args.calories, args.budget, args.meals_per_day)
    agent = train_q_agent(env, args.episodes)

    plan = generate_weekly_plan(env, agent)

    # Summarise the plan
    total_cost = env.cum_cost
    total_calories = env.cum_calories
    total_protein = env.cum_protein
    total_fibre = env.cum_fibre

    print("Recommended weekly meal plan:\n")
    # Group actions into days.  The environment runs for 7 × meals_per_day steps
    # regardless of the value of ``meals_per_day``, so computing the number
    # of days from ``env.max_days`` makes this code future‑proof.
    num_days = env.max_days // env.meals_per_day
    for day in range(num_days):
        start = day * env.meals_per_day
        end = start + env.meals_per_day
        print(f"Day {day + 1}:")
        for idx_in_day, action in enumerate(plan[start:end], start=1):
            meal = MEALS[action]
            print(
                f"  Meal {idx_in_day}: {meal['name']}\n"
                f"    Calories: {meal['calories']} kcal\n"
                f"    Protein:  {meal['protein']} g\n"
                f"    Fibre:    {meal['fibre']} g\n"
                f"    Cost:     ₹{meal['cost']}\n"
            )
        print()

    print("Summary:\n")
    print(f"Total calories consumed: {total_calories:.0f} kcal")
    print(f"Total protein consumed:  {total_protein} g")
    print(f"Total fibre consumed:    {total_fibre} g")
    print(f"Total cost:            ₹{total_cost:.2f}\n")
    print("Thank you for using ForkCast!\n")


if __name__ == "__main__":
    main()