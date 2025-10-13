"""
meal_planner_rl.py
-------------------

This module implements a simple reinforcement learning environment and agent for
constructing a weekly meal plan.  The environment models a seven‑day planning
task where an agent must select one meal per day from a small catalogue of
healthy and affordable dishes.  Each meal has an associated calorie count,
protein and fibre content and a cost.  The agent receives a reward at each
step based on how nutritious and affordable the chosen meal is and how well
the cumulative calories align with a weekly target.  The goal of the agent is
to maximise total reward across the week, thereby balancing nutrition and
budget.  A tabular Q‑learning algorithm is used to learn a policy from
experience.

The code defines three main parts:

1. A data structure describing a small meal catalogue.  Each entry in the
   catalogue includes the meal name, calories, protein, fibre and cost.
2. A ``MealPlanEnv`` class implementing the environment.  The state consists
   of the current day and coarse categories for the remaining calorie and
   cost budgets.  The environment exposes ``reset`` and ``step`` methods
   consistent with the OpenAI Gym API.
3. A ``QLearningAgent`` class implementing tabular Q‑learning with an
   epsilon‑greedy exploration strategy.  The agent maintains a Q‑table
   keyed by state tuples and updates its estimates on every transition.

At the bottom of the file is a ``train_and_evaluate`` function that trains
the agent for a fixed number of episodes and then evaluates its performance
against a random baseline.  During evaluation the script reports the average
weekly reward, cost and nutrition scores as well as a simple measure of
diversity.  This function is called when the module is executed as a
standalone script.
"""

import random
from collections import defaultdict, Counter
import math


# -----------------------------------------------------------------------------
# Meal catalogue
# -----------------------------------------------------------------------------

# A small catalogue of meals with approximate nutrition and cost values.
# Vegetarian meal catalogue
MEALS = [
    {
        "name": "Paneer tikka with salad",
        "calories": 450,
        "protein": 25,
        "fibre": 2,
        "cost": 4.0,
    },
    {
        "name": "Vegetable biryani",
        "calories": 500,
        "protein": 12,
        "fibre": 5,
        "cost": 3.5,
    },
    {
        "name": "Chole masala with rice",
        "calories": 600,
        "protein": 20,
        "fibre": 10,
        "cost": 3.5,
    },
    {
        "name": "Dal roti",
        "calories": 400,
        "protein": 15,
        "fibre": 5,
        "cost": 2.0,
    },
    {
        "name": "Hummus and vegetable wrap",
        "calories": 450,
        "protein": 12,
        "fibre": 8,
        "cost": 3.5,
    },
    {
        "name": "Spinach and chickpea curry",
        "calories": 350,
        "protein": 17,
        "fibre": 7,
        "cost": 3.0,
    },
    {
        "name": "Greek salad with feta",
        "calories": 350,
        "protein": 8,
        "fibre": 4,
        "cost": 3.0,
    },
    {
        "name": "Oatmeal with fruits and nuts",
        "calories": 300,
        "protein": 7,
        "fibre": 6,
        "cost": 2.5,
    },
    {
        "name": "Rajma chawal",
        "calories": 500,
        "protein": 18,
        "fibre": 9,
        "cost": 2.5,
    },
    {
        "name": "Tofu stir‑fry with brown rice",
        "calories": 450,
        "protein": 20,
        "fibre": 9,
        "cost": 3.8,
    },
]


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class MealPlanEnv:
    """A weekly meal planning environment for reinforcement learning.

    An episode spans a full week.  Each time step corresponds to one meal.  By
    default the environment assumes a single meal per day (7 steps per week),
    but this can be increased via the ``meals_per_day`` argument to model
    breakfast, lunch, snacks and dinner separately.  The state captures the
    current step index and discretised remaining calorie and cost budgets.  The
    reward encourages nutritious meals while penalising high cost, large
    deviations from the weekly calorie target and repeated meals.  The weekly
    calorie target and cost budget can be configured via the constructor.
    """

    def __init__(
        self,
        meals,
        weekly_calorie_target=14000,
        weekly_cost_budget=30.0,
        daily_protein_target=50,
        daily_fibre_target=30,
        meals_per_day=1,
    ):
        """
        Initialise a weekly meal planning environment.

        Args:
            meals: Sequence of meal dictionaries defining the catalogue.  Each
                dictionary must include ``name``, ``calories``, ``protein``,
                ``fibre`` and ``cost`` keys.
            weekly_calorie_target: Total calories allowed over the week.  This
                is used to compute the per‑step target when more than one meal
                per day is requested.
            weekly_cost_budget: Maximum rupee budget for the week.
            daily_protein_target: Approximate protein target per day (g).  When
                there are multiple meals per day this target is divided evenly
                across the meals and used to scale the nutrition reward.
            daily_fibre_target: Approximate fibre target per day (g).  As with
                protein, this is divided over the meals per day.
            meals_per_day: Number of meals consumed per day.  Use 1 for a
                single large meal or 4 for breakfast/lunch/snacks/dinner.  The
                environment will run for ``7 * meals_per_day`` steps per
                episode.
        """
        self.meals = meals
        self.n_actions = len(meals)
        self.weekly_calorie_target = weekly_calorie_target
        self.weekly_cost_budget = weekly_cost_budget
        self.daily_protein_target = daily_protein_target
        self.daily_fibre_target = daily_fibre_target
        # number of meals per day (e.g. breakfast, lunch, snack, dinner)
        self.meals_per_day = max(1, meals_per_day)
        # total number of steps in an episode = days * meals per day
        self.max_days = 7 * self.meals_per_day
        self.reset()

    def _discretise(self, remaining, total_budget):
        """Map a remaining quantity to a coarse category 0 (low), 1 (medium), 2 (high)."""
        if remaining >= 0.66 * total_budget:
            return 2  # high budget remaining
        elif remaining >= 0.33 * total_budget:
            return 1  # medium
        else:
            return 0  # low

    def reset(self):
        """Reset the environment to the start of a new week."""
        self.day = 0
        self.cum_calories = 0
        self.cum_cost = 0.0
        self.cum_protein = 0
        self.cum_fibre = 0
        state = self._get_state()
        return state

    def _get_state(self):
        """Return a tuple representing the current discretised state."""
        rem_calories = max(0, self.weekly_calorie_target - self.cum_calories)
        rem_cost = max(0.0, self.weekly_cost_budget - self.cum_cost)
        cal_cat = self._discretise(rem_calories, self.weekly_calorie_target)
        cost_cat = self._discretise(rem_cost, self.weekly_cost_budget)
        return (self.day, cal_cat, cost_cat)

    def step(self, action):
        """Take an action (select a meal) and return next_state, reward, done."""
        meal = self.meals[action]
        # Update cumulative sums
        self.cum_calories += meal["calories"]
        self.cum_cost += meal["cost"]
        self.cum_protein += meal["protein"]
        self.cum_fibre += meal["fibre"]

        # Compute reward
        # Nutrition score encourages meals high in protein and fibre relative to targets
        # When multiple meals are consumed per day we divide the daily
        # protein and fibre targets evenly across the meals.  This ensures
        # that each meal is rewarded for contributing an appropriate
        # fraction of the daily target rather than the full amount.  For
        # example, if the daily protein target is 50 g and there are
        # four meals per day then each meal should ideally contain
        # 12.5 g of protein.
        per_meal_protein_target = self.daily_protein_target / self.meals_per_day
        per_meal_fibre_target = self.daily_fibre_target / self.meals_per_day
        protein_score = meal["protein"] / per_meal_protein_target
        fibre_score = meal["fibre"] / per_meal_fibre_target
        # Average the two ratios to obtain the nutrition score.  Values >1
        # indicate a meal exceeding its share of the daily target.
        nutrition_score = (protein_score + fibre_score) / 2.0
        # Cost penalty scaled to budget
        cost_penalty = meal["cost"] / (self.weekly_cost_budget / self.max_days)
        # Calorie penalty encourages staying near average per meal
        target_per_step = self.weekly_calorie_target / self.max_days
        deviation = abs((self.cum_calories / (self.day + 1)) - target_per_step) / target_per_step
        calorie_penalty = deviation
        reward = nutrition_score - 0.2 * cost_penalty - 0.5 * calorie_penalty

        # Advance day
        self.day += 1
        done = self.day >= self.max_days
        next_state = self._get_state()
        return next_state, reward, done

    def evaluate_policy(self, policy_fn, n_episodes=100):
        """Evaluate a policy over several episodes and return average statistics."""
        total_rewards = []
        total_costs = []
        total_nutrition = []
        diversity_scores = []
        for _ in range(n_episodes):
            state = self.reset()
            episode_reward = 0.0
            meals_chosen = []
            for _d in range(self.max_days):
                action = policy_fn(state)
                meals_chosen.append(action)
                state, r, done = self.step(action)
                episode_reward += r
                if done:
                    break
            total_rewards.append(episode_reward)
            total_costs.append(self.cum_cost)
            # Compute nutrition balance score: average of protein and fibre relative to targets
            protein_balance = (self.cum_protein / (self.daily_protein_target * self.max_days))
            fibre_balance = (self.cum_fibre / (self.daily_fibre_target * self.max_days))
            total_nutrition.append((protein_balance + fibre_balance) / 2.0)
            # Diversity index (Shannon entropy) of meals chosen
            counts = Counter(meals_chosen)
            prob = [c / len(meals_chosen) for c in counts.values()]
            entropy = -sum(p * math.log(p) for p in prob)
            diversity_scores.append(entropy)
        return {
            "avg_reward": sum(total_rewards) / n_episodes,
            "avg_cost": sum(total_costs) / n_episodes,
            "avg_nutrition": sum(total_nutrition) / n_episodes,
            "avg_diversity": sum(diversity_scores) / n_episodes,
        }


# -----------------------------------------------------------------------------
# Q‑learning Agent
# -----------------------------------------------------------------------------

class QLearningAgent:
    """Tabular Q‑learning agent with epsilon‑greedy exploration."""

    def __init__(self, n_actions, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        # Nested dictionary: Q[state][action] = value
        self.q = defaultdict(lambda: [0.0] * n_actions)

    def select_action(self, state):
        """Epsilon‑greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            qs = self.q[state]
            max_val = max(qs)
            # Break ties randomly
            best_actions = [i for i, v in enumerate(qs) if v == max_val]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Update the Q‑table for a given transition."""
        current_q = self.q[state][action]
        next_max = max(self.q[next_state]) if not done else 0.0
        target = reward + self.gamma * next_max
        # Q‑learning update
        self.q[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Decay epsilon but do not go below the minimum."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay


# -----------------------------------------------------------------------------
# Training and evaluation routine
# -----------------------------------------------------------------------------

def train_and_evaluate(num_train_episodes=5000, evaluation_episodes=200):
    """Train the Q‑learning agent and evaluate its performance against a random baseline.

    Returns a dictionary containing evaluation statistics for the learned policy
    and the random baseline for comparison.
    """
    env = MealPlanEnv(MEALS)
    agent = QLearningAgent(n_actions=env.n_actions)

    # Training loop
    for episode in range(num_train_episodes):
        state = env.reset()
        for _ in range(env.max_days):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()
    # Define learned policy
    def learned_policy(state):
        qs = agent.q[state]
        max_val = max(qs)
        best_actions = [i for i, v in enumerate(qs) if v == max_val]
        return random.choice(best_actions)

    def random_policy(state):
        return random.randrange(env.n_actions)

    learned_stats = env.evaluate_policy(learned_policy, n_episodes=evaluation_episodes)
    random_stats = env.evaluate_policy(random_policy, n_episodes=evaluation_episodes)
    return {
        "learned": learned_stats,
        "random": random_stats,
        "agent": agent,
        "env": env,
    }


if __name__ == "__main__":
    results = train_and_evaluate(num_train_episodes=2000, evaluation_episodes=100)
    print("Evaluation of learned policy:")
    for k, v in results["learned"].items():
        print(f"{k}: {v:.3f}")
    print("\nEvaluation of random policy:")
    for k, v in results["random"].items():
        print(f"{k}: {v:.3f}")