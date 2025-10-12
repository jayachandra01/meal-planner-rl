
"""
meal_planner_rl.py (vegetarian + diversity-aware + preview)
-----------------------------------------------------------
RL environment and tabular Q-learning agent for weekly vegetarian
meal planning. Adds a reward preview helper for debugging/UI.
"""
import random
import math
from collections import defaultdict, Counter
from typing import Dict, Tuple

MEALS = [
    {"name": "Paneer tikka with salad",          "calories": 450, "protein": 25, "fibre":  2, "cost": 4.0},
    {"name": "Vegetable biryani",                "calories": 500, "protein": 12, "fibre":  5, "cost": 3.5},
    {"name": "Chole masala with rice",           "calories": 600, "protein": 20, "fibre": 10, "cost": 3.5},
    {"name": "Dal roti",                         "calories": 400, "protein": 15, "fibre":  5, "cost": 2.0},
    {"name": "Hummus and vegetable wrap",        "calories": 450, "protein": 12, "fibre":  8, "cost": 3.5},
    {"name": "Spinach and chickpea curry",       "calories": 350, "protein": 17, "fibre":  7, "cost": 3.0},
    {"name": "Greek salad with feta",            "calories": 350, "protein":  8, "fibre":  4, "cost": 3.0},
    {"name": "Oatmeal with fruits and nuts",     "calories": 300, "protein":  7, "fibre":  6, "cost": 2.5},
    {"name": "Rajma chawal",                     "calories": 500, "protein": 18, "fibre":  9, "cost": 2.5},
    {"name": "Tofu stir-fry with brown rice",    "calories": 450, "protein": 20, "fibre":  9, "cost": 3.8},
]

class MealPlanEnv:
    """Weekly vegetarian meal-planning environment for RL.
    State: (day, calorie_bucket, cost_bucket).
    Action: index of a meal in MEALS.
    Reward: nutrition score - cost penalty - calorie penalty - diversity penalty.
    """
    def __init__(
        self,
        meals,
        weekly_calorie_target: int = 14_000,
        weekly_cost_budget: float = 30.0,
        daily_protein_target: int = 50,
        daily_fibre_target: int = 30,
        max_days: int = 7,
        diversity_weight: float = 0.2,
        recent_window: int = 2,
        max_repeats: int = 2,
    ) -> None:
        self.meals = meals
        self.n_actions = len(meals)
        self.max_days = max_days
        self.weekly_calorie_target = weekly_calorie_target
        self.weekly_cost_budget = weekly_cost_budget
        self.daily_protein_target = daily_protein_target
        self.daily_fibre_target = daily_fibre_target
        self.diversity_weight = diversity_weight
        self.recent_window = max(0, int(recent_window))
        self.max_repeats = max(1, int(max_repeats))
        self.reset()

    def reset(self):
        self.day = 0
        self.cum_calories = 0
        self.cum_cost = 0.0
        self.cum_protein = 0.0
        self.cum_fibre = 0.0
        self.history = []
        self.meal_counts = Counter()
        return self._get_state()

    def _bucket(self, frac: float) -> int:
        if frac <= 0.2: return 0
        if frac <= 0.4: return 1
        if frac <= 0.6: return 2
        if frac <= 0.8: return 3
        return 4

    def _get_state(self) -> Tuple[int, int, int]:
        rem_cals = max(0.0, (self.weekly_calorie_target - self.cum_calories)) / max(1.0, self.weekly_calorie_target)
        rem_cost = max(0.0, (self.weekly_cost_budget - self.cum_cost)) / max(1.0, self.weekly_cost_budget)
        return (self.day, self._bucket(rem_cals), self._bucket(rem_cost))

    # ---- reward components helper (no mutation) ----
    def reward_components_if(self, action: int):
        meal = self.meals[action]
        cum_calories = self.cum_calories + meal["calories"]
        cum_cost = self.cum_cost + meal["cost"]
        cum_protein = self.cum_protein + meal["protein"]
        cum_fibre = self.cum_fibre + meal["fibre"]
        prot_ratio = (cum_protein / (self.daily_protein_target * self.max_days))
        fib_ratio  = (cum_fibre  / (self.daily_fibre_target  * self.max_days))
        nutrition_score = (prot_ratio + fib_ratio) / 2.0
        cost_penalty = max(0.0, cum_cost - self.weekly_cost_budget) / max(1.0, self.weekly_cost_budget)
        avg_cals_so_far = cum_calories / (self.day + 1 if self.day + 1 > 0 else 1)
        calorie_penalty = abs(avg_cals_so_far - (self.weekly_calorie_target / self.max_days)) / 2000.0
        recent_repeat = 1 if action in self.history[-self.recent_window:] else 0
        over_cap = max(0, self.meal_counts[action] + 1 - self.max_repeats)
        diversity_penalty = self.diversity_weight * (recent_repeat + over_cap)
        reward = nutrition_score - 0.2 * cost_penalty - 0.5 * calorie_penalty - diversity_penalty
        return {
            "nutrition_score": nutrition_score,
            "cost_penalty": cost_penalty,
            "calorie_penalty": calorie_penalty,
            "diversity_penalty": diversity_penalty,
            "reward": reward,
        }

    def step(self, action: int):
        meal = self.meals[action]
        self.cum_calories += meal["calories"]
        self.cum_cost += meal["cost"]
        self.cum_protein += meal["protein"]
        self.cum_fibre += meal["fibre"]
        comps = self.reward_components_if(action)
        reward = comps["reward"]
        self.meal_counts[action] += 1
        self.history.append(action)
        self.day += 1
        done = self.day >= self.max_days
        return self._get_state(), reward, done

    def evaluate_policy(self, policy_fn, n_episodes: int = 100):
        totals_r, totals_c, totals_n, totals_d = [], [], [], []
        for _ in range(n_episodes):
            state = self.reset()
            ep_r = 0.0
            chosen = []
            for _d in range(self.max_days):
                a = policy_fn(state)
                chosen.append(a)
                state, r, done = self.step(a)
                ep_r += r
                if done: break
            totals_r.append(ep_r)
            totals_c.append(self.cum_cost)
            prot_bal = self.cum_protein / (self.daily_protein_target * self.max_days)
            fib_bal  = self.cum_fibre  / (self.daily_fibre_target  * self.max_days)
            totals_n.append((prot_bal + fib_bal) / 2.0)
            counts = Counter(chosen)
            probs = [c / len(chosen) for c in counts.values()]
            entropy = -sum(p * math.log(p) for p in probs)
            totals_d.append(entropy)
        return {
            "avg_reward": sum(totals_r)/n_episodes,
            "avg_cost": sum(totals_c)/n_episodes,
            "avg_nutrition": sum(totals_n)/n_episodes,
            "avg_diversity": sum(totals_d)/n_episodes,
        }

class QLearningAgent:
    def __init__(self, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q: Dict[Tuple[int, int, int], list[float]] = defaultdict(lambda: [0.0] * n_actions)

    def select_action(self, state: Tuple[int, int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        qs = self.q[state]
        m = max(qs)
        best = [i for i, v in enumerate(qs) if v == m]
        return random.choice(best)

    def update(self, state, action, reward, next_state, done: bool):
        current = self.q[state][action]
        next_max = 0.0 if done else max(self.q[next_state])
        target = reward + self.gamma * next_max
        self.q[state][action] += self.alpha * (target - current)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
