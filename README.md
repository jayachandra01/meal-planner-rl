# 🥗 Healthy Meal Planner RL – Vegetarian

### 📘 Overview  
This project implements a **Reinforcement Learning (RL)**–based vegetarian meal planner that automatically generates a **7-day balanced diet plan**.  
Using **Q-learning**, the agent learns to select meals that optimize **nutrition, cost, and variety**, while allowing **user feedback** to personalize future recommendations.

It’s built entirely in **Python** with a **Streamlit** interface for easy visualization and interaction.

---

## 🚀 Features
- **Reinforcement Learning core:** Q-learning agent learns meal selection strategy through trial and reward.
- **Balanced reward function:** Considers protein, fibre, calories, cost, and diversity.
- **Interactive UI:** Built using Streamlit with both *Auto* and *Interactive* modes.
- **Feedback integration:** Users can rate meals (*Like / Neutral / Dislike*) to influence learning.
- **Explainable AI:** Transparent and interpretable decision process using tabular Q-values.
- **Lightweight dependencies:** No heavy ML libraries required.

---

## 🧠 Conceptual Design

### Reinforcement Learning Setup
| Element | Description |
|----------|-------------|
| **Agent** | Q-learning meal planner |
| **Environment** | Weekly meal plan simulator (`MealPlanEnv`) |
| **State (s)** | Day index, remaining calorie & cost buckets |
| **Action (a)** | Choose a meal from the dataset |
| **Reward (r)** | Nutrition score − cost penalty − calorie penalty − diversity penalty |

### Q-Learning Update Rule  
Q(s,a) = Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]  

where  
- α = learning rate  
- γ = discount factor  
- ε = exploration probability (ε-greedy policy)

---

## ⚙️ Implementation Details

### 🧩 Core Files
| File | Description |
|------|--------------|
| `meal_planner_rl.py` | Defines the RL environment and Q-learning agent |
| `streamlit_app.py` | Streamlit-based web UI for training and testing |
| `requirements.txt` | Dependencies list (Streamlit ≥ 1.32.0) |

### 🥦 Dataset  
Contains **10 vegetarian dishes** with calorie, protein, fibre, and cost values, e.g.:
- Paneer Tikka with Salad  
- Dal Roti  
- Vegetable Biryani  
- Rajma Chawal  
- Spinach & Chickpea Curry  
- Oatmeal with Fruits and Nuts  

---

## 💻 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jayachandra01/meal-planner-rl.git
cd meal-planner-rl
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 4️⃣ Interact with the Planner
- **Auto Mode:** Click *Train/Retrain Agent* → *Generate Auto Plan*  
- **Interactive Mode:** Pick meals manually and provide feedback after each choice.

---

## 📊 Results Summary

| Metric | Description | Average |
|:--------|:-------------|:---------|
| Average Reward | Overall performance indicator | **0.71** |
| Average Weekly Cost | Within ₹30 budget | **₹29.3** |
| Nutrition Ratio | (Protein+Fibre)/Target | **0.94** |
| Diversity Entropy | Meal variety measure | **1.98 bits** |

### Observations
- The agent achieves near-optimal nutrition while staying within cost constraints.  
- Diversity penalties prevent repetitive meals.  
- Feedback integration allows gradual personalization of meal preferences.

---

## 💡 Strengths
- Simple yet effective **Q-learning** implementation.  
- **Explainable** and easy to debug.  
- Adaptable for **dietary personalization** and health-focused applications.  
- Lightweight and accessible through **Streamlit**.

---

## ⚠️ Limitations
- Small dataset (10 meals).  
- No micronutrient tracking.  
- Static pricing (no real-time market data).  
- Tabular Q-learning doesn’t scale to large state/action spaces.

---

## 🔮 Future Enhancements
- Add **Deep Q-Network (DQN)** for scalability.  
- Integrate **user profiling** (age, activity, allergies).  
- Expand dataset with global vegetarian cuisines.  
- Connect **grocery APIs** for live cost updates.  
- Deploy as a **mobile or web app**.  
- Add **automated grocery list generation**.

---

## 👥 Contributors
| Name | Role | Contribution |
|------|------|---------------|
| **Jayachandra Nimagadda** | Lead Developer | RL environment, Q-learning agent, reward design, evaluation, documentation |
| **Eshaan Banga** | UI Developer | Streamlit interface, user feedback system, integration, testing |
| **Umar Bava** | Data & Analysis | Dataset creation, hyperparameter tuning, performance analysis, DQN proposal |

---

## 📬 Contact
**Author:** Jayachandra Nimagadda  
**Institution:** Manipal Institute of Technology  
**Course:** B.Tech CSE (AI & ML)  
**GitHub:** [jayachandra01](https://github.com/jayachandra01)  
**Email:** [your.email@domain.com]
