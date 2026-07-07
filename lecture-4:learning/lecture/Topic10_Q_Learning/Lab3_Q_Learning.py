import numpy as np
import matplotlib as _mpl
_mpl.use("Agg")  # headless backend: save figures to files instead of opening a window
import matplotlib.pyplot as plt
import os as _os
_os.makedirs("plots", exist_ok=True)
_FIG_N = [0]
def _save():
    """Save the current figure to plots/figure_NN.png (replaces plt.show)."""
    _FIG_N[0] += 1
    plt.savefig(_os.path.join("plots", f"figure_{_FIG_N[0]:02d}.png"), dpi=150, bbox_inches="tight")
    plt.close()
import pandas as pd

# 2x3 grid of rewards
# Row 0 (top):    [Start(-10), 1, 0]
# Row 1 (bottom): [0,         -10, 10]
env = np.array([
    [-10,   1,   0],   # row 0: Start=top-left, 1=top-mid, 0=top-right
    [  0, -10,  10]    # row 1: 0=bot-left, -10=danger, 10=GOAL
])

ROWS, COLS = env.shape
NUM_STATES  = ROWS * COLS   # 6 states (0..5)
NUM_ACTIONS = 4             # Up=0, Down=1, Right=2, Left=3

# State index mapping: state = row * COLS + col
# State 0 = (0,0) Start | State 1 = (0,1) | State 2 = (0,2)
# State 3 = (1,0)       | State 4 = (1,1) danger | State 5 = (1,2) GOAL

START_STATE = 0   # (row=0, col=0)
GOAL_STATE  = 5   # (row=1, col=2) → reward 10

print('Environment (reward grid):')
print(env)
print(f'\nStates: {NUM_STATES}, Actions: {NUM_ACTIONS}')
print('Action mapping: 0=Up, 1=Down, 2=Right, 3=Left')

# Q-table: shape (6 states, 4 actions), initialised to zeros
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

print('Initial Q-Table (states × actions):')
df_q = pd.DataFrame(Q, columns=['Up', 'Down', 'Right', 'Left'],
                    index=[f'S{i}' for i in range(NUM_STATES)])
print(df_q)

def state_to_rc(state):
    """Convert flat state index to (row, col) coordinates."""
    return divmod(state, COLS)   # e.g. state 4 → (1, 1)

def rc_to_state(row, col):
    """Convert (row, col) back to flat state index."""
    return row * COLS + col

def step(state, action):
    """
    Take an action from the current state.
    Returns: (next_state, reward, done)
    
    If the action would move the agent off-grid, the agent stays in the same state.
    """
    row, col = state_to_rc(state)

    # Compute new position based on action
    if action == 0:   # Up
        new_row, new_col = row - 1, col
    elif action == 1: # Down
        new_row, new_col = row + 1, col
    elif action == 2: # Right
        new_row, new_col = row, col + 1
    elif action == 3: # Left
        new_row, new_col = row, col - 1

    # Boundary check — stay in grid
    if 0 <= new_row < ROWS and 0 <= new_col < COLS:
        next_state = rc_to_state(new_row, new_col)
    else:
        next_state = state   # wall → stay put
        new_row, new_col = row, col

    reward = env[new_row, new_col]  # reward at the new cell
    done   = (next_state == GOAL_STATE)  # episode ends at goal

    return next_state, reward, done

print('Helper functions defined: state_to_rc, rc_to_state, step')

# ── Hyperparameters ─────────────────────────────────────────────
ALPHA        = 0.1      # Learning Rate: how much new info overrides old Q-values
GAMMA        = 0.9      # Discount Factor: how much future rewards are valued
EPSILON      = 0.1      # Exploration rate: 10% random action, 90% greedy (best Q)
EPISODES     = 20000    # Total training iterations
MAX_STUCK    = 4        # Optional: terminate episode if 4 moves don't reach goal
# ─────────────────────────────────────────────────────────────────

# Track total reward per episode for plotting convergence
episode_rewards = []

for episode in range(EPISODES):
    state = START_STATE   # always begin at Start
    total_reward = 0
    stuck_counter = 0     # counts non-goal moves

    while True:
        # ── Epsilon-greedy action selection ──────────────────────
        # With probability EPSILON → explore (random action)
        # Otherwise             → exploit (action with highest Q-value)
        if np.random.uniform(0, 1) < EPSILON:
            action = np.random.randint(NUM_ACTIONS)   # random action
        else:
            action = np.argmax(Q[state])              # greedy action

        # Take the action, observe next state and reward
        next_state, reward, done = step(state, action)

        # ── Q-value update (Bellman equation) ────────────────────
        # Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]
        best_next_q = np.max(Q[next_state])           # best Q in next state
        td_target   = reward + GAMMA * best_next_q    # temporal difference target
        td_error    = td_target - Q[state, action]    # how wrong was our estimate
        Q[state, action] += ALPHA * td_error          # update Q-value

        total_reward += reward
        state         = next_state

        # Optional: end episode early if stuck (4+ non-goal moves)
        if not done:
            stuck_counter += 1
            if stuck_counter >= MAX_STUCK:
                break   # terminate this episode to avoid infinite loops

        if done:
            break   # reached the goal — end episode

    episode_rewards.append(total_reward)

print(f'Training complete: {EPISODES} episodes')
print(f'Average reward (last 1000 episodes): {np.mean(episode_rewards[-1000:]):.4f}')

# ── FINAL Q-VALUES ── (this cell output is the deliverable)
state_labels = ['S0 (Start)', 'S1 (1)', 'S2 (0)', 'S3 (0)', 'S4 (-10 Danger)', 'S5 (10 GOAL)']

df_final = pd.DataFrame(
    np.round(Q, 4),
    columns=['Up', 'Down', 'Right', 'Left'],
    index=state_labels
)
print('='*60)
print('       FINAL Q-TABLE AFTER TRAINING')
print('='*60)
print(df_final.to_string())
print('='*60)
print()
print('Optimal action per state (greedy policy):')
action_names = ['Up', 'Down', 'Right', 'Left']
for i, label in enumerate(state_labels):
    best_action = action_names[np.argmax(Q[i])]
    best_value  = np.max(Q[i])
    print(f'  {label:30s} → Best action: {best_action:5s}  (Q={best_value:.4f})')

# Visualise training convergence — rolling average of episode rewards
window = 500
rolling_avg = pd.Series(episode_rewards).rolling(window).mean()

plt.figure(figsize=(12, 4))
plt.plot(rolling_avg, color='steelblue', linewidth=1.5)
plt.title(f'Q-Learning Convergence — Rolling Average Reward (window={window})', fontsize=13)
plt.xlabel('Episode')
plt.ylabel('Avg Total Reward')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
_save()

# Visualise the Q-table as a heatmap for easy reading
plt.figure(figsize=(8, 5))
sns_data = pd.DataFrame(np.round(Q, 2), columns=['Up', 'Down', 'Right', 'Left'],
                        index=[f'S{i}' for i in range(NUM_STATES)])
import seaborn as sns
sns.heatmap(sns_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.5, cbar_kws={'label': 'Q-Value'})
plt.title('Q-Table Heatmap (States × Actions)', fontsize=13)
plt.tight_layout()
_save()
