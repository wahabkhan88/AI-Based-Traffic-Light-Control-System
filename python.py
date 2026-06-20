import random
import matplotlib.pyplot as plt

NUM_LANES = 4
STEPS = 500
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPSILON = 0.2
AGING_WEIGHT = 0.3

lane_densities = [0] * NUM_LANES
waiting_time = [0] * NUM_LANES
emergency = [0] * NUM_LANES
green_light = 0
signal_timer = 0

q_table = {}
history = {'densities': [], 'rewards': []}

def get_state():
    return tuple(lane_densities + waiting_time + emergency + [green_light])

def calculate_reward():
    return -sum(lane_densities) - 0.2 * sum(waiting_time)

print("Starting AI Traffic Simulation...\n")

for step in range(STEPS):

    is_rush_hour = (step // 60) % 24 in [8, 9, 17, 18]
    prob = 0.4 if is_rush_hour else 0.2

    for i in range(NUM_LANES):
        if random.random() < prob:
            lane_densities[i] += 1

    if random.random() < 0.02:
        lane = random.randint(0, NUM_LANES - 1)
        emergency[lane] = 1
        print(f"🚑 Ambulance detected in Lane {lane}")

    state = get_state()
    if state not in q_table:
        q_table[state] = [0.0] * NUM_LANES

    if random.random() < EPSILON:
        action = random.randint(0, NUM_LANES - 1)
    else:
        action = q_table[state].index(max(q_table[state]))

    if 1 in emergency:
        action = emergency.index(1)

    priorities = [
        lane_densities[i] + AGING_WEIGHT * waiting_time[i]
        for i in range(NUM_LANES)
    ]
    action = priorities.index(max(priorities))

    if action != green_light:
        green_light = action
        signal_timer = min(max(5, lane_densities[green_light] * 2), 60)

    if signal_timer > 0 and lane_densities[green_light] > 0:
        passed = min(2, lane_densities[green_light])
        lane_densities[green_light] -= passed
        signal_timer -= 1
        waiting_time[green_light] = 0
        emergency[green_light] = 0

    for i in range(NUM_LANES):
        if i != green_light:
            waiting_time[i] += 1

    reward = calculate_reward()
    next_state = get_state()
    if next_state not in q_table:
        q_table[next_state] = [0.0] * NUM_LANES

    old_value = q_table[state][action]
    next_max = max(q_table[next_state])
    q_table[state][action] = old_value + LEARNING_RATE * (
        reward + DISCOUNT * next_max - old_value
    )

    history['densities'].append(list(lane_densities))
    history['rewards'].append(reward)

    if step % 100 == 0:
        print(f"Step {step} | Green: Lane {green_light} | Queues: {lane_densities} | Waiting: {waiting_time} | Reward: {reward}")

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(history['densities'])
plt.title("Traffic Density per Lane")
plt.ylabel("Cars Waiting")

plt.subplot(2, 1, 2)
plt.plot(history['rewards'], color='green')
plt.title("AI Performance (Reward)")
plt.ylabel("Reward")

plt.tight_layout()
plt.show()

print(f"\nSimulation Finished. Q-Table learned {len(q_table)} states.")
