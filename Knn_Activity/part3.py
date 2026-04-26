import matplotlib.pyplot as plt
import numpy as np

# Test point (center)
test = np.array([5, 3])

groups = [
    [("Row 5", 0), ("Row 2", 1), ("Row 9", 1)],  # K = 3
    [("Row 10", 0), ("Row 1", 0)],              # 2 → K = 5
    [("Row 7", 0), ("Row 3", 0)],               #2 → K = 7
    [("Row 6", 1), ("Row 4", 1), ("Row 8", 1)]  # outside
]

# Radii for placing points (layers)
point_radii = [0.8, 1.6, 2.4, 3.4]

points = []

# ---- Place points in circular layers ----
for layer_idx, group in enumerate(groups):
    radius = point_radii[layer_idx]
    angle_step = 2 * np.pi / len(group)

    for i, (name, label) in enumerate(group):
        angle = i * angle_step + layer_idx  # slight rotation per layer
        x = test[0] + radius * np.cos(angle)
        y = test[1] + radius * np.sin(angle)
        points.append((name, x, y, label))

# ---- Plot ----
plt.figure(figsize=(8, 8))

# Plot training points
for name, x, y, label in points:
    color = 'gold' if label == 0 else 'purple'
    plt.scatter(x, y, color=color, s=200, edgecolor='black')
    plt.text(x, y, name.split()[1], ha='center', va='center', weight='bold')

# Plot test point
plt.scatter(test, color='red', marker='', s=300, label='Test Point')

# ---- Draw K circles (slightly larger than point layers) ----
circle_radii = [
    point_radii[0] + 0.4,  # K = 3
    point_radii[1] + 0.5,  # K = 5
    point_radii[2] + 0.6   # K = 7
]

k_vals = [3, 5, 7]
colors = ['green', 'blue', 'purple']

for r, k, c in zip(circle_radii, k_vals, colors):
    circle = plt.Circle(test, r, fill=False, linestyle='--', color=c, linewidth=2)
    plt.gca().add_patch(circle)
    plt.text(test[0] + r, test[1], f"K={k}", color=c, fontsize=10)

# ---- Styling ----
plt.gca().set_aspect('equal')
plt.xlim(1, 9)
plt.ylim(0, 7)
plt.grid(True)

plt.title("KNN Visualization (Correct Grouping & Circles)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.legend()
plt.show()