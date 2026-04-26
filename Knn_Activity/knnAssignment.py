import math

# Dataset: (Wine, Alcohol, Flavanoids, Class)
data = [
    ("Wine1", 14.23, 3.06, "A"),
    ("Wine2", 13.20, 2.76, "A"),
    ("Wine3", 13.16, 3.24, "A"),
    ("Wine4", 12.37, 2.30, "B"),
    ("Wine5", 12.33, 1.95, "B"),
    ("Wine6", 12.64, 2.20, "B"),
    ("Wine7", 13.67, 2.91, "A"),
    ("Wine8", 12.85, 2.52, "B"),
    ("Wine9", 13.50, 2.85, "A"),
    ("Wine10", 12.75, 2.05, "B"),
]

# Test wine
test_point = (13.00, 2.50)

# Euclidean distance function
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Step 1: Compute distances
distances = []
for wine, alcohol, flav, label in data:
    dist = euclidean_distance((alcohol, flav), test_point)
    distances.append((wine, dist, label))

# Step 2: Sort distances
distances.sort(key=lambda x: x[1])

# Print distance table
print("Sorted Distances:")
for wine, dist, label in distances:
    print(f"{wine}: {dist:.3f}, Class={label}")

# Step 3: KNN (k=3)
k = 3
neighbors = distances[:k]

# Voting
votes = {}
for _, _, label in neighbors:
    votes[label] = votes.get(label, 0) + 1

# Prediction
predicted_class = max(votes, key=votes.get)

print("\nK Nearest Neighbors:")
for n in neighbors:
    print(n)

print("\nVotes:", votes)
print("Predicted Class:", predicted_class)