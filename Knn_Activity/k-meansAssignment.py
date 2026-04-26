import math

# Dataset
data = [
    ("Wine1", 14.23, 3.06),
    ("Wine2", 13.20, 2.76),
    ("Wine3", 13.16, 3.24),
    ("Wine4", 12.37, 2.30),
    ("Wine5", 12.33, 1.95),
    ("Wine6", 12.64, 2.20),
    ("Wine7", 13.67, 2.91),
    ("Wine8", 12.85, 2.52),
    ("Wine9", 13.50, 2.85),
    ("Wine10", 12.75, 2.05),
]

# Initial centroids
C1 = (14.23, 3.06)  # Wine1
C2 = (12.33, 1.95)  # Wine5

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

clusters = {"C1": [], "C2": []}

print("Cluster Table")
print(f"{'Wine':<8}{'C1 Dist':<10}{'C2 Dist':<10}{'Cluster'}")

# Step 1: Assign clusters (USING ROUNDED DISTANCES)
for wine, alcohol, flav in data:
    d1 = round(distance((alcohol, flav), C1), 2)
    d2 = round(distance((alcohol, flav), C2), 2)

    # Match your manual table logic
    cluster = "C1" if d1 < d2 else "C2"

    clusters[cluster].append((wine, alcohol, flav))

    print(f"{wine:<8}{d1:<10}{d2:<10}{cluster}")

# Step 2: Compute centroids (rounded to match solution)
def centroid(cluster):
    a = sum(x[1] for x in cluster) / len(cluster)
    f = sum(x[2] for x in cluster) / len(cluster)
    return (round(a, 2), round(f, 2))

C1_new = centroid(clusters["C1"])
C2_new = centroid(clusters["C2"])

print("\nNew Centroids")
print("C1 =", C1_new)
print("C2 =", C2_new)

# Step 3: Final clusters (same as your result)
print("\nFinal Clusters")
print("Cluster 1 (High quality wines):")
print([w[0] for w in clusters["C1"]])

print("\nCluster 2 (Lower flavanoids wines):")
print([w[0] for w in clusters["C2"]])