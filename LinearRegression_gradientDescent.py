import pandas as pd
import numpy as np

# Load data
file_path = r"C:\Users\User\OneDrive - UOW Malaysia KDU\Documents\Education\01 Degree Studies\Y2S1\IS\Assignment\SET C.csv"
df = pd.read_csv(file_path)

X = df["Hours"].values.reshape(-1, 1)
y = df["Scores"].values

print(f"Dataset: {len(X)} samples")
print()

# Feature Scaling
mean_X = np.mean(X)
std_X = np.std(X)
X_scaled = (X - mean_X) / std_X

print("Feature Scaling Applied:")
print(f"Mean: {mean_X:.4f}, Std: {std_X:.4f}")
print()

# Add intercept term
X_b = np.column_stack([np.ones(len(X_scaled)), X_scaled])

# Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Initialize theta to zeros
theta = np.zeros(2)
print(f"Initial theta: {theta}")

# Compute initial cost with theta = [0, 0]
initial_cost = compute_cost(X_b, y, theta)
print(f"Initial Cost J(theta): {initial_cost:.4f}")
print()

# Gradient Descent
alpha = 0.01
iterations = 1500
m = len(y)

print(f"Learning rate (alpha): {alpha}")
print(f"Iterations: {iterations}")
print()

cost_history = []

for i in range(iterations):
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X_b.T.dot(errors)
    theta = theta - alpha * gradient
    
    cost = compute_cost(X_b, y, theta)
    cost_history.append(cost)
    
    if (i + 1) % 300 == 0:
        print(f"Iteration {i+1}: Cost = {cost:.4f}, theta = [{theta[0]:.4f}, {theta[1]:.4f}]")

print()

# Final Results
print("Final Results:")
print(f"Optimized theta (scaled): [{theta[0]:.4f}, {theta[1]:.4f}]")
print(f"Final Cost: {cost_history[-1]:.4f}")
print(f"Cost Reduction: {initial_cost - cost_history[-1]:.4f}")
print()

# Convert to original scale
slope = theta[1] / std_X
intercept = theta[0] - (slope * mean_X)

print(f"Final Model: Score = {intercept:.4f} + {slope:.4f} * Hours")
print()

print("Done!")