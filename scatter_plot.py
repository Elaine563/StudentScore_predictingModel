import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load CSV
file_path = r"C:\Users\User\OneDrive - UOW Malaysia KDU\Documents\Education\01 Degree Studies\Y2S1\IS\Assignment\SET C.csv"
df = pd.read_csv(file_path)

# Extract columns
x = df["Hours"]
y = df["Scores"]

# Linear regression
m, b = np.polyfit(x, y, 1)

# Line from x=0 to max
x_line = np.linspace(0, x.max(), 200)
y_line = m * x_line + b

plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(
    x, y,
    color='blue', alpha=0.6, s=100,
    edgecolors='black', linewidth=1.2
)

# Regression line
plt.plot(x_line, y_line, linewidth=2, color='black')

# Labels & title
plt.title("Student Study Hours vs Exam Scores", fontsize=16, fontweight='bold')
plt.xlabel("Hours of Study", fontsize=13)
plt.ylabel("Exam Scores", fontsize=13)

# Force Y-axis to start at 0
plt.ylim(bottom=0)

# --- Custom Axis Ticks ---
plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10])
plt.yticks(ticks=range(0, 101, 10))   # 0,10,20,...100

# --- Green graph paper grid ---
plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='green', alpha=0.35)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, color='green', alpha=0.18)
plt.minorticks_on()

plt.tight_layout()
plt.show()
