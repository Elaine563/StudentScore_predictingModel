import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("SET C.csv") 

x = df["Hours"].values
y = df["Scores"].values

# Use optimized θ from QC
theta0 = 2.484   # interception point
theta1 = 9.776   # slope

def predict(hours):
    return theta0 + theta1 * hours

# Compute predictions
y_pred = predict(x)

#Compute R² Score
ss_res = np.sum((y - y_pred) ** 2)         # residual sum of squares
ss_tot = np.sum((y - np.mean(y)) ** 2)     # total variability
r2 = 1 - (ss_res / ss_tot)

print("Final Linear Regression Model:")
print(f"   h(x) = {theta0} + {theta1}x")
print("\nModel Evaluation:")
print("   R2 Score:", r2)

# 4. Compute MSE MAE
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))

print("   MSE:", mse)
print("   MAE:", mae)

#Prediction Outcome for 8 hrs study
hours_input = 8
prediction_8_hours = predict(hours_input)
print(f"\nPrediction for {hours_input} study hours:", prediction_8_hours)

#Actual vs Predicted
plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred, label="Fitted Line", linewidth=2)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

#Residual Plot
residuals = y - y_pred
plt.scatter(x, residuals)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Hours")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
