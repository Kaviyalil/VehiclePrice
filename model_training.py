import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load & Preprocess ──────────────────────────────────────────────────────
df = pd.read_csv("dataset.csv")
df.dropna(inplace=True)

cat_cols = ["Make", "Model", "Transmission", "Fuel_Type", "Owner_Type"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── 2. K-Means Clustering (feature enhancement) ───────────────────────────────
scaler = StandardScaler()
features = ["Year", "Engine_Size", "Mileage"]
scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled)

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
X = df.drop("Price", axis=1)
y = df["Price"]

feat_scaler = StandardScaler()
X_scaled = feat_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── 4. Locally Weighted Regression (LWR) ─────────────────────────────────────
def lwr_predict(X_train, y_train, X_test, tau=1.0):
    predictions = []
    for x in X_test:
        diff = X_train - x
        weights = np.exp(-np.sum(diff ** 2, axis=1) / (2 * tau ** 2))
        W = np.diag(weights)
        X_b = np.c_[np.ones(len(X_train)), X_train]
        x_b = np.r_[1, x]
        try:
            theta = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y_train.values)
            predictions.append(x_b @ theta)
        except np.linalg.LinAlgError:
            predictions.append(np.mean(y_train))
    return np.array(predictions)

# ── 5. Train Models ───────────────────────────────────────────────────────────
dt = DecisionTreeRegressor(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

svr = SVR(kernel="rbf", C=100000, epsilon=1000)
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

lwr_pred = lwr_predict(X_train, y_train, X_test, tau=1.5)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": round(mae,2), "MSE": round(mse,2),
            "RMSE": round(rmse,2), "R2": round(r2,4)}

results = [
    evaluate("Decision Tree", y_test, dt_pred),
    evaluate("LWR",           y_test, lwr_pred),
    evaluate("SVR",           y_test, svr_pred),
]

results_df = pd.DataFrame(results).set_index("Model")
print("\nModel Comparison:\n")
print(results_df.to_string())

# ── 7. Select Best Model ──────────────────────────────────────────────────────
best_row  = results_df["R2"].idxmax()
best_r2   = results_df.loc[best_row, "R2"]
model_map = {"Decision Tree": dt, "LWR": None, "SVR": svr}

print(f'\n>> The best model is "{best_row}" with R2 accuracy = {best_r2}')

# ── 8. Save Best Model ────────────────────────────────────────────────────────
best_model = model_map[best_row]

save_obj = {
    "model":       best_model,
    "model_name":  best_row,
    "scaler":      feat_scaler,
    "encoders":    encoders,
    "kmeans":      kmeans,
    "km_scaler":   scaler,
    "features":    list(X.columns),
    "lwr_train_X": X_train if best_row == "LWR" else None,
    "lwr_train_y": y_train if best_row == "LWR" else None,
}

with open("model.pkl", "wb") as f:
    pickle.dump(save_obj, f)

print(f'model.pkl saved (best model: {best_row})')

# ── 9. Comparison Plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ["#4C72B0" if m != best_row else "#DD8452" for m in results_df.index]

axes[0].bar(results_df.index, results_df["R2"], color=colors)
axes[0].set_title("R² Score Comparison")
axes[0].set_ylabel("R² Score")
axes[0].set_ylim(0, 1)
for i, v in enumerate(results_df["R2"]):
    axes[0].text(i, v + 0.01, str(v), ha="center", fontsize=10)

axes[1].bar(results_df.index, results_df["RMSE"], color=colors)
axes[1].set_title("RMSE Comparison")
axes[1].set_ylabel("RMSE")
for i, v in enumerate(results_df["RMSE"]):
    axes[1].text(i, v + 50, str(v), ha="center", fontsize=10)

fig.suptitle(f"Best Model: {best_row} (highlighted in orange)", fontsize=13)
plt.tight_layout()
plt.savefig("static/model_comparison.png")
plt.savefig("static/model_comparison.png")
plt.close()
print("Comparison chart saved to static/model_comparison.png")
