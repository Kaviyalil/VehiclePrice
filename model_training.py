# Import pandas library for data handling and analysis
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import pickle to save the trained model
import pickle

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import preprocessing tools: LabelEncoder (for categorical data), StandardScaler (for scaling)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import KMeans clustering algorithm
from sklearn.cluster import KMeans

# Import Decision Tree Regressor model
from sklearn.tree import DecisionTreeRegressor

# Import Support Vector Regression model
from sklearn.svm import SVR

# Import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── 1. Load & Preprocess ─────────────────────────────────────────────

# Read dataset from CSV file
df = pd.read_csv("dataset.csv")

# Remove rows with missing values
df.dropna(inplace=True)

# Define categorical columns that need encoding
cat_cols = ["Make", "Model", "Transmission", "Fuel_Type", "Owner_Type"]

# Create dictionary to store encoders
encoders = {}

# Loop through each categorical column
for col in cat_cols:
    
    # Create LabelEncoder object
    le = LabelEncoder()
    
    # Convert categorical text data into numeric form
    df[col] = le.fit_transform(df[col])
    
    # Store encoder for future use
    encoders[col] = le


# ── 2. K-Means Clustering (Feature Engineering) ─────────────────────

# Initialize StandardScaler for normalization
scaler = StandardScaler()

# Select features for clustering
features = ["Year", "Engine_Size", "Mileage"]

# Scale selected features
scaled = scaler.fit_transform(df[features])

# Initialize KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

# Fit model and create new "Cluster" feature
df["Cluster"] = kmeans.fit_predict(scaled)


# ── 3. Train/Test Split ─────────────────────────────────────────────

# Separate input features (X) by dropping target column "Price"
X = df.drop("Price", axis=1)

# Define target variable (y)
y = df["Price"]

# Initialize feature scaler
feat_scaler = StandardScaler()

# Scale all features
X_scaled = feat_scaler.fit_transform(X)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ── 4. Locally Weighted Regression (Custom Implementation) ─────────

# Define function for LWR prediction
def lwr_predict(X_train, y_train, X_test, tau=1.0):
    
    # Initialize list to store predictions
    predictions = []
    
    # Loop through each test sample
    for x in X_test:
        
        # Compute difference between training points and test point
        diff = X_train - x
        
        # Compute Gaussian weights based on distance
        weights = np.exp(-np.sum(diff ** 2, axis=1) / (2 * tau ** 2))
        
        # Convert weights into diagonal matrix
        W = np.diag(weights)
        
        # Add bias (intercept term) to training data
        X_b = np.c_[np.ones(len(X_train)), X_train]
        
        # Add bias to test sample
        x_b = np.r_[1, x]
        
        try:
            # Compute regression parameters (theta)
            theta = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y_train.values)
            
            # Predict output for test sample
            predictions.append(x_b @ theta)
        
        except np.linalg.LinAlgError:
            # If error occurs, use mean value as fallback
            predictions.append(np.mean(y_train))
    
    # Return predictions as numpy array
    return np.array(predictions)


# ── 5. Train Models ────────────────────────────────────────────────

# Initialize Decision Tree model
dt = DecisionTreeRegressor(max_depth=6, random_state=42)

# Train Decision Tree model
dt.fit(X_train, y_train)

# Predict using Decision Tree
dt_pred = dt.predict(X_test)


# Initialize SVR model with RBF kernel
svr = SVR(kernel="rbf", C=100000, epsilon=1000)

# Train SVR model
svr.fit(X_train, y_train)

# Predict using SVR
svr_pred = svr.predict(X_test)


# Generate predictions using LWR
lwr_pred = lwr_predict(X_train, y_train, X_test, tau=1.5)


# ── 6. Evaluate Models ─────────────────────────────────────────────

# Define function to evaluate model performance
def evaluate(name, y_true, y_pred):
    
    # Calculate Mean Absolute Error
    mae  = mean_absolute_error(y_true, y_pred)
    
    # Calculate Mean Squared Error
    mse  = mean_squared_error(y_true, y_pred)
    
    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Calculate R² score (accuracy)
    r2   = r2_score(y_true, y_pred)
    
    # Return results as dictionary
    return {"Model": name, "MAE": round(mae,2), "MSE": round(mse,2),
            "RMSE": round(rmse,2), "R2": round(r2,4)}


# Evaluate all models and store results
results = [
    evaluate("Decision Tree", y_test, dt_pred),
    evaluate("LWR",           y_test, lwr_pred),
    evaluate("SVR",           y_test, svr_pred),
]

# Convert results into DataFrame
results_df = pd.DataFrame(results).set_index("Model")

# Print comparison table
print("\nModel Comparison:\n")
print(results_df.to_string())


# ── 7. Select Best Model ───────────────────────────────────────────

# Find model with highest R² score
best_row  = results_df["R2"].idxmax()

# Get best R² value
best_r2   = results_df.loc[best_row, "R2"]

# Map model names to objects
model_map = {"Decision Tree": dt, "LWR": None, "SVR": svr}

# Print best model information
print(f'\n>> The best model is "{best_row}" with R2 accuracy = {best_r2}')


# ── 8. Save Best Model ─────────────────────────────────────────────

# Select best model object
best_model = model_map[best_row]

# Create dictionary containing all required objects
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

# Save model and objects to file
with open("model.pkl", "wb") as f:
    pickle.dump(save_obj, f)

# Print confirmation
print(f'model.pkl saved (best model: {best_row})')


# ── 9. Visualization ───────────────────────────────────────────────

# Create subplot with 2 graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Define colors (highlight best model)
colors = ["#4C72B0" if m != best_row else "#DD8452" for m in results_df.index]

# Plot R² scores
axes[0].bar(results_df.index, results_df["R2"], color=colors)
axes[0].set_title("R² Score Comparison")
axes[0].set_ylabel("R² Score")
axes[0].set_ylim(0, 1)

# Add value labels on bars
for i, v in enumerate(results_df["R2"]):
    axes[0].text(i, v + 0.01, str(v), ha="center", fontsize=10)


# Plot RMSE values
axes[1].bar(results_df.index, results_df["RMSE"], color=colors)
axes[1].set_title("RMSE Comparison")
axes[1].set_ylabel("RMSE")

# Add labels on bars
for i, v in enumerate(results_df["RMSE"]):
    axes[1].text(i, v + 50, str(v), ha="center", fontsize=10)


# Add main title
fig.suptitle(f"Best Model: {best_row} (highlighted in orange)", fontsize=13)

# Adjust layout
plt.tight_layout()

# Save plot as image
plt.savefig("static/model_comparison.png")

# Display plot
plt.show()

# Print confirmation
print("Comparison chart saved to static/model_comparison.png")