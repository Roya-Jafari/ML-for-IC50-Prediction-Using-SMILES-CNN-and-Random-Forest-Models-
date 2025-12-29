
# -------------------
# Random Forest Model
# -------------------

# Safety: remove inf and clip values
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64)

# --- Feature selection: remove low-variance descriptors ---
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.01)  # remove features with variance < 0.01
X_train = sel.fit_transform(X_train)
X_test  = sel.transform(X_test)

# --- Now fit Random Forest ---

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

rf = RandomForestRegressor(
    n_estimators=50,      # start small
    max_depth=None,       # unlimited depth
    min_samples_leaf=3,   # small leaves
    max_features='sqrt',  # sqrt(num_features) per split
    warm_start=True,      # allow incremental training
    oob_score=True,   # 👈 add this
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

for i in range(50):  # total trees = 50 + 50*50 = 2550
    rf.n_estimators += 50
    rf.fit(X_train, y_train)
    print(f"Trained {rf.n_estimators} trees")

rf_pred = rf.predict(X_test)

print("\nRandom Forest Results")
print("R²:", r2_score(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
