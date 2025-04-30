import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("phishing_dataset_updated.csv")

# Adjust based on actual column name in your CSV
non_feature_cols = ['Website_Name', 'URL', 'Result']

# Extract for reporting
website_names = df['Website_Name']
urls = df['URL']

# Prepare features and label
X = df.drop(columns=non_feature_cols)
y = df['Result']

# Split dataset
X_train, X_test, y_train, y_test, urls_train, urls_test, names_train, names_test = train_test_split(
    X, y, urls, website_names, test_size=0.3, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Train SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# Evaluation
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

print("\n=== SVM ===")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

# Show results
results_df = pd.DataFrame({
    'Website_Name': names_test,
    'URL': urls_test,
    'Actual': y_test,
    'RF_Predicted': rf_preds,
    'SVM_Predicted': svm_preds
})
print("\nSample Predictions:\n")
print(results_df.head(10))
