import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- STEP 1: LOAD THE BALANCED DATASET ---
data = pd.read_csv("upi_fraud_dataset_balanced.csv")

# --- STEP 2: CLEANING & FEATURE SELECTION ---
X = data[['trans_hour', 'trans_day', 'trans_month', 'trans_amount', 'age']]
y = data['fraud_risk']

# --- STEP 3: SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- STEP 4: SCALE ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler immediately so app.py can use it
joblib.dump(scaler, "scaler.pkl")

# --- STEP 5: TRAIN MODELS ---
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True)
}

print("Training Models on Balanced Dataset...")
print("-" * 60)

# Track the best model based on RECALL (Crucial for Fraud)
best_score = 0
best_model_name = ""
best_model_obj = None

for model_name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {acc:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    
    # Detailed report for your own analysis (Great for Resumes!)
    print(classification_report(y_test, y_pred))
    
    # LOGIC: We choose the model with the highest RECALL (catching fraud)
    # You can swap 'recall' with 'f1' if you want a balance.
    if recall > best_score:
        best_score = recall
        best_model_name = model_name
        best_model_obj = model

print("-" * 60)
print(f"Winner: {best_model_name} with Recall: {best_score:.4f}")

# --- STEP 6: SAVE THE BEST MODEL ---
# We save it as a GENERIC name so app.py doesn't crash
joblib.dump(best_model_obj, "best_model.pkl")
print(f"Saved {best_model_name} as 'best_model.pkl'")