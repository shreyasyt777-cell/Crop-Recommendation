# model_training.py
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ---------------------------
# Load Dataset
# ---------------------------
PATH = "Crop_recommendation.csv"
if not os.path.exists(PATH):
    print("❌ Dataset file not found! Please ensure 'Crop_recommendation.csv' is in the same folder.")
    exit()

df = pd.read_csv(PATH)
print("✅ Dataset Loaded Successfully!")

# Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# Features and labels
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train RandomForest model
rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
rf.fit(X_train_s, y_train)

# Evaluate model
y_pred = rf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
top3_acc = top_k_accuracy_score(y_test, rf.predict_proba(X_test_s), k=3)

print(f"🎯 Model Accuracy: {acc*100:.2f}%")
print(f"🌾 Top-3 Accuracy: {top3_acc*100:.2f}%")

# Save model, scaler, and label encoder
with open("crop_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model, Scaler, and Label Encoder saved successfully!")
