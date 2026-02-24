import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Column names for NSL-KDD dataset
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

train_path = "dataset/KDDTrain+.txt"
test_path = "dataset/KDDTest+.txt"

# Load data
train_data = pd.read_csv(train_path, names=columns)
test_data = pd.read_csv(test_path, names=columns)

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

# Convert labels to binary
train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical columns
encoder = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.transform(test_data[col])

print("\nCategorical columns encoded successfully!")

# Prepare features and labels
X_train = train_data.drop(['label', 'difficulty'], axis=1)
y_train = train_data['label']

X_test = test_data.drop(['label', 'difficulty'], axis=1)
y_test = test_data['label']

print("\nFeature shape:", X_train.shape)
print("Label shape:", y_train.shape)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model =  RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


import joblib

joblib.dump(model, "ids_model.pkl")
print("IDS model saved successfully!")
