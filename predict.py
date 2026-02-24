import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1. Load trained model
# ==============================
model = joblib.load("ids_model.pkl")
print("IDS model loaded!")

# ==============================
# 2. Define column names (IMPORTANT)
# ==============================
columns = [
'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
'num_compromised','root_shell','su_attempted','num_root',
'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login','is_guest_login','count','srv_count','serror_rate',
'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
'dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate',
'label','difficulty'
]


data = pd.read_csv("dataset/KDDTest+.txt", names=columns)


data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)


categorical_cols = ['protocol_type', 'service', 'flag']

for col in categorical_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])


X = data.drop(['label', 'difficulty'], axis=1)

# Take first row as sample
sample = X.iloc[0:1].values


prediction = model.predict(sample)

if prediction[0] == 0:
    print("Prediction: NORMAL traffic")
else:
    print("Prediction: ATTACK detected")
