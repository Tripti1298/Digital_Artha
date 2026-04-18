# ================= INSTALL =================
%pip install xgboost
%pip install shap
%pip install networkx

# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import networkx as nx
import joblib
import os

# ================= LOAD =================
df = pd.read_csv('/Volumes/workspace/default/upi_dataset/PS_20174392719_1491204439457_log.csv')

# ================= FILTER =================
df = df[df['type'].isin(['TRANSFER', 'PAYMENT'])]

# ================= FEATURES =================
df['hour'] = df['step'] % 24
df['is_night'] = (df['hour'] < 6).astype(int)
df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['orig_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
df['dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)

# ================= GRAPH DATA =================
df_graph = df[['nameOrig','nameDest','amount']].copy()

# ================= MODEL DATA =================
df_model = pd.get_dummies(df.copy(), columns=['type'], drop_first=True)
df_model.drop(['nameOrig','nameDest'], axis=1, inplace=True)

X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================= MODEL =================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    ))
])

# ================= TUNING =================
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 4, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.7, 0.8],
    'model__colsample_bytree': [0.7, 0.8]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=5,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# ================= GRAPH =================
pdf = df_graph.sample(50000, random_state=42)

G = nx.DiGraph()
G.add_weighted_edges_from(pdf.values)

out_amt = dict(G.out_degree(weight='weight'))
tx_count = dict(G.out_degree())
unique_receivers = {n: len(list(G.successors(n))) for n in G.nodes()}

def normalize(d):
    if not d: return {}
    m = max(d.values())
    return {k: v/m if m else 0 for k,v in d.items()}

out_amt = normalize(out_amt)
tx_count = normalize(tx_count)
unique_receivers = normalize(unique_receivers)

graph_scores = {}
for n in G.nodes():
    graph_scores[n] = (
        0.4*out_amt.get(n,0) +
        0.3*tx_count.get(n,0) +
        0.3*unique_receivers.get(n,0)
    )

graph_scores = {k: np.log1p(v) for k,v in graph_scores.items()}
m = max(graph_scores.values()) if graph_scores else 1
graph_scores = {k: v/m for k,v in graph_scores.items()}

# ================= SAVE =================
save_path = "/Workspace/Files"
os.makedirs(save_path, exist_ok=True)

joblib.dump(best_model, f"{save_path}/fraud_best_model.pkl")
joblib.dump(graph_scores, f"{save_path}/fraud_graph_scores.pkl")

print("SAVED SUCCESSFULLY")
