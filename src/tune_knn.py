import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

CSV_PATH = project_root / "data" / "features.csv"  
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)

exclude_cols = {'ID', 'Image_ID', 'Diagnosis', 'Cancer'}
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['Cancer'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_neighbors': list(range(1, 51, 2)),  
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan','minkowski']
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',   
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\n BEST RESULT")
print("Best n_neighbors:", grid.best_params_['n_neighbors'])
print("Best weights:", grid.best_params_['weights'])
print("Best metric:", grid.best_params_['metric'])
print("Best CV score:", grid.best_score_)

