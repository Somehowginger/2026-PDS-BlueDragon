import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
from pathlib import Path


class KNNSkinCancerClassifier:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_ids = None
        self.test_img_ids = None
        
    def load_and_prepare(self, features_path, test_size=0.2, random_state=42):
        df = pd.read_csv(features_path)
        
        # Extract features and target
        feature_cols = ['A_asymmetry', 'B_compactness', 'C_hue', 'C_saturation', 'C_brightness']
        X = df[feature_cols].values
        y = df['Cancer'].values
        ids = df['ID'].values
        img_ids = df['Image_ID'].values
        
        # Split data and keep track of test indices
        train_idx, test_idx, self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            np.arange(len(X)), X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Store test patient IDs and image IDs
        self.test_ids = ids[test_idx]
        self.test_img_ids = img_ids[test_idx]
        
        print(f"Data loaded: {len(self.X_train)} training, {len(self.X_test)} testing samples")
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained")
    
    def evaluate(self):
        pred = self.model.predict(self.X_test)
        
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy_score(self.y_test, pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, pred, pos_label='Cancerous', zero_division=0):.4f}")
        print(f"Recall:    {recall_score(self.y_test, pred, pos_label='Cancerous', zero_division=0):.4f}")
        print(f"F1-Score:  {f1_score(self.y_test, pred, pos_label='Cancerous', zero_division=0):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, pred))
        print("\n" + classification_report(self.y_test, pred))
        print("="*50)
    
    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.model, open(path, 'wb'))
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        self.model = pickle.load(open(path, 'rb'))
        print(f"Model loaded from {path}")
    
    def predict(self, X=None):
        return self.model.predict(X if X is not None else self.X_test)
    
    def save_predictions(self, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pred = self.predict()
        proba = self.model.predict_proba(self.X_test)
        
        # Get class labels and their corresponding probabilities
        classes = self.model.classes_
        cancerous_idx = list(classes).index('Cancerous')
        prob_cancerous = proba[:, cancerous_idx]
        
        df = pd.DataFrame({
            'Patient_ID': self.test_ids,
            'Image_ID': self.test_img_ids,
            'Prediction': pred,
            'Probability': prob_cancerous,
            'Actual': self.y_test
        })
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
