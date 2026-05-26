import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import pickle
from pathlib import Path


class KNNSkinCancerClassifier:
    """KNN classifier for skin cancer detection using all available features."""
    
    def __init__(self, n_neighbors=33, weights='distance', metric='euclidean'):
        """
        Initialize KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors to use.
            weights: Weight function, we use distance instead of uniform.
            metric: Distance metric, we use euclidean instead of manhattan or minkowski.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_ids = None
        self.test_img_ids = None
        self.feature_cols = None
        
    def load_and_prepare(self, features_path, test_size=0.2, random_state=42):
        """
        Load features from CSV and split into training/testing sets.
        
        Args:
            features_path: Path to features.csv file.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
        """
        df = pd.read_csv(features_path)
        
        # Automatically detect feature columns (exclude ID, Image_ID, Diagnosis, Cancer)
        exclude_cols = {'ID', 'Image_ID', 'Diagnosis', 'Cancer'}
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_cols].values
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
        print(f"Features used: {self.feature_cols}")
    
    def train(self):
        """Train the KNN classifier on the training data."""
        self.model.fit(self.X_train, self.y_train)
        print("Model trained")
    
    def evaluate(self, threshold=0.45):
        """Evaluate model performance on test set and display metrics."""
        proba = self.model.predict_proba(self.X_test)
        
        # Get probabilities for Cancerous class
        cancerous_idx = list(self.model.classes_).index('Cancerous')
        prob_cancerous = proba[:, cancerous_idx]
        
        # Apply threshold
        pred = np.array(['Cancerous' if p >= threshold else 'Benign' for p in prob_cancerous])
        
        # Convert labels to binary (Cancerous=1, Benign=0) for AUC calculation
        y_test_binary = np.array([1 if label == 'Cancerous' else 0 for label in self.y_test])
        auc = roc_auc_score(y_test_binary, prob_cancerous)
        
        print("\n" + "="*50)
        print(f"RESULTS (Threshold: {threshold})")
        print("="*50)
        print(f"Accuracy:  {accuracy_score(self.y_test, pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, pred, pos_label='Cancerous', zero_division=0):.4f}")
        print(f"Recall:    {recall_score(self.y_test, pred, pos_label='Cancerous', zero_division=0):.4f}")
        print(f"F1-Score:  {f1_score(self.y_test, pred, pos_label='Cancerous', zero_division=0):.4f}")
        print(f"AUC:       {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, pred))
        print("\n" + classification_report(self.y_test, pred))
        print("="*50)
    
    def save_model(self, path):
        """
        Save trained model to disk.
        
        Args:
            path: File path to save model.
        """
        pickle.dump(self.model, open(path, 'wb'))
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load trained model from disk.
        
        Args:
            path: File path to load model.
        """
        self.model = pickle.load(open(path, 'rb'))
        print(f"Model loaded from {path}")
    
    def predict(self, X=None, threshold=0.45):
        """
        Make predictions on test set or provided features.
        
        Args:
            X: Feature array to predict on. If None, uses test set.
            threshold: Probability threshold for Cancerous classification (default 0.45).
            
        Returns:
            Array of predictions (Cancerous or Benign).
        """
        X_to_predict = X if X is not None else self.X_test
        proba = self.model.predict_proba(X_to_predict)
        cancerous_idx = list(self.model.classes_).index('Cancerous')
        prob_cancerous = proba[:, cancerous_idx]
        return np.array(['Cancerous' if p >= threshold else 'Benign' for p in prob_cancerous])
    

    
    def save_predictions(self, output_path):
        """
        Save predictions to CSV file with probabilities and actual labels.
        
        Args:
            output_path: File path to save predictions CSV.
        """
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
