import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import pickle

class DecisionTree_SkinLeasion_Classifier:
    """Decision Tree classifier for skin cancer detection."""

    def __init__(self,
                 criterion='gini',
                 max_depth=5,
                 min_samples_split=2,
                 random_state=42):
        """
        Initialize Decision Tree classifier.

        Args:
            criterion: Split quality measure ('gini' or 'entropy')
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            random_state: Random seed
        """

        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_ids = None
        self.test_img_ids = None
        self.feature_cols = None

    def load_and_prepare(self, features_path,
                         test_size=0.2,
                         random_state=42):

        df = pd.read_csv(features_path)

        exclude_cols = {'ID', 'Image_ID', 'Diagnosis', 'Cancer', 'D_hue', 'D_saturation', 'D_brightness', 'Hair_mean'}
        self.feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
        ]

        X = df[self.feature_cols].values
        y = df['Cancer'].values
        ids = df['ID'].values
        img_ids = df['Image_ID'].values

        train_idx, test_idx, self.X_train, self.X_test, \
        self.y_train, self.y_test = train_test_split(
            np.arange(len(X)),
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        self.test_ids = ids[test_idx]
        self.test_img_ids = img_ids[test_idx]

        print(f"Data loaded: {len(self.X_train)} training, "
              f"{len(self.X_test)} testing samples")

        print(f"Features used: {self.feature_cols}")

    def train(self):
        """Train Decision Tree classifier."""
        self.model.fit(self.X_train, self.y_train)
        print("Model trained")

    def evaluate(self):
        """Evaluate model performance."""

        pred = self.model.predict(self.X_test)

        proba = self.model.predict_proba(self.X_test)

        classes = self.model.classes_
        cancerous_idx = list(classes).index('Cancerous')
        prob_cancerous = proba[:, cancerous_idx]

        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)

        print(f"Accuracy:  "
              f"{accuracy_score(self.y_test, pred):.4f}")

        print(f"Precision: "
              f"{precision_score(self.y_test,
                                 pred,
                                 pos_label='Cancerous',
                                 zero_division=0):.4f}")

        print(f"Recall:    "
              f"{recall_score(self.y_test,
                              pred,
                              pos_label='Cancerous',
                              zero_division=0):.4f}")

        print(f"F1-Score:  "
              f"{f1_score(self.y_test,
                          pred,
                          pos_label='Cancerous',
                          zero_division=0):.4f}")

        print(f"ROC AUC:   "
              f"{roc_auc_score(self.y_test, prob_cancerous):.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, pred))

        print("\n" + classification_report(self.y_test, pred))
        print("="*50)

    def predict(self, X=None):
        """
        Make predictions.
        """
        return self.model.predict(
            X if X is not None else self.X_test
        )

    def save_model(self, path):
        """
        Save trained model.
        """
        pickle.dump(self.model, open(path, 'wb'))
        # print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load trained model.
        """
        self.model = pickle.load(open(path, 'rb'))
        # print(f"Model loaded from {path}")

    def save_predictions(self, output_path):
        """
        Save predictions and probabilities.
        """

        pred = self.predict()
        proba = self.model.predict_proba(self.X_test)

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

        # print(f"Predictions saved to {output_path}")