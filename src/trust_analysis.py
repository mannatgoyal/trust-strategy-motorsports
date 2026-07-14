import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class TrustAnalyzer:
    """
    Random Forest Regressor to analyze trust dynamics and feature importance.
    Uses windowed historical lap times, track positions, and trust scores.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_names = ['LapTime', 'Position', 'Trust']
        self.test_score = 0.0  # R-squared score on test set
        self.is_trained = False
        
    def create_features(self, data, window=5):
        """
        Creates rolling windowed features from the F1 dataset.
        For window=5, creates a feature vector of length 15 (5 laps * 3 features).
        """
        X, y = [], []
        
        if len(data) <= window:
            return np.array([]), np.array([])
            
        for i in range(window, len(data)):
            features = []
            for w in range(window):
                for col in self.feature_names:
                    features.append(data.iloc[i-window+w][col])
            X.append(features)
            y.append(data.iloc[i]['Trust'])
            
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """
        Trains the model using a train/test split.
        Saves the performance metrics and fits on the full dataset afterwards.
        """
        if len(X) <= 5:
            self.test_score = 0.0
            return
            
        # 1. Train-test split (80% training, 20% validation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        # 2. Fit and scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
        # 3. Evaluate model accuracy on unseen validation data
        if len(X_test) > 0:
            X_test_scaled = self.scaler.transform(X_test)
            predictions = self.model.predict(X_test_scaled)
            self.test_score = r2_score(y_test, predictions)
        else:
            self.test_score = 0.0
            
        # 4. Refit on entire dataset for final model inference/feature importances
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
            
    def feature_importance(self):
        """
        Retrieves feature importance weights if the model is trained.
        """
        if self.is_trained and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
