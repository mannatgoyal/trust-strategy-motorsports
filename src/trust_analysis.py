import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from typing import Dict, Tuple, Any

class StrategyConfidenceEstimator:
    """
    Performance and Strategy Confidence Estimator.
    Combines pace consistency, degradation stability, prediction uncertainty,
    fuel consistency, and anomaly metrics to estimate overall strategy confidence.
    """
    def __init__(self):
        self.is_trained = False
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.test_score_rf = 0.0
        self.test_score_gbm = 0.0
        self.metrics = {}

    def calculate_confidence(
        self,
        pace_consistency: float,
        degradation_stability: float,
        prediction_certainty: float,
        fuel_consistency: float,
        anomaly_score: float
    ) -> float:
        """
        Computes the weighted Performance Confidence score.
        Formula:
          Conf = 0.25*Pace + 0.20*Deg + 0.20*Pred + 0.15*Fuel + 0.20*(1 - Anomaly)
        """
        score = (
            0.25 * pace_consistency +
            0.20 * degradation_stability +
            0.20 * prediction_certainty +
            0.15 * fuel_consistency +
            0.20 * (1.0 - anomaly_score)
        )
        return float(np.clip(score, 0.0, 1.0))

    def create_features(self, data: pd.DataFrame, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds feature matrices from telemetry parameters.
        Includes Sector splits, speeds, tyre compound indicators, gaps, and ERS telemetry.
        """
        df = data.copy()
        
        # Categorise Tyre Compounds (one-hot encoding)
        df['IsSoft'] = df['TyreCompound'].apply(lambda x: 1.0 if str(x).lower() == 'soft' else 0.0)
        df['IsHard'] = df['TyreCompound'].apply(lambda x: 1.0 if str(x).lower() == 'hard' else 0.0)
        
        # Calculate Rolling degradation slope
        df['PaceSlope'] = df['LapTime'].diff().rolling(window=3).mean().fillna(0.0)
        
        # Expanded feature set
        feature_cols = [
            'Sector1', 'Sector2', 'Sector3',
            'TopSpeed', 'AvgSpeed',
            'Throttle', 'Brake', 'Steering',
            'RPM', 'DRS', 'TyreAge',
            'TrackTemp', 'AmbientTemp', 'RainProbability',
            'GapAhead', 'GapBehind', 'Position',
            'IsSoft', 'IsHard', 'PaceSlope'
        ]
        
        # Make sure columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X_list = []
        y_list = []
        
        # Build sliding windows
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i][available_cols].values.flatten()
            target = df.iloc[i]['LapTime']
            
            X_list.append(window_data)
            y_list.append(target)
            
        return np.array(X_list), np.array(y_list)

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Trains and compares Random Forest and Gradient Boosting Regressors.
        Reports validation MAE, RMSE, and R2 coefficients.
        """
        if len(X) < 10:
            return {'status': 'Insufficient data'}
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Random Forest training
        self.rf_model.fit(X_train, y_train)
        y_pred_rf = self.rf_model.predict(X_test)
        
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        rmse_rf = root_mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        cv_scores_rf = cross_val_score(self.rf_model, X, y, cv=3)
        
        # 2. Gradient Boosting training
        self.gbm_model.fit(X_train, y_train)
        y_pred_gbm = self.gbm_model.predict(X_test)
        
        mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
        rmse_gbm = root_mean_squared_error(y_test, y_pred_gbm)
        r2_gbm = r2_score(y_test, y_pred_gbm)
        cv_scores_gbm = cross_val_score(self.gbm_model, X, y, cv=3)
        
        self.is_trained = True
        self.metrics = {
            'RF': {
                'MAE': mae_rf,
                'RMSE': rmse_rf,
                'R2': r2_rf,
                'CV_Mean': cv_scores_rf.mean()
            },
            'GBM': {
                'MAE': mae_gbm,
                'RMSE': rmse_gbm,
                'R2': r2_gbm,
                'CV_Mean': cv_scores_gbm.mean()
            }
        }
        return self.metrics

    def get_feature_importances(self, feature_names: list) -> np.ndarray:
        """
        Returns feature importances from the trained Random Forest model.
        """
        if not self.is_trained:
            return np.zeros(len(feature_names))
        return self.rf_model.feature_importances_
