"""
Delay Prediction Model
Uses machine learning to predict flight delays.
"""

import os
from typing import Dict, List, Optional

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DelayPredictor:
    """Machine learning model to predict flight delays."""

    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize the delay predictor.

        Args:
            model_type: 'xgboost', 'random_forest', 'ensemble', or 'nn'
            random_state: seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_columns: List[str] = []
        self.is_trained = False
        self.scaler: Optional[StandardScaler] = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling, including weather and crew signals."""
        df_ml = df.copy()

        # Time features
        df_ml['Scheduled_Time'] = pd.to_datetime(df_ml['Scheduled_Time'])
        df_ml['Hour'] = df_ml['Scheduled_Time'].dt.hour
        df_ml['Day_of_Week'] = df_ml['Scheduled_Time'].dt.dayofweek
        df_ml['Month'] = df_ml['Scheduled_Time'].dt.month
        df_ml['Is_Weekend'] = df_ml['Day_of_Week'].isin([5, 6]).astype(int)

        # Peak and congestion indicators
        df_ml['Is_Morning_Peak'] = df_ml['Hour'].isin([6, 7, 8, 9]).astype(int)
        df_ml['Is_Evening_Peak'] = df_ml['Hour'].isin([18, 19, 20, 21]).astype(int)
        df_ml['Is_Night'] = df_ml['Hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

        df_ml['Congestion_Factor'] = df_ml.get('Congestion_Factor', 1.0).fillna(1.0)

        peak_mapping = {'super_peak': 4, 'peak': 3, 'moderate': 2, 'low': 1}
        df_ml['Peak_Category_Numeric'] = df_ml.get('Peak_Category', '').map(peak_mapping).fillna(2)

        df_ml['Runway_Efficiency'] = df_ml.get('Runway_Efficiency', 1.0).fillna(1.0)
        df_ml['Runway_Capacity'] = df_ml.get('Runway_Capacity', 30).fillna(30)

        df_ml['Is_Congested_Airport'] = df_ml.get('Origin', '').isin(['BOM', 'DEL']).astype(int)

        if 'Runway_Capacity' in df_ml.columns:
            hourly_flights = df_ml.groupby(['Origin', 'Hour']).size().reset_index(name='Hourly_Flights')
            df_ml = df_ml.merge(hourly_flights, on=['Origin', 'Hour'], how='left')
            df_ml['Runway_Utilization'] = df_ml['Hourly_Flights'] / df_ml['Runway_Capacity']
            df_ml['Runway_Utilization'] = df_ml['Runway_Utilization'].fillna(0.5)
        else:
            df_ml['Runway_Utilization'] = 0.5

        # Weather and environmental signals
        weather_mapping = {
            'Clear': 0,
            'Partly Cloudy': 1,
            'Cloudy': 1,
            'Light Rain': 2,
            'Moderate Rain': 3,
            'Heavy Rain': 4,
            'Fog': 4,
            'Thunderstorm': 5
        }
        df_ml['Weather_Score'] = df_ml.get('Weather_Condition', '').map(weather_mapping).fillna(0)
        df_ml['Wind_Speed'] = df_ml.get('Wind_Speed', 0.0).fillna(0.0)
        df_ml['Visibility_km'] = df_ml.get('Visibility_km', 10.0).fillna(10.0)
        df_ml['Precipitation_mm'] = df_ml.get('Precipitation_mm', 0.0).fillna(0.0)

        # Airline reputation and operational reputation signals
        reputation_map = {
            'AI': 0.82, '6E': 0.78, 'UK': 0.70, 'SG': 0.72, 'G8': 0.68,
            'QP': 0.65, 'I5': 0.66, '9W': 0.70, 'VT': 0.74, 'IX': 0.69
        }
        df_ml['Airline_Reputation'] = df_ml['Airline'].map(reputation_map).fillna(0.65)

        # Crew and connection risk signals
        df_ml['Crew_Rest_Risk'] = (df_ml.get('Crew_Hours_Since_Rest', 12).fillna(12) < 10).astype(int)
        df_ml['Connecting_Count'] = df_ml.get('Connecting_Flights', '').fillna('').apply(
            lambda x: len([item for item in str(x).split(',') if item.strip()])
        )

        # Runway / flight type encoding
        df_ml = pd.concat([df_ml, pd.get_dummies(df_ml.get('Airline', pd.Series(dtype='object')), prefix='Airline')], axis=1)
        df_ml = pd.concat([df_ml, pd.get_dummies(df_ml.get('Aircraft_Type', pd.Series(dtype='object')), prefix='Aircraft')], axis=1)
        df_ml = pd.concat([df_ml, pd.get_dummies(df_ml.get('Runway', pd.Series(dtype='object')), prefix='Runway')], axis=1)

        major_destinations = ['BOM', 'DEL', 'BLR', 'MAA', 'CCU', 'HYD']
        for dest in major_destinations:
            df_ml[f'Dest_{dest}'] = (df_ml.get('Destination', '') == dest).astype(int)

        df_ml['Is_High_Impact_Int'] = df_ml.get('Is_High_Impact', False).astype(int)
        df_ml['Is_Large_Aircraft'] = (df_ml.get('Capacity', 0).fillna(0) > 250).astype(int)

        df_ml['Capacity_Category'] = pd.cut(
            df_ml.get('Capacity', 0).fillna(0),
            bins=[0, 150, 200, 250, 350],
            labels=['Small', 'Medium', 'Large', 'XLarge']
        )
        df_ml = pd.concat([df_ml, pd.get_dummies(df_ml['Capacity_Category'], prefix='Cap')], axis=1)

        if 'Hourly_Flight_Count' in df_ml.columns:
            df_ml['Is_Congested_Hour'] = (
                df_ml['Hourly_Flight_Count'] > df_ml['Hourly_Flight_Count'].quantile(0.75)
            ).astype(int)

        if 'Runway_Hourly_Count' in df_ml.columns:
            df_ml['Is_Runway_Congested'] = (
                df_ml['Runway_Hourly_Count'] > df_ml['Runway_Hourly_Count'].quantile(0.75)
            ).astype(int)

        return df_ml

    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for modeling."""
        features = [
            'Hour', 'Day_of_Week', 'Month', 'Is_Weekend',
            'Is_Morning_Peak', 'Is_Evening_Peak', 'Is_Night',
            'Capacity', 'Is_Large_Aircraft', 'Is_High_Impact_Int',
            'Congestion_Factor', 'Peak_Category_Numeric', 'Runway_Efficiency',
            'Runway_Capacity', 'Is_Congested_Airport', 'Runway_Utilization',
            'Weather_Score', 'Wind_Speed', 'Visibility_km', 'Precipitation_mm',
            'Airline_Reputation', 'Crew_Rest_Risk', 'Connecting_Count'
        ]

        features.extend([col for col in df.columns if col.startswith('Airline_')])
        features.extend([col for col in df.columns if col.startswith('Aircraft_')])
        features.extend([col for col in df.columns if col.startswith('Runway_')])
        features.extend([col for col in df.columns if col.startswith('Dest_')])
        features.extend([col for col in df.columns if col.startswith('Cap_')])

        for col in ['Hourly_Flight_Count', 'Runway_Hourly_Count', 'Is_Congested_Hour',
                    'Is_Runway_Congested', 'Hourly_Flights']:
            if col in df.columns:
                features.append(col)

        return [f for f in features if f in df.columns]

    def create_nn_model(self, input_dim: int) -> tf.keras.Model:
        """Create a simple neural network for delay prediction."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def build_ensemble(self) -> VotingRegressor:
        """Build an ensemble regressor from several base models."""
        estimators = [
            ('xgb', xgb.XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )),
            ('rf', RandomForestRegressor(
                n_estimators=120,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1
            )),
            ('hgb', HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ))
        ]
        return VotingRegressor(estimators=estimators, n_jobs=-1)

    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Evaluate the model using k-fold cross-validation."""
        if self.model is None:
            raise ValueError('Model must be initialized before cross-validation')

        scoring = {
            'neg_mean_absolute_error': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        results = cross_validate(self.model, X, y, cv=KFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                                 scoring=scoring, return_train_score=True, n_jobs=-1)

        return {
            'cv_mae_mean': -np.mean(results['test_neg_mean_absolute_error']),
            'cv_mae_std': np.std(results['test_neg_mean_absolute_error']),
            'cv_r2_mean': np.mean(results['test_r2'])
        }

    def tune_hyperparameters(self, df: pd.DataFrame, n_trials: int = 20) -> Dict[str, float]:
        """Tune hyperparameters for XGBoost using Optuna."""
        df_ml = self.prepare_features(df)
        self.feature_columns = self.select_features(df_ml)
        X = df_ml[self.feature_columns].fillna(0)
        y = df_ml['Delay_Minutes']

        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
            }

            model = xgb.XGBRegressor(**params, random_state=self.random_state, n_jobs=-1)
            scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores.append(mean_absolute_error(y_test, y_pred))
            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    def train(self,
              df: pd.DataFrame,
              target_column: str = 'Delay_Minutes',
              cv: int = 5,
              tune: bool = False) -> Dict[str, object]:
        """Train the delay prediction model and return evaluation metrics."""
        df_ml = self.prepare_features(df)
        self.feature_columns = self.select_features(df_ml)

        X = df_ml[self.feature_columns].fillna(0)
        y = df_ml[target_column]

        if tune and self.model_type == 'xgboost':
            best_params = self.tune_hyperparameters(df, n_trials=20)
            self.model = xgb.XGBRegressor(**best_params, random_state=self.random_state, n_jobs=-1)
        elif self.model_type == 'ensemble':
            self.model = self.build_ensemble()
        elif self.model_type == 'nn':
            self.scaler = StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            nn_model = self.create_nn_model(X.shape[1])
            history = nn_model.fit(X, y, validation_split=0.15, epochs=40, batch_size=32, verbose=0)
            self.model = nn_model
            self.is_trained = True
            y_pred_test = self.model.predict(X).ravel()
            metrics = {
                'train_mae': mean_absolute_error(y, y_pred_test),
                'train_r2': r2_score(y, y_pred_test),
                'feature_importance': {},
                'history': history.history
            }
            if cv > 1:
                metrics.update(self.cross_validate_model(X, y, cv=cv))
            return metrics
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=(y > 0)
        )

        if self.model_type == 'nn':
            pass
        else:
            self.model.fit(X_train, y_train)
            self.is_trained = True

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': self.get_feature_importance()
        }

        if cv > 1:
            metrics.update(self.cross_validate_model(X, y, cv=cv))

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict delay minutes for new data."""
        if not self.is_trained:
            raise ValueError('Model must be trained before making predictions')

        df_ml = self.prepare_features(df)
        X = df_ml[self.feature_columns].fillna(0)

        if self.model_type == 'nn':
            if self.scaler is not None:
                X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            return self.model.predict(X).ravel()

        return self.model.predict(X)

    def predict_delay_probability(self, df: pd.DataFrame, threshold: float = 15.0) -> np.ndarray:
        """Predict probability of delay exceeding a threshold."""
        delay_predictions = self.predict(df)
        return (delay_predictions > threshold).astype(float)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances when available."""
        if not self.is_trained or self.model is None:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_columns, importances))

        return {}

    def save_model(self, filepath: str):
        """Save trained model and metadata."""
        if not self.is_trained:
            raise ValueError('No trained model to save')

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'random_state': self.random_state,
            'scaler': self.scaler
        }, filepath)

    def load_model(self, filepath: str):
        """Load model metadata from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.model_type = data.get('model_type', self.model_type)
        self.random_state = data.get('random_state', self.random_state)
        self.scaler = data.get('scaler', None)
        self.is_trained = True

    def log_model_mlflow(self, X: pd.DataFrame, y: pd.Series, experiment_name: str = 'flight-delay-prediction') -> None:
        """Log the current model to MLflow."""
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_param('model_type', self.model_type)
            mlflow.log_param('feature_count', len(self.feature_columns))
            mlflow.log_param('random_state', self.random_state)
            if hasattr(self.model, 'get_params'):
                mlflow.log_params({k: v for k, v in self.model.get_params().items() if isinstance(v, (int, float, str, bool))})
            mlflow.sklearn.log_model(self.model, 'delay_model')


def create_risk_heatmap(df: pd.DataFrame, predictor: DelayPredictor) -> pd.DataFrame:
    """Create a risk heatmap for upcoming time slots."""
    delay_predictions = predictor.predict(df)
    delay_probabilities = predictor.predict_delay_probability(df)

    df_risk = df.copy()
    df_risk['Predicted_Delay'] = delay_predictions
    df_risk['Delay_Risk'] = delay_probabilities
    df_risk['Hour'] = pd.to_datetime(df_risk['Scheduled_Time']).dt.hour
    df_risk['Date'] = pd.to_datetime(df_risk['Scheduled_Time']).dt.date

    risk_summary = df_risk.groupby(['Date', 'Hour']).agg({
        'Delay_Risk': 'mean',
        'Predicted_Delay': 'mean',
        'Flight_ID': 'count'
    }).rename(columns={'Flight_ID': 'Flight_Count'}).reset_index()

    risk_summary['Risk_Level'] = pd.cut(
        risk_summary['Delay_Risk'],
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    return risk_summary


def plot_feature_importance(importance_dict: Dict[str, float], top_n: int = 15):
    """Plot feature importance."""
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_features:
        return

    features, importances = zip(*sorted_features)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances for Delay Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def main():
    """Train and evaluate delay prediction model."""
    df = pd.read_csv('data/flight_schedule_data.csv')

    predictor = DelayPredictor(model_type='ensemble')
    metrics = predictor.train(df, cv=5, tune=False)

    print('\n=== Model Performance ===')
    print(f"Test MAE: {metrics['test_mae']:.2f} minutes")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    print(f"CV MAE: {metrics.get('cv_mae_mean', 0):.2f}")
    print(f"CV R²: {metrics.get('cv_r2_mean', 0):.3f}")

    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/delay_predictor_ensemble.joblib')

    risk_df = create_risk_heatmap(df, predictor)
    risk_df.to_csv('data/delay_risk_heatmap.csv', index=False)
    
    print(f"\n=== Risk Analysis ===")
    print(risk_df['Risk_Level'].value_counts())
    
    # Feature importance
    importance = predictor.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n=== Top 10 Important Features ===")
    for feature, imp in top_features:
        print(f"{feature}: {imp:.3f}")

if __name__ == "__main__":
    main()
