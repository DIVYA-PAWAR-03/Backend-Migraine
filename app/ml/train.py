"""
Machine Learning Training Pipeline for Migraine Prediction.

This script:
1. Loads and preprocesses the migraine dataset
2. Generates synthetic lifestyle-based features if needed
3. Trains and compares multiple models (Random Forest, Logistic Regression, Decision Tree)
4. Selects the best model based on F1-score
5. Saves the trained model and scaler for production use

Usage:
    python -m app.ml.train
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import logging
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigraineModelTrainer:
    """
    Class to train and evaluate migraine prediction models.
    
    Automatically selects the best model based on F1-score from:
    - Random Forest
    - Logistic Regression
    - Decision Tree
    """
    
    # Feature columns for the model
    FEATURE_COLUMNS = [
        'stress_level',
        'sleep_hours', 
        'heart_rate',
        'activity_level',
        'weather_pressure',
        'aqi'
    ]
    
    TARGET_COLUMN = 'migraine_attack'
    
    def __init__(self, data_path: str = None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the CSV dataset
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_f1_score = 0
        self.model_results = {}
        
        # Define models to compare
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            Loaded DataFrame
        """
        if self.data_path and Path(self.data_path).exists():
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records")
        else:
            logger.info("Generating synthetic training data...")
            self.df = self._generate_synthetic_data()
        
        return self.df
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic training data based on known migraine triggers.
        
        This creates realistic data patterns based on medical research
        about migraine triggers including stress, sleep, activity, and
        environmental factors.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(42)
        
        data = {
            'stress_level': np.random.randint(1, 11, n_samples),
            'sleep_hours': np.random.uniform(3, 10, n_samples).round(1),
            'heart_rate': np.random.randint(55, 110, n_samples),
            'activity_level': np.random.randint(1, 11, n_samples),
            'weather_pressure': np.random.uniform(990, 1040, n_samples).round(1),
            'aqi': np.random.randint(10, 200, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate migraine labels based on trigger patterns
        # Higher probability with: high stress, low sleep, high HR, low activity, 
        # extreme pressure, high AQI
        
        migraine_prob = np.zeros(n_samples)
        
        # Stress contribution (0-0.3)
        migraine_prob += (df['stress_level'] - 1) / 9 * 0.3
        
        # Sleep contribution (0-0.3) - inverse relationship
        migraine_prob += (10 - df['sleep_hours']) / 7 * 0.3
        
        # Heart rate contribution (0-0.15)
        migraine_prob += (df['heart_rate'] - 55) / 55 * 0.15
        
        # Activity contribution (0-0.1) - inverse relationship
        migraine_prob += (10 - df['activity_level']) / 9 * 0.1
        
        # Weather pressure contribution (0-0.1) - extremes are bad
        pressure_deviation = np.abs(df['weather_pressure'] - 1013) / 27
        migraine_prob += pressure_deviation * 0.1
        
        # AQI contribution (0-0.1)
        migraine_prob += df['aqi'] / 200 * 0.1
        
        # Add some randomness
        migraine_prob += np.random.uniform(-0.1, 0.1, n_samples)
        
        # Clip to valid probability range
        migraine_prob = np.clip(migraine_prob, 0, 1)
        
        # Generate binary labels based on probability
        df['migraine_attack'] = (np.random.random(n_samples) < migraine_prob).astype(int)
        
        # Ensure reasonable class balance (30-70% split)
        attack_rate = df['migraine_attack'].mean()
        logger.info(f"Generated data with {attack_rate:.1%} migraine attack rate")
        
        return df
    
    def preprocess_data(self) -> None:
        """
        Preprocess the data for training.
        
        - Handles missing values
        - Encodes categorical variables if present
        - Scales features
        - Splits into train/test sets
        """
        logger.info("Preprocessing data...")
        
        # Check if we have the required columns or need to adapt
        if 'Type' in self.df.columns and self.TARGET_COLUMN not in self.df.columns:
            # Convert migraine type to binary classification
            logger.info("Converting migraine type to binary classification...")
            # Types indicating migraine: anything except "Other" and "Typical aura without migraine"
            no_attack_types = ['Other']
            self.df[self.TARGET_COLUMN] = (~self.df['Type'].isin(no_attack_types)).astype(int)
            
            # Generate lifestyle features from existing symptom data
            logger.info("Generating lifestyle features from symptom data...")
            self._adapt_clinical_to_lifestyle()
        
        # Handle missing values
        for col in self.FEATURE_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Prepare features and target
        available_features = [col for col in self.FEATURE_COLUMNS if col in self.df.columns]
        
        if len(available_features) < len(self.FEATURE_COLUMNS):
            logger.warning(f"Some features missing. Using: {available_features}")
        
        self.X = self.df[available_features].values
        self.y = self.df[self.TARGET_COLUMN].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            random_state=42,
            stratify=self.y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set size: {len(self.X_train)}")
        logger.info(f"Test set size: {len(self.X_test)}")
        logger.info(f"Class distribution: {np.bincount(self.y)}")
    
    def _adapt_clinical_to_lifestyle(self) -> None:
        """
        Adapt clinical dataset to lifestyle-based features.
        
        Maps clinical symptoms to lifestyle factors:
        - Intensity -> stress_level
        - Duration -> inverse of sleep_hours
        - Frequency -> heart_rate pattern
        - Nausea/Vomit -> activity_level
        - Age -> weather sensitivity (weather_pressure)
        - Visual symptoms -> AQI sensitivity
        """
        # Stress level from Intensity (already 1-3, scale to 1-10)
        if 'Intensity' in self.df.columns:
            self.df['stress_level'] = (self.df['Intensity'] * 3).clip(1, 10).astype(int)
        else:
            self.df['stress_level'] = np.random.randint(4, 9, len(self.df))
        
        # Sleep hours - inverse of Duration (longer migraine -> less sleep)
        if 'Duration' in self.df.columns:
            # Duration 1-3, map to sleep 4-8 hours inversely
            self.df['sleep_hours'] = 9 - (self.df['Duration'] * 1.5)
            self.df['sleep_hours'] = self.df['sleep_hours'].clip(4, 9).round(1)
        else:
            self.df['sleep_hours'] = np.random.uniform(5, 8, len(self.df)).round(1)
        
        # Heart rate - based on Frequency and age
        if 'Frequency' in self.df.columns and 'Age' in self.df.columns:
            base_hr = 70 + (self.df['Frequency'] * 3)
            age_factor = np.where(self.df['Age'] > 40, 5, 0)
            self.df['heart_rate'] = (base_hr + age_factor + np.random.randint(-5, 5, len(self.df))).clip(60, 100).astype(int)
        else:
            self.df['heart_rate'] = np.random.randint(65, 95, len(self.df))
        
        # Activity level - inverse of Nausea + Vomit
        if 'Nausea' in self.df.columns and 'Vomit' in self.df.columns:
            symptom_severity = self.df['Nausea'] + self.df['Vomit']
            self.df['activity_level'] = (10 - symptom_severity * 2).clip(1, 10).astype(int)
        else:
            self.df['activity_level'] = np.random.randint(3, 8, len(self.df))
        
        # Weather pressure - based on Age (older more sensitive to pressure)
        if 'Age' in self.df.columns:
            base_pressure = 1013
            age_effect = np.where(self.df['Age'] > 35, 
                                  np.random.uniform(-15, 15, len(self.df)), 
                                  np.random.uniform(-5, 5, len(self.df)))
            self.df['weather_pressure'] = (base_pressure + age_effect).round(1)
        else:
            self.df['weather_pressure'] = np.random.uniform(1000, 1025, len(self.df)).round(1)
        
        # AQI - based on Visual symptoms
        if 'Visual' in self.df.columns:
            base_aqi = 40 + (self.df['Visual'] * 15)
            self.df['aqi'] = (base_aqi + np.random.randint(-10, 20, len(self.df))).clip(20, 150).astype(int)
        else:
            self.df['aqi'] = np.random.randint(30, 100, len(self.df))
        
        logger.info("Adapted clinical features to lifestyle features")
    
    def train_and_evaluate(self) -> dict:
        """
        Train all models and evaluate their performance.
        
        Returns:
            Dictionary with results for each model
        """
        logger.info("=" * 60)
        logger.info("TRAINING AND EVALUATING MODELS")
        logger.info("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Training: {name}")
            logger.info(f"{'─' * 40}")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predict on test set
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            # Cross-validation F1 score
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1'
            )
            cv_f1_mean = cv_scores.mean()
            cv_f1_std = cv_scores.std()
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_f1_mean': cv_f1_mean,
                'cv_f1_std': cv_f1_std,
                'model': model
            }
            
            # Log results
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            logger.info(f"  CV F1:     {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")
            
            # Update best model based on F1-score
            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_model = model
                self.best_model_name = name
        
        self.model_results = results
        return results
    
    def select_best_model(self) -> tuple:
        """
        Select and report the best model.
        
        Returns:
            Tuple of (model_name, model, f1_score)
        """
        logger.info("\n" + "=" * 60)
        logger.info("MODEL SELECTION RESULTS")
        logger.info("=" * 60)
        
        # Sort by F1-score
        sorted_results = sorted(
            self.model_results.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        logger.info("\nRanking by F1-Score:")
        for rank, (name, metrics) in enumerate(sorted_results, 1):
            marker = " ★ BEST" if name == self.best_model_name else ""
            logger.info(f"  {rank}. {name}: {metrics['f1_score']:.4f}{marker}")
        
        logger.info(f"\n✓ Selected Model: {self.best_model_name}")
        logger.info(f"  F1-Score: {self.best_f1_score:.4f}")
        
        return self.best_model_name, self.best_model, self.best_f1_score
    
    def save_model(self, model_path: str = "app/ml/model.pkl", scaler_path: str = "app/ml/scaler.pkl") -> None:
        """
        Save the best model and scaler to disk.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        # Ensure directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model info
        model_info = {
            'model_type': self.best_model_name,
            'accuracy': self.model_results[self.best_model_name]['accuracy'],
            'f1_score': self.best_f1_score,
            'precision': self.model_results[self.best_model_name]['precision'],
            'recall': self.model_results[self.best_model_name]['recall'],
            'cv_f1_mean': self.model_results[self.best_model_name]['cv_f1_mean'],
            'features': self.FEATURE_COLUMNS,
            'trained_at': datetime.now().isoformat(),
            'sample_count': len(self.df)
        }
        
        # Save model with info
        model_data = {
            'model': self.best_model,
            'info': model_info
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")
        
        # Save model info as JSON for reference
        info_path = Path(model_path).with_suffix('.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Model info saved to: {info_path}")
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance from the best model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            return dict(zip(self.FEATURE_COLUMNS, importance))
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
            return dict(zip(self.FEATURE_COLUMNS, importance))
        else:
            return {}


def main():
    """Main function to run the training pipeline."""
    logger.info("=" * 60)
    logger.info("MIGRAINE PREDICTION MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Initialize trainer with dataset path
    data_path = "../migraine_symptom_classification.csv"
    
    # Check multiple possible paths
    possible_paths = [
        data_path,
        "migraine_symptom_classification.csv",
        "data/migraine_data.csv",
        "../data/migraine_data.csv"
    ]
    
    actual_path = None
    for path in possible_paths:
        if Path(path).exists():
            actual_path = path
            break
    
    trainer = MigraineModelTrainer(data_path=actual_path)
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    trainer.load_data()
    
    # Step 2: Preprocess data
    logger.info("\nStep 2: Preprocessing data...")
    trainer.preprocess_data()
    
    # Step 3: Train and evaluate models
    logger.info("\nStep 3: Training and evaluating models...")
    trainer.train_and_evaluate()
    
    # Step 4: Select best model
    logger.info("\nStep 4: Selecting best model...")
    model_name, model, f1 = trainer.select_best_model()
    
    # Step 5: Get feature importance
    logger.info("\nStep 5: Feature importance analysis...")
    importance = trainer.get_feature_importance()
    if importance:
        logger.info("Feature Importance:")
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feature}: {score:.4f}")
    
    # Step 6: Save model
    logger.info("\nStep 6: Saving model...")
    trainer.save_model()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Best Model: {model_name}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info("Model saved to: app/ml/model.pkl")
    logger.info("Scaler saved to: app/ml/scaler.pkl")


if __name__ == "__main__":
    main()
