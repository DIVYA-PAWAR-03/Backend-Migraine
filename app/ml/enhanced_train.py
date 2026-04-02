"""
Enhanced Machine Learning Training Pipeline for Migraine Prediction.

This script provides comprehensive migraine classification using:
1. Symptom-based classification (migraine type prediction)
2. Lifestyle-based risk prediction (trigger analysis)
3. Combined multi-model approach for accurate predictions

Migraine Types Classification:
- Migraine without aura
- Typical aura with migraine
- Basilar-type aura
- Sporadic hemiplegic migraine
- Familial hemiplegic migraine
- Typical aura without migraine
- Other

Usage:
    python -m app.ml.enhanced_train
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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


class EnhancedMigraineModelTrainer:
    """
    Enhanced trainer for migraine prediction with symptom-based classification.
    
    Trains two models:
    1. Symptom Classifier - Predicts migraine type from symptoms
    2. Risk Predictor - Predicts migraine risk from lifestyle factors
    """
    
    # Symptom features from the dataset
    SYMPTOM_FEATURES = [
        'Age', 'Duration', 'Frequency', 'Location', 'Character', 
        'Intensity', 'Nausea', 'Vomit', 'Phonophobia', 'Photophobia',
        'Visual', 'Sensory', 'Dysphasia', 'Dysarthria', 'Vertigo',
        'Tinnitus', 'Hypoacusis', 'Diplopia', 'Defect', 'Ataxia',
        'Conscience', 'Paresthesia', 'DPF'
    ]
    
    # Lifestyle features for risk prediction
    LIFESTYLE_FEATURES = [
        'stress_level', 'sleep_hours', 'heart_rate',
        'activity_level', 'weather_pressure', 'aqi'
    ]
    
    # Migraine types for classification
    MIGRAINE_TYPES = [
        'Migraine without aura',
        'Typical aura with migraine', 
        'Basilar-type aura',
        'Sporadic hemiplegic migraine',
        'Familial hemiplegic migraine',
        'Typical aura without migraine',
        'Other'
    ]
    
    def __init__(self, symptom_data_path: str = None, lifestyle_data_path: str = None):
        """Initialize the enhanced trainer."""
        self.symptom_data_path = symptom_data_path
        self.lifestyle_data_path = lifestyle_data_path
        
        # Symptom classifier components
        self.symptom_df = None
        self.symptom_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.symptom_model = None
        self.symptom_model_results = {}
        
        # Risk predictor components
        self.lifestyle_df = None
        self.lifestyle_scaler = StandardScaler()
        self.risk_model = None
        self.risk_model_results = {}
        
        # Feature importance mapping for suggestions
        self.symptom_importance = {}
        self.trigger_importance = {}
        
    def load_symptom_data(self) -> pd.DataFrame:
        """Load and validate the symptom classification dataset."""
        if self.symptom_data_path and Path(self.symptom_data_path).exists():
            logger.info(f"Loading symptom data from {self.symptom_data_path}")
            self.symptom_df = pd.read_csv(self.symptom_data_path)
            logger.info(f"Loaded {len(self.symptom_df)} symptom records")
            
            # Show class distribution
            if 'Type' in self.symptom_df.columns:
                logger.info(f"Migraine types distribution:\n{self.symptom_df['Type'].value_counts()}")
        else:
            logger.error("Symptom data file not found!")
            self.symptom_df = None
            
        return self.symptom_df
    
    def preprocess_symptom_data(self):
        """Preprocess symptom data for classification."""
        if self.symptom_df is None:
            logger.error("No symptom data loaded!")
            return
        
        logger.info("Preprocessing symptom data...")
        
        # Handle missing values
        for col in self.SYMPTOM_FEATURES:
            if col in self.symptom_df.columns:
                self.symptom_df[col] = self.symptom_df[col].fillna(0)
        
        # Get available features
        available_features = [f for f in self.SYMPTOM_FEATURES if f in self.symptom_df.columns]
        logger.info(f"Using {len(available_features)} symptom features")
        
        # Prepare X and y
        X = self.symptom_df[available_features].values
        y = self.label_encoder.fit_transform(self.symptom_df['Type'].values)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.symptom_scaler.fit_transform(X_train)
        X_test_scaled = self.symptom_scaler.transform(X_test)
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, available_features
    
    def train_symptom_classifier(self):
        """Train the symptom-based migraine type classifier."""
        logger.info("=" * 60)
        logger.info("TRAINING SYMPTOM-BASED MIGRAINE CLASSIFIER")
        logger.info("=" * 60)
        
        X_train, X_test, y_train, y_test, features = self.preprocess_symptom_data()
        
        # Models to compare
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
        
        best_model = None
        best_model_name = None
        best_f1 = 0
        
        for name, model in models.items():
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Training: {name}")
            logger.info(f"{'─' * 40}")
            
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            self.symptom_model_results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'model': model
            }
            
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name
        
        self.symptom_model = best_model
        logger.info(f"\n✓ Best Symptom Classifier: {best_model_name} (F1: {best_f1:.4f})")
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.symptom_importance = dict(zip(features, best_model.feature_importances_))
            logger.info("\nTop 10 Important Symptom Features:")
            for feat, imp in sorted(self.symptom_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {feat}: {imp:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        y_pred_final = best_model.predict(X_test)
        report = classification_report(
            y_test, y_pred_final, 
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        logger.info(f"\n{report}")
        
        return best_model_name, best_f1
    
    def generate_lifestyle_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate synthetic lifestyle data for risk prediction training."""
        logger.info(f"Generating {n_samples} synthetic lifestyle samples...")
        
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
        
        # Calculate migraine probability based on research-backed trigger weights
        migraine_prob = np.zeros(n_samples)
        
        # Stress is a major trigger (weight: 0.25)
        migraine_prob += (df['stress_level'] - 1) / 9 * 0.25
        
        # Sleep deprivation is critical (weight: 0.25)
        migraine_prob += np.clip((7 - df['sleep_hours']) / 4, 0, 1) * 0.25
        
        # Heart rate elevation (weight: 0.15)
        migraine_prob += np.clip((df['heart_rate'] - 70) / 40, 0, 1) * 0.15
        
        # Low activity (weight: 0.10)
        migraine_prob += (10 - df['activity_level']) / 9 * 0.10
        
        # Weather pressure extremes (weight: 0.15)
        pressure_deviation = np.abs(df['weather_pressure'] - 1013) / 30
        migraine_prob += np.clip(pressure_deviation, 0, 1) * 0.15
        
        # Poor air quality (weight: 0.10)
        migraine_prob += np.clip(df['aqi'] / 150, 0, 1) * 0.10
        
        # Add realistic noise
        migraine_prob += np.random.uniform(-0.08, 0.08, n_samples)
        migraine_prob = np.clip(migraine_prob, 0, 1)
        
        # Generate labels with threshold
        df['migraine_attack'] = (migraine_prob > 0.45).astype(int)
        df['migraine_probability'] = migraine_prob
        
        attack_rate = df['migraine_attack'].mean()
        logger.info(f"Generated data with {attack_rate:.1%} migraine attack rate")
        
        self.lifestyle_df = df
        return df
    
    def train_risk_predictor(self):
        """Train the lifestyle-based risk prediction model."""
        logger.info("=" * 60)
        logger.info("TRAINING LIFESTYLE-BASED RISK PREDICTOR")
        logger.info("=" * 60)
        
        if self.lifestyle_df is None:
            self.generate_lifestyle_data()
        
        # Prepare data
        X = self.lifestyle_df[self.LIFESTYLE_FEATURES].values
        y = self.lifestyle_df['migraine_attack'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.lifestyle_scaler.fit_transform(X_train)
        X_test_scaled = self.lifestyle_scaler.transform(X_test)
        
        # Models to compare
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        }
        
        best_model = None
        best_model_name = None
        best_f1 = 0
        
        for name, model in models.items():
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Training: {name}")
            logger.info(f"{'─' * 40}")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1'
            )
            
            self.risk_model_results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'model': model
            }
            
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            logger.info(f"  CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name
        
        self.risk_model = best_model
        logger.info(f"\n✓ Best Risk Predictor: {best_model_name} (F1: {best_f1:.4f})")
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.trigger_importance = dict(zip(self.LIFESTYLE_FEATURES, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            self.trigger_importance = dict(zip(self.LIFESTYLE_FEATURES, np.abs(best_model.coef_[0])))
        
        if self.trigger_importance:
            logger.info("\nTrigger Feature Importance:")
            for feat, imp in sorted(self.trigger_importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {feat}: {imp:.4f}")
        
        return best_model_name, best_f1
    
    def save_models(self, base_path: str = "app/ml"):
        """Save all trained models and related artifacts."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Save symptom classifier
        if self.symptom_model:
            symptom_data = {
                'model': self.symptom_model,
                'scaler': self.symptom_scaler,
                'label_encoder': self.label_encoder,
                'features': self.SYMPTOM_FEATURES,
                'classes': list(self.label_encoder.classes_),
                'feature_importance': self.symptom_importance,
                'info': {
                    'model_type': type(self.symptom_model).__name__,
                    'accuracy': self.symptom_model_results.get(
                        type(self.symptom_model).__name__.replace('Classifier', ''), {}
                    ).get('accuracy', 0),
                    'f1_score': self.symptom_model_results.get(
                        type(self.symptom_model).__name__.replace('Classifier', ''), {}
                    ).get('f1_score', 0),
                    'trained_at': timestamp,
                    'task': 'symptom_classification'
                }
            }
            
            symptom_path = base_path / "symptom_classifier.pkl"
            joblib.dump(symptom_data, symptom_path)
            logger.info(f"Symptom classifier saved to: {symptom_path}")
        
        # Save risk predictor
        if self.risk_model:
            risk_data = {
                'model': self.risk_model,
                'scaler': self.lifestyle_scaler,
                'features': self.LIFESTYLE_FEATURES,
                'feature_importance': self.trigger_importance,
                'info': {
                    'model_type': type(self.risk_model).__name__,
                    'accuracy': max(r.get('accuracy', 0) for r in self.risk_model_results.values()),
                    'f1_score': max(r.get('f1_score', 0) for r in self.risk_model_results.values()),
                    'trained_at': timestamp,
                    'task': 'risk_prediction'
                }
            }
            
            risk_path = base_path / "model.pkl"
            joblib.dump(risk_data, risk_path)
            logger.info(f"Risk predictor saved to: {risk_path}")
            
            # Also save scaler separately for compatibility
            scaler_path = base_path / "scaler.pkl"
            joblib.dump(self.lifestyle_scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
        
        # Save combined model info
        model_info = {
            'version': '2.0.0',
            'trained_at': timestamp,
            'symptom_classifier': {
                'model_type': type(self.symptom_model).__name__ if self.symptom_model else None,
                'classes': list(self.label_encoder.classes_) if self.symptom_model else [],
                'features': self.SYMPTOM_FEATURES,
                'feature_importance': self.symptom_importance
            },
            'risk_predictor': {
                'model_type': type(self.risk_model).__name__ if self.risk_model else None,
                'features': self.LIFESTYLE_FEATURES,
                'feature_importance': self.trigger_importance
            },
            'migraine_types': self.MIGRAINE_TYPES
        }
        
        info_path = base_path / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Model info saved to: {info_path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("ENHANCED MIGRAINE PREDICTION TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Find symptom dataset
    possible_paths = [
        "../../DatasetsNew/migraine_symptom_classification.csv",
        "../../../DatasetsNew/migraine_symptom_classification.csv",
        "migraine_symptom_classification.csv",
        "DatasetsNew/migraine_symptom_classification.csv"
    ]
    
    symptom_path = None
    for path in possible_paths:
        if Path(path).exists():
            symptom_path = path
            break
    
    # Initialize trainer
    trainer = EnhancedMigraineModelTrainer(symptom_data_path=symptom_path)
    
    # Load and train symptom classifier
    if symptom_path:
        logger.info(f"\nLoading symptom data from: {symptom_path}")
        trainer.load_symptom_data()
        trainer.train_symptom_classifier()
    else:
        logger.warning("Symptom dataset not found - skipping symptom classifier training")
    
    # Train risk predictor
    trainer.generate_lifestyle_data(n_samples=3000)
    trainer.train_risk_predictor()
    
    # Save all models
    trainer.save_models()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
