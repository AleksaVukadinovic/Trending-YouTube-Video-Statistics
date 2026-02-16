import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class ClusterPredictor:
    """
    Train classifiers to predict cluster membership for new videos.
    """

    def __init__(self, X, labels, feature_names):
        self.X = X
        self.labels = labels
        self.feature_names = feature_names
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()

    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into train/test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.labels, test_size=test_size, random_state=random_state, stratify=self.labels
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Class distribution (train): {np.bincount(self.y_train)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_random_forest(self, n_estimators=100):
        """Train Random Forest classifier."""
        print("Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(self.X_train, self.y_train)
        
        train_acc = rf.score(self.X_train, self.y_train)
        test_acc = rf.score(self.X_test, self.y_test)
        
        self.models['random_forest'] = {
            'model': rf,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        
        return rf

    def train_gradient_boosting(self, n_estimators=100):
        """Train Gradient Boosting classifier."""
        print("Training Gradient Boosting...")
        
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=42,
            max_depth=5
        )
        gb.fit(self.X_train, self.y_train)
        
        train_acc = gb.score(self.X_train, self.y_train)
        test_acc = gb.score(self.X_test, self.y_test)
        
        self.models['gradient_boosting'] = {
            'model': gb,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        
        return gb

    def train_logistic_regression(self):
        """Train Logistic Regression classifier."""
        print("Training Logistic Regression...")
        
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            multi_class='multinomial'
        )
        lr.fit(self.X_train, self.y_train)
        
        train_acc = lr.score(self.X_train, self.y_train)
        test_acc = lr.score(self.X_test, self.y_test)
        
        self.models['logistic_regression'] = {
            'model': lr,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        
        return lr

    def train_all_models(self):
        """Train all classifier models."""
        if not hasattr(self, 'X_train'):
            self.prepare_data()
        
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_logistic_regression()
        
        
        best_name = max(self.models.keys(), key=lambda k: self.models[k]['test_accuracy'])
        self.best_model = self.models[best_name]['model']
        print(f"\nBest model: {best_name} (test accuracy: {self.models[best_name]['test_accuracy']:.3f})")
        
        return self.models

    def cross_validate(self, model_name='random_forest', cv=5):
        """Perform cross-validation."""
        if model_name not in self.models:
            print(f"Model {model_name} not trained yet")
            return None
        
        model = self.models[model_name]['model']
        scores = cross_val_score(model, self.X, self.labels, cv=cv, scoring='accuracy')
        
        print(f"\n{model_name} Cross-Validation Results:")
        print(f"  Scores: {scores}")
        print(f"  Mean: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
        
        return scores

    def hyperparameter_tuning(self, model_type='random_forest'):
        """Perform hyperparameter tuning using GridSearchCV."""
        print(f"Tuning {model_type} hyperparameters...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            model = GradientBoostingClassifier(random_state=42)
        else:
            print(f"Tuning not implemented for {model_type}")
            return None
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        
        self.models[f'{model_type}_tuned'] = {
            'model': grid_search.best_estimator_,
            'train_accuracy': grid_search.best_estimator_.score(self.X_train, self.y_train),
            'test_accuracy': grid_search.best_estimator_.score(self.X_test, self.y_test),
            'best_params': grid_search.best_params_
        }
        
        return grid_search.best_estimator_

    def plot_model_comparison(self):
        """Compare model performances."""
        if not self.models:
            self.train_all_models()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        
        ax1 = axes[0]
        model_names = list(self.models.keys())
        train_accs = [self.models[m]['train_accuracy'] for m in model_names]
        test_accs = [self.models[m]['test_accuracy'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', '\n') for m in model_names], fontsize=9)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        
        ax2 = axes[1]
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title('Confusion Matrix (Best Model)')
        
        plt.tight_layout()
        plt.show()
        
        
        print("\nClassification Report (Best Model):")
        print(classification_report(self.y_test, y_pred))

    def predict_new_video(self, features):
        """Predict cluster for a new video."""
        if self.best_model is None:
            print("No model trained yet. Training all models...")
            self.train_all_models()
        
        features = np.array(features).reshape(1, -1)
        prediction = self.best_model.predict(features)[0]
        probabilities = self.best_model.predict_proba(features)[0]
        
        return prediction, probabilities


class ViewCountPredictor:
    """
    Predict view counts using regression models.
    """

    def __init__(self, X, y_views, feature_names, cluster_labels=None):
        self.X = X
        self.y_views = y_views
        self.feature_names = feature_names
        self.cluster_labels = cluster_labels
        self.models = {}
        self.best_model = None

    def prepare_data(self, test_size=0.2, random_state=42, log_transform_target=True):
        """Prepare data for regression."""
        
        if log_transform_target:
            self.y = np.log1p(self.y_views)
            self.log_transformed = True
        else:
            self.y = self.y_views
            self.log_transformed = False
        
        
        if self.cluster_labels is not None:
            cluster_dummies = pd.get_dummies(self.cluster_labels, prefix='cluster')
            self.X_with_cluster = np.hstack([self.X, cluster_dummies.values])
            self.feature_names_extended = list(self.feature_names) + list(cluster_dummies.columns)
        else:
            self.X_with_cluster = self.X
            self.feature_names_extended = self.feature_names
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_with_cluster, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Target range: {self.y.min():.2f} to {self.y.max():.2f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_random_forest(self, n_estimators=100):
        """Train Random Forest regressor."""
        print("Training Random Forest Regressor...")
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        
        train_pred = rf.predict(self.X_train)
        test_pred = rf.predict(self.X_test)
        
        self.models['random_forest'] = {
            'model': rf,
            'train_r2': r2_score(self.y_train, train_pred),
            'test_r2': r2_score(self.y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)),
            'test_mae': mean_absolute_error(self.y_test, test_pred)
        }
        
        print(f"  Train R²: {self.models['random_forest']['train_r2']:.3f}")
        print(f"  Test R²: {self.models['random_forest']['test_r2']:.3f}")
        print(f"  Test RMSE: {self.models['random_forest']['test_rmse']:.3f}")
        
        return rf

    def train_gradient_boosting(self, n_estimators=100):
        """Train Gradient Boosting regressor."""
        print("Training Gradient Boosting Regressor...")
        
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=42,
            max_depth=5
        )
        gb.fit(self.X_train, self.y_train)
        
        train_pred = gb.predict(self.X_train)
        test_pred = gb.predict(self.X_test)
        
        self.models['gradient_boosting'] = {
            'model': gb,
            'train_r2': r2_score(self.y_train, train_pred),
            'test_r2': r2_score(self.y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)),
            'test_mae': mean_absolute_error(self.y_test, test_pred)
        }
        
        print(f"  Train R²: {self.models['gradient_boosting']['train_r2']:.3f}")
        print(f"  Test R²: {self.models['gradient_boosting']['test_r2']:.3f}")
        print(f"  Test RMSE: {self.models['gradient_boosting']['test_rmse']:.3f}")
        
        return gb

    def train_ridge_regression(self, alpha=1.0):
        """Train Ridge regression."""
        print("Training Ridge Regression...")
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(self.X_train, self.y_train)
        
        train_pred = ridge.predict(self.X_train)
        test_pred = ridge.predict(self.X_test)
        
        self.models['ridge'] = {
            'model': ridge,
            'train_r2': r2_score(self.y_train, train_pred),
            'test_r2': r2_score(self.y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)),
            'test_mae': mean_absolute_error(self.y_test, test_pred)
        }
        
        print(f"  Train R²: {self.models['ridge']['train_r2']:.3f}")
        print(f"  Test R²: {self.models['ridge']['test_r2']:.3f}")
        print(f"  Test RMSE: {self.models['ridge']['test_rmse']:.3f}")
        
        return ridge

    def train_all_models(self):
        """Train all regression models."""
        if not hasattr(self, 'X_train'):
            self.prepare_data()
        
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_ridge_regression()
        
        
        best_name = max(self.models.keys(), key=lambda k: self.models[k]['test_r2'])
        self.best_model = self.models[best_name]['model']
        print(f"\nBest model: {best_name} (test R²: {self.models[best_name]['test_r2']:.3f})")
        
        return self.models

    def plot_model_comparison(self):
        """Compare regression model performances."""
        if not self.models:
            self.train_all_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        
        ax1 = axes[0, 0]
        model_names = list(self.models.keys())
        train_r2 = [self.models[m]['train_r2'] for m in model_names]
        test_r2 = [self.models[m]['test_r2'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, train_r2, width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, test_r2, width, label='Test', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', '\n') for m in model_names])
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.legend()
        
        
        ax2 = axes[0, 1]
        train_rmse = [self.models[m]['train_rmse'] for m in model_names]
        test_rmse = [self.models[m]['test_rmse'] for m in model_names]
        
        ax2.bar(x - width/2, train_rmse, width, label='Train', alpha=0.8)
        ax2.bar(x + width/2, test_rmse, width, label='Test', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', '\n') for m in model_names])
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Comparison')
        ax2.legend()
        
        
        ax3 = axes[1, 0]
        y_pred = self.best_model.predict(self.X_test)
        ax3.scatter(self.y_test, y_pred, alpha=0.3, s=10)
        
        
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax3.set_xlabel('Actual (log views)' if self.log_transformed else 'Actual Views')
        ax3.set_ylabel('Predicted (log views)' if self.log_transformed else 'Predicted Views')
        ax3.set_title('Actual vs Predicted (Best Model)')
        ax3.legend()
        
        
        ax4 = axes[1, 1]
        residuals = self.y_test - y_pred
        ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--')
        ax4.set_xlabel('Residual')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Residuals Distribution')
        
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n=15):
        """Plot feature importance for tree-based models."""
        if 'random_forest' not in self.models:
            self.train_random_forest()
        
        rf = self.models['random_forest']['model']
        importances = rf.feature_importances_
        
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(top_n), importances[indices], alpha=0.8)
        plt.xticks(range(top_n), [self.feature_names_extended[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance for View Count Prediction')
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame({
            'feature': [self.feature_names_extended[i] for i in indices],
            'importance': importances[indices]
        })

    def predict_views(self, features):
        """Predict view count for new video."""
        if self.best_model is None:
            self.train_all_models()
        
        features = np.array(features).reshape(1, -1)
        prediction = self.best_model.predict(features)[0]
        
        if self.log_transformed:
            prediction = np.expm1(prediction)  
        
        return prediction
