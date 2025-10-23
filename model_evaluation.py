import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("TITANIC SURVIVAL PREDICTION - MODEL TRAINING & EVALUATION")
print("="*80)

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================
print("\nSTEP 1: Loading preprocessed data...")
print("-"*80)

X_train_full = pd.read_csv('X_train_processed.csv')
y_train_full = pd.read_csv('y_train.csv')['Survived']
X_test_final = pd.read_csv('X_test_processed.csv')
test_ids = pd.read_csv('test_ids.csv')

print(f"✓ Full Training Data: {X_train_full.shape}")
print(f"✓ Target Variable: {y_train_full.shape}")
print(f"✓ Test Data: {X_test_final.shape}")

# ============================================================================
# STEP 2: SPLIT TRAINING DATA INTO TRAIN AND VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Splitting data into Train and Validation sets")
print("-"*80)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_full
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"\nTarget distribution in training:")
print(y_train.value_counts(normalize=True))
print(f"\nTarget distribution in validation:")
print(y_val.value_counts(normalize=True))

# ============================================================================
# STEP 3: BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Training Baseline Logistic Regression Model")
print("-"*80)

# Train baseline model
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)

# Predictions
y_train_pred_baseline = baseline_model.predict(X_train)
y_val_pred_baseline = baseline_model.predict(X_val)

# Calculate metrics
train_acc_baseline = accuracy_score(y_train, y_train_pred_baseline)
val_acc_baseline = accuracy_score(y_val, y_val_pred_baseline)

print(f"\n✓ Baseline Model Trained")
print(f"Training Accuracy: {train_acc_baseline:.4f}")
print(f"Validation Accuracy: {val_acc_baseline:.4f}")
print(f"Difference (Overfitting): {train_acc_baseline - val_acc_baseline:.4f}")

# ============================================================================
# STEP 4: HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Hyperparameter Tuning with GridSearchCV")
print("-"*80)

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced']
}

print("Testing combinations of:")
print(f"  - C (Regularization): {param_grid['C']}")
print(f"  - Penalty: {param_grid['penalty']}")
print(f"  - Solver: {param_grid['solver']}")
print(f"  - Class Weight: {param_grid['class_weight']}")

# Perform grid search
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Grid Search Complete!")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# Predictions with best model
y_train_pred_best = best_model.predict(X_train)
y_val_pred_best = best_model.predict(X_val)

# Calculate metrics for best model
train_acc_best = accuracy_score(y_train, y_train_pred_best)
val_acc_best = accuracy_score(y_val, y_val_pred_best)

print(f"\nBest Model Performance:")
print(f"Training Accuracy: {train_acc_best:.4f}")
print(f"Validation Accuracy: {val_acc_best:.4f}")
print(f"Difference: {train_acc_best - val_acc_best:.4f}")

# ============================================================================
# STEP 5: DETAILED EVALUATION METRICS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Detailed Evaluation Metrics")
print("-"*80)

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate and display various metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# Calculate metrics for both sets
train_metrics = calculate_metrics(y_train, y_train_pred_best, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred_best, "Validation")

# Classification Report
print("\n" + "-"*80)
print("Classification Report (Validation Set):")
print("-"*80)
print(classification_report(y_val, y_val_pred_best, 
                          target_names=['Died', 'Survived']))

# ============================================================================
# STEP 6: VISUALIZATION - TRAIN VS VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Creating Visualizations")
print("-"*80)

# Figure 1: Train vs Validation Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance: Training vs Validation', fontsize=16, fontweight='bold')

# 1. Accuracy Comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_values = [train_metrics['accuracy'], train_metrics['precision'], 
                train_metrics['recall'], train_metrics['f1']]
val_values = [val_metrics['accuracy'], val_metrics['precision'], 
              val_metrics['recall'], val_metrics['f1']]

x = np.arange(len(metrics_names))
width = 0.35

axes[0, 0].bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='skyblue')
axes[0, 0].bar(x + width/2, val_values, width, label='Validation', alpha=0.8, color='lightcoral')
axes[0, 0].set_xlabel('Metrics')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Metrics Comparison: Train vs Validation')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metrics_names)
axes[0, 0].legend()
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (tv, vv) in enumerate(zip(train_values, val_values)):
    axes[0, 0].text(i - width/2, tv + 0.02, f'{tv:.3f}', ha='center', va='bottom', fontsize=9)
    axes[0, 0].text(i + width/2, vv + 0.02, f'{vv:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Confusion Matrix - Training
cm_train = confusion_matrix(y_train, y_train_pred_best)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
axes[0, 1].set_title('Confusion Matrix - Training Set')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')

# 3. Confusion Matrix - Validation
cm_val = confusion_matrix(y_val, y_val_pred_best)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
axes[1, 0].set_title('Confusion Matrix - Validation Set')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 4. ROC Curve
y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_val_proba = best_model.predict_proba(X_val)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)

roc_auc_train = roc_auc_score(y_train, y_train_proba)
roc_auc_val = roc_auc_score(y_val, y_val_proba)

axes[1, 1].plot(fpr_train, tpr_train, label=f'Training (AUC = {roc_auc_train:.3f})', 
                linewidth=2, color='skyblue')
axes[1, 1].plot(fpr_val, tpr_val, label=f'Validation (AUC = {roc_auc_val:.3f})', 
                linewidth=2, color='lightcoral')
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('train_vs_validation_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_vs_validation_comparison.png")
plt.show()

# ============================================================================
# STEP 7: TRAIN FINAL MODEL ON FULL TRAINING DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Training Final Model on Full Training Data")
print("-"*80)

# Train on full training data with best parameters
final_model = LogisticRegression(**grid_search.best_params_, max_iter=1000, random_state=42)
final_model.fit(X_train_full, y_train_full)

# Evaluate on full training data
y_train_full_pred = final_model.predict(X_train_full)
train_full_acc = accuracy_score(y_train_full, y_train_full_pred)

print(f"✓ Final Model Trained on Full Training Data")
print(f"Full Training Accuracy: {train_full_acc:.4f}")

# Cross-validation on full training data
cv_scores = cross_val_score(final_model, X_train_full, y_train_full, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 8: PREDICT ON TEST DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Making Predictions on Test Data")
print("-"*80)

# Predict on test data
y_test_pred = final_model.predict(X_test_final)
y_test_proba = final_model.predict_proba(X_test_final)[:, 1]

print(f"✓ Predictions Complete")
print(f"Predicted Survivors: {y_test_pred.sum()} ({y_test_pred.sum()/len(y_test_pred)*100:.1f}%)")
print(f"Predicted Deaths: {len(y_test_pred) - y_test_pred.sum()} ({(len(y_test_pred) - y_test_pred.sum())/len(y_test_pred)*100:.1f}%)")

# ============================================================================
# STEP 9: FINAL COMPARISON VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Creating Final Comparison Visualization")
print("-"*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Complete Model Analysis: Train, Validation & Test', fontsize=16, fontweight='bold')

# 1. Accuracy across datasets
datasets = ['Training\n(Split)', 'Validation', 'Full Training', 'CV Mean']
accuracies = [train_acc_best, val_acc_best, train_full_acc, cv_scores.mean()]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

bars = axes[0, 0].bar(datasets, accuracies, color=colors, alpha=0.8, edgecolor='black')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy Comparison Across Datasets')
axes[0, 0].set_ylim([0.7, 0.9])
axes[0, 0].grid(axis='y', alpha=0.3)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. Survival Rate Comparison
survival_rates = {
    'Training': y_train.mean(),
    'Validation': y_val.mean(),
    'Full Train': y_train_full.mean(),
    'Test (Pred)': y_test_pred.mean()
}

axes[0, 1].bar(survival_rates.keys(), survival_rates.values(), 
               color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'],
               alpha=0.8, edgecolor='black')
axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].set_title('Survival Rate Comparison')
axes[0, 1].set_ylim([0, 0.6])
axes[0, 1].grid(axis='y', alpha=0.3)

for i, (name, rate) in enumerate(survival_rates.items()):
    axes[0, 1].text(i, rate + 0.01, f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Cross-Validation Scores
axes[0, 2].plot(range(1, 6), cv_scores, marker='o', linewidth=2, markersize=8, color='green')
axes[0, 2].axhline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
axes[0, 2].fill_between(range(1, 6), 
                        cv_scores.mean() - cv_scores.std(), 
                        cv_scores.mean() + cv_scores.std(),
                        alpha=0.2, color='red')
axes[0, 2].set_xlabel('Fold')
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].set_title('5-Fold Cross-Validation Scores')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)
axes[0, 2].set_xticks(range(1, 6))

# 4. Feature Importance (Top 15)
feature_importance = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Coefficient': final_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False).head(15)

axes[1, 0].barh(range(len(feature_importance)), feature_importance['Coefficient'], 
                color=['green' if x > 0 else 'red' for x in feature_importance['Coefficient']],
                alpha=0.7)
axes[1, 0].set_yticks(range(len(feature_importance)))
axes[1, 0].set_yticklabels(feature_importance['Feature'], fontsize=8)
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 15 Feature Importance')
axes[1, 0].axvline(0, color='black', linewidth=0.8)
axes[1, 0].grid(axis='x', alpha=0.3)

# 5. Prediction Distribution
axes[1, 1].hist(y_test_proba, bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Test Set: Prediction Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 6. Model Parameters Summary
param_text = "Best Model Parameters:\n\n"
for param, value in grid_search.best_params_.items():
    param_text += f"{param}: {value}\n"
param_text += f"\nModel: Logistic Regression\n"
param_text += f"Features: {X_train_full.shape[1]}\n"
param_text += f"Training Samples: {len(X_train_full)}\n"
param_text += f"Test Samples: {len(X_test_final)}"

axes[1, 2].text(0.1, 0.5, param_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 2].axis('off')
axes[1, 2].set_title('Model Configuration')

plt.tight_layout()
plt.savefig('complete_model_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: complete_model_analysis.png")
plt.show()

# ============================================================================
# STEP 10: CREATE SUBMISSION FILE
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Creating Submission File")
print("-"*80)

submission = pd.DataFrame({
    'PassengerId': test_ids['PassengerId'],
    'Survived': y_test_pred
})

submission.to_csv('submission.csv', index=False)
print("✓ Submission file created: submission.csv")
print("\nFirst 10 predictions:")
print(submission.head(10))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

summary = f"""
MODEL PERFORMANCE:
------------------
Baseline Model:
  - Training Accuracy:    {train_acc_baseline:.4f}
  - Validation Accuracy:  {val_acc_baseline:.4f}

Best Model (After Tuning):
  - Training Accuracy:    {train_acc_best:.4f}
  - Validation Accuracy:  {val_acc_best:.4f}
  - Full Train Accuracy:  {train_full_acc:.4f}
  - CV Mean Accuracy:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  - ROC-AUC (Validation): {roc_auc_val:.4f}

Best Parameters:
{grid_search.best_params_}

FILES CREATED:
--------------
1. train_vs_validation_comparison.png - Train/Val comparison plots
2. complete_model_analysis.png - Comprehensive analysis
3. submission.csv - Kaggle submission file

RECOMMENDATIONS:
----------------
1. Overfitting Check: {("✓ Good" if abs(train_acc_best - val_acc_best) < 0.05 else "⚠ Possible overfitting")}
2. Model Stability: {("✓ Good" if cv_scores.std() < 0.03 else "⚠ High variance")}
3. Ready for Submission: ✓ Yes

NEXT STEPS:
-----------
1. Submit 'submission.csv' to Kaggle
2. If score is lower than expected, consider:
   - Feature engineering
   - Trying other models (Random Forest, XGBoost)
   - Ensemble methods
"""

print(summary)

print("="*80)
print("MODEL EVALUATION COMPLETE! ✓")
print("="*80)