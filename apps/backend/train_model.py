#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def load_augmented_data():
    """
    Load the dataset with Normal and Mild cases.
    """
    print("="*60)
    print("LOADING DATASET")
    print("="*60)
    
    # Load augmented data
    df = pd.read_csv('./data/dataset.csv')
    print(f" dataset shape: {df.shape}")
    
    # Check distribution
    categories = ['Depression_Category', 'Anxiety_Category', 'Stress_Category']
    labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    
    print("\nDataset distribution:")
    for cat in categories:
        print(f"\n{cat.replace('_Category', '')}:")
        dist = df[cat].value_counts().sort_index()
        for i, count in dist.items():
            pct = count / len(df) * 100
            label = labels[i] if i < len(labels) else f"Category_{i}"
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    return df

def prepare_features_and_targets(df):
    """
    Separate features and target variables.
    """
    print("\n" + "="*40)
    print("PREPARING FEATURES AND TARGETS")
    print("="*40)
    
    # Target columns
    target_columns = ['Depression_Category', 'Anxiety_Category', 'Stress_Category']
    
    # Feature columns (everything except targets)
    feature_columns = [col for col in df.columns if col not in target_columns]
    
    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {len(target_columns)}")
    
    # Separate features and targets
    X = df[feature_columns].copy()
    y = df[target_columns].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    
    return X, y, feature_columns, target_columns

def train_balanced_model(X, y, feature_names, target_names):
    """
    Train a new model with the dataset.
    """
    print("\n" + "="*40)
    print("TRAINING MODEL")
    print("="*40)
    
    # Split data - use stratify on Depression only since multi-output stratification can fail
    # when some combinations have too few samples
    print("Splitting data with stratification on Depression category...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['Depression_Category']
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Check distribution after split
    print("\nTarget distribution in training set:")
    for target in target_names:
        dist = y_train[target].value_counts().sort_index()
        print(f"{target}: {dict(dist)}")
    
    # Create and train model
    print("\nTraining RandomForest with MultiOutput...")
    
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model = MultiOutputClassifier(base_model, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    # Evaluate model
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy for multi-output - need to do this per target
    individual_accuracies = []
    labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    
    for i, target in enumerate(target_names):
        target_accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        individual_accuracies.append(target_accuracy)
        print(f"{target} accuracy: {target_accuracy:.4f} ({target_accuracy*100:.2f}%)")
        
        # Confusion matrix
        cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
        print(f"\n{target} Confusion Matrix:")
        
        # Get unique classes in this target
        unique_classes = sorted(y_test.iloc[:, i].unique())
        
        print("Predicted ->", end="")
        for class_idx in unique_classes:
            print(f"\t{labels[class_idx][:4]}", end="")
        print()
        
        for true_idx, true_class in enumerate(unique_classes):
            print(f"True {labels[true_class][:4]}\t", end="")
            for pred_idx, pred_class in enumerate(unique_classes):
                if true_idx < len(cm) and pred_idx < len(cm[true_idx]):
                    print(f"\t{cm[true_idx][pred_idx]}", end="")
                else:
                    print(f"\t0", end="")
            print()
        print()
    
    # Overall accuracy (average of individual accuracies)
    overall_accuracy = np.mean(individual_accuracies)
    print(f"\nOverall average accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Alternative: Exact match accuracy (all targets must be correct)
    exact_matches = np.all(y_test.values == y_pred, axis=1)
    exact_match_accuracy = np.mean(exact_matches)
    print(f"Exact match accuracy: {exact_match_accuracy:.4f} ({exact_match_accuracy*100:.2f}%)")
    
    return model, X_test, y_test, y_pred

def save_balanced_model(model, feature_names, target_names, accuracy):
    """
    Save the trained model and metadata.
    """
    print("\n" + "="*40)
    print("SAVING MODEL")
    print("="*40)
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save model
    model_path = 'model/dass_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save feature names
    feature_path = 'model/feature_names.pkl'
    joblib.dump(feature_names, feature_path)
    print(f"✅ Feature names saved: {feature_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'MultiOutputClassifier(RandomForest)',
        'accuracy': accuracy,
        'dataset_size': 'dataset',
        'targets': target_names,
        'features': len(feature_names),
        'version': 'v1.0'
    }
    
    metadata_path = 'model/model_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"✅ Metadata saved: {metadata_path}")
    
    # Load and save existing preprocessing objects for compatibility
    try:
        label_encoders = joblib.load('model/new_label_encoders.pkl')
        scaler = joblib.load('model/new_scaler.pkl')
        
        # Save copies for model
        joblib.dump(label_encoders, 'model/label_encoders.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        print(f"✅ Preprocessing objects copied for model")
    except Exception as e:
        print(f"⚠️  Could not copy preprocessing objects: {e}")
    
    print(f"\n🎉 Model training complete!")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Model: {model_path}")

def main():
    """
    Main training pipeline.
    """
    print("🚀 TRAINING DASS MODEL")
    print("="*60)
    
    # 1. Load augmented data
    df = load_augmented_data()
    
    # 2. Prepare features and targets
    X, y, feature_names, target_names = prepare_features_and_targets(df)
    
    # 3. Train model
    model, X_test, y_test, y_pred = train_balanced_model(X, y, feature_names, target_names)
    
    # 4. Test with all "Never" responses
   
    
    # 5. Calculate overall accuracy and save model
    # Use the exact match accuracy as the main metric
    exact_matches = np.all(y_test.values == y_pred, axis=1)
    accuracy = np.mean(exact_matches)
    save_balanced_model(model, feature_names, target_names, accuracy)
    
    return model

if __name__ == "__main__":
    trained_model = main() 