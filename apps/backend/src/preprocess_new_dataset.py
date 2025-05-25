import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_new_dataset(file_path="data/dataset.csv"):
    """Load the new DASS dataset."""
    print("Loading DASS dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df

def calculate_dass_scores(df):
    """Calculate Depression, Anxiety, and Stress scores from DASS-21 questions."""
    print("Calculating DASS scores...")

    dass_mapping = {
        'Depression': [3, 5, 10, 13, 16, 17, 21],
        'Anxiety': [2, 4, 7, 9, 15, 19, 20],
        'Stress': [1, 6, 8, 11, 12, 14, 18]
    }
    
    result_df = df.copy()
    
    for category, questions in dass_mapping.items():
        question_cols = [f"Q{q}A" for q in questions]
        result_df[f'{category}_Score'] = result_df[question_cols].sum(axis=1) * 2
        
        print(f"{category} Score - Range: {result_df[f'{category}_Score'].min()}-{result_df[f'{category}_Score'].max()}")
    
    return result_df

def categorize_dass_scores(df):
    """Convert DASS scores to severity categories."""
    print("Creating severity categories...")
    

    categories = {
        'Depression': [(0, 9, 0), (10, 13, 1), (14, 20, 2), (21, 27, 3), (28, float('inf'), 4)],
        'Anxiety': [(0, 7, 0), (8, 9, 1), (10, 14, 2), (15, 19, 3), (20, float('inf'), 4)],
        'Stress': [(0, 14, 0), (15, 18, 1), (19, 25, 2), (26, 33, 3), (34, float('inf'), 4)]
    }
    
    labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    
    result_df = df.copy()
    
    for category in ['Depression', 'Anxiety', 'Stress']:
        score_col = f'{category}_Score'
        category_col = f'{category}_Category'
        
        def categorize_score(score):
            for min_val, max_val, cat_num in categories[category]:
                if min_val <= score <= max_val:
                    return cat_num
            return 4
        
        result_df[category_col] = result_df[score_col].apply(categorize_score)
        
        dist = result_df[category_col].value_counts().sort_index()
        print(f"{category} distribution:")
        for i, count in dist.items():
            print(f"  {labels[i]}: {count} ({count/len(result_df)*100:.1f}%)")
    
    return result_df

def select_and_preprocess_features(df):
    """Select relevant features and preprocess them for machine learning."""
    print("Selecting and preprocessing features...")
    
    dass_questions = [f"Q{i}A" for i in range(1, 22)]
    demographic_features = ['age', 'gender', 'education', 'race', 'religion', 'married']
    tipi_features = [f"TIPI{i}" for i in range(1, 11) if f"TIPI{i}" in df.columns]
    
    other_features = []
    for col in ['country', 'familysize', 'orientation', 'voted', 'engnat', 'hand']:
        if col in df.columns:
            other_features.append(col)
    
    feature_columns = dass_questions + demographic_features + tipi_features + other_features
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"Selected {len(available_features)} features:")
    print(f"  DASS questions: {len(dass_questions)}")
    print(f"  Demographics: {len([f for f in demographic_features if f in df.columns])}")
    print(f"  TIPI personality: {len(tipi_features)}")
    print(f"  Other features: {len([f for f in other_features if f in df.columns])}")
    
    features_df = df[available_features].copy()
    
    print("Handling missing values...")
    
    for col in dass_questions:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(features_df[col].median())
    
    categorical_features = ['gender', 'education', 'race', 'religion', 'married', 'country', 'orientation', 'voted', 'engnat', 'hand']
    for col in categorical_features:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(-1)
    
    numeric_features = ['age', 'familysize'] + tipi_features
    for col in numeric_features:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(features_df[col].median())
    

    if 'age' in features_df.columns:
        features_df['age'] = features_df['age'].clip(13, 100)
    
    print("Feature preprocessing completed!")
    return features_df, available_features

def encode_categorical_features(features_df):
    """Encode categorical features for machine learning."""
    print("Encoding categorical features...")
    
    encoded_df = features_df.copy()
    label_encoders = {}
    
    categorical_cols = ['gender', 'education', 'race', 'religion', 'married', 
                       'country', 'orientation', 'voted', 'engnat', 'hand']
    
    for col in categorical_cols:
        if col in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded {col}: {len(le.classes_)} unique values")
    
    return encoded_df, label_encoders

def scale_features(features_df):
    """Scale numerical features for better model performance."""
    print("Scaling features...")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
    
    print(f"Scaled {len(features_df.columns)} features")
    return scaled_df, scaler

def preprocess_new_dataset(input_file="data/dataset.csv", output_file="preprocessed_dass_data.csv"):
    """Complete preprocessing pipeline for the new DASS dataset."""
    print("="*60)
    print("PREPROCESSING NEW DASS DATASET")
    print("="*60)
    
    df = load_new_dataset(input_file)
    df_with_scores = calculate_dass_scores(df)
    df_categorized = categorize_dass_scores(df_with_scores)
    features_df, feature_names = select_and_preprocess_features(df_categorized)
    encoded_df, label_encoders = encode_categorical_features(features_df)
    scaled_df, scaler = scale_features(encoded_df)
    
    target_columns = ['Depression_Category', 'Anxiety_Category', 'Stress_Category']
    targets_df = df_categorized[target_columns].copy()
    
    final_df = pd.concat([scaled_df, targets_df], axis=1)
    
    final_df.to_csv(output_file, index=False)
    print(f"\nPreprocessed dataset saved to: {output_file}")
    print(f"   Shape: {final_df.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Targets: {len(target_columns)}")
    
    import joblib
    joblib.dump(label_encoders, "model/new_label_encoders.pkl")
    joblib.dump(scaler, "model/new_scaler.pkl")
    joblib.dump(feature_names, "model/new_feature_names.pkl")
    
    print("\nPreprocessing objects saved:")
    print("   - model/new_label_encoders.pkl")
    print("   - model/new_scaler.pkl") 
    print("   - model/new_feature_names.pkl")
    
    return final_df, feature_names, label_encoders, scaler

if __name__ == "__main__":
    processed_df, features, encoders, scaler = preprocess_new_dataset()
    
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original dataset: 39,775 samples × 172 columns")
    print(f"Processed dataset: {processed_df.shape[0]} samples × {processed_df.shape[1]} columns")
    print(f"Ready for machine learning training!") 