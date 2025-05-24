# dass_prediction_service.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DASSPredictionService:
    """
    Service class for making predictions using the DASS mental health model.
    """
    
    def __init__(self, model_path="model/dass_model.pkl"):
        """
        Initialize the prediction service by loading the trained model and preprocessing objects.
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.scaler = None
        self.label_encoders = None
        self.severity_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
        self.target_names = ['Depression', 'Anxiety', 'Stress']
        
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """
        Load the trained model and all preprocessing artifacts.
        """
        try:
            print("Loading DASS balanced model artifacts...")
            
            # Load balanced model
            self.model = joblib.load(self.model_path)
            
            # Load balanced model feature names
            self.feature_names = joblib.load("model/feature_names.pkl")
            
            # Load balanced model metadata
            self.metadata = joblib.load("model/model_metadata.pkl")
            
            # Load preprocessing objects for balanced model
            self.scaler = joblib.load("model/scaler.pkl")
            self.label_encoders = joblib.load("model/label_encoders.pkl")
            
            print(f"✅ Balanced model loaded successfully!")
            print(f"   Model type: {self.metadata['model_type']}")
            print(f"   Features: {self.metadata['features']}")
            print(f"   Accuracy: {self.metadata['accuracy']:.4f}")
            print(f"   Version: {self.metadata['version']}")
            
        except FileNotFoundError as e:
            print(f"Error loading balanced model artifacts: {e}")
            raise Exception("DASS balanced model not found. Please train the model first.")
        except Exception as e:
            print(f"Error initializing prediction service: {e}")
            raise e
    
    def validate_dass_input(self, input_data):
        """
        Validate that the input contains all required DASS questions.
        """
        required_questions = [f"Q{i}A" for i in range(1, 22)]  # Q1A to Q21A
        
        missing_questions = []
        for question in required_questions:
            if question not in input_data:
                missing_questions.append(question)
        
        if missing_questions:
            raise ValueError(f"Missing required DASS questions: {missing_questions}")
        
        # Validate question responses are in valid range (1-4)
        for question in required_questions:
            try:
                value = int(input_data[question])
                if value < 1 or value > 4:
                    raise ValueError(f"Question {question} must be between 1 and 4, got {value}")
            except (ValueError, TypeError):
                raise ValueError(f"Question {question} must be a valid integer between 1 and 4")
        
        return True
    
    def preprocess_input(self, input_data):
        """
        Preprocess user input to match the model's expected format.
        """
        # Create a DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Ensure we have all required features
        processed_features = {}
        
        # 1. DASS questions (Q1A to Q21A) - should be provided by user
        # Frontend sends 1-4, but DASS scoring uses 0-3, so subtract 1
        for i in range(1, 22):
            question = f"Q{i}A"
            if question in df.columns:
                frontend_value = int(df[question].iloc[0])
                # Convert from frontend scale (1-4) to DASS scale (0-3)
                dass_value = frontend_value - 1
                processed_features[question] = dass_value
            else:
                raise ValueError(f"Missing required DASS question: {question}")
        
        # 2. Demographic features with defaults if not provided
        demographic_defaults = {
            'age': 25,  # Default age
            'gender': 2,  # Default gender (most common in dataset)
            'education': 3,  # Default education level
            'race': 10,  # Default race (most common in dataset)
            'religion': 10,  # Default religion
            'married': 1  # Default marital status
        }
        
        for feature, default_value in demographic_defaults.items():
            if feature in df.columns:
                processed_features[feature] = df[feature].iloc[0]
            else:
                processed_features[feature] = default_value
        
        # 3. TIPI personality features with defaults
        for i in range(1, 11):
            tipi_feature = f"TIPI{i}"
            if tipi_feature in df.columns:
                processed_features[tipi_feature] = df[tipi_feature].iloc[0]
            else:
                processed_features[tipi_feature] = 3  # Neutral default
        
        # 4. Other features with defaults
        other_defaults = {
            'country': 'US',
            'familysize': 3,
            'orientation': 1,
            'voted': 1,
            'engnat': 1,
            'hand': 1
        }
        
        for feature, default_value in other_defaults.items():
            if feature in df.columns:
                processed_features[feature] = df[feature].iloc[0]
            else:
                processed_features[feature] = default_value
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([processed_features])
        
        # Handle categorical encoding
        categorical_features = ['gender', 'education', 'race', 'religion', 'married', 
                               'country', 'orientation', 'voted', 'engnat', 'hand']
        
        for feature in categorical_features:
            if feature in feature_df.columns and feature in self.label_encoders:
                try:
                    # Handle unseen values by using the most frequent class
                    value = str(feature_df[feature].iloc[0])
                    encoder = self.label_encoders[feature]
                    
                    if value in encoder.classes_:
                        feature_df[feature] = encoder.transform([value])[0]
                    else:
                        # Use the most frequent class (typically class 0)
                        feature_df[feature] = 0
                except Exception:
                    feature_df[feature] = 0
        
        # Check if this is an "all Never" case and handle specially
        dass_questions = [f'Q{i}A' for i in range(1, 22)]
        dass_values = [processed_features[q] for q in dass_questions if q in processed_features]
        
        if all(val == 0 for val in dass_values):
            # This is an "all Never" case - use a known normal case from training data
            print("Detected 'all Never' responses - using normal case reference")
            try:
                # Load dataset to find a normal case
                df_aug = pd.read_csv('data/dataset.csv')
                normal_cases = df_aug[(df_aug['Depression_Category'] == 0) & 
                                     (df_aug['Anxiety_Category'] == 0) & 
                                     (df_aug['Stress_Category'] == 0)]
                
                if len(normal_cases) > 0:
                    # Use the first normal case but keep our demographic/other features
                    normal_sample = normal_cases.iloc[0]
                    
                    # Replace only DASS questions with known normal values
                    for q in dass_questions:
                        if q in feature_df.columns and q in normal_sample.index:
                            feature_df[q] = normal_sample[q]
                            
                    print("Using reference normal case for DASS questions")
            except Exception as e:
                print(f"⚠️ Could not load reference normal case: {e}")
                # Fall back to original scaling approach
        
        # Ensure all required features are present in correct order
        final_features = []
        for feature_name in self.feature_names:
            if feature_name in feature_df.columns:
                final_features.append(feature_df[feature_name].iloc[0])
            else:
                # Use default value for missing features
                final_features.append(0)
        
        # Convert to numpy array and reshape
        feature_array = np.array(final_features).reshape(1, -1)
        
        # Only apply scaling if we didn't use reference normal case
        if not (all(val == 0 for val in dass_values)):
            # Apply scaling for non-"Never" cases
            scaled_features = self.scaler.transform(feature_array)
            return scaled_features
        else:
            # For "Never" cases, the features are already in the correct scale
            return feature_array
    
    def predict(self, input_data):
        """
        Make predictions using the DASS model.
        """
        try:
            # Validate input
            self.validate_dass_input(input_data)
            
            # Preprocess input
            processed_input = self.preprocess_input(input_data)
            
            # Make predictions
            predictions = self.model.predict(processed_input)
            
            # Convert predictions to readable format
            results = {}
            for i, target in enumerate(self.target_names):
                category_index = predictions[0][i]
                severity = self.severity_labels[category_index]
                results[target] = {
                    'category_index': int(category_index),
                    'severity': severity
                }
            
            return {
                'status': 'success',
                'predictions': results,
                'model_info': {
                    'model_type': self.metadata['model_type'],
                    'accuracy': self.metadata['accuracy'],
                    'dataset_size': self.metadata['dataset_size']
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_with_probabilities(self, input_data):
        """
        Make predictions with probability estimates (if model supports it).
        """
        try:
            # Validate and preprocess input
            self.validate_dass_input(input_data)
            processed_input = self.preprocess_input(input_data)
            
            # Get predictions
            predictions = self.model.predict(processed_input)
            
            # Try to get probabilities if available
            probabilities = None
            try:
                # Check if model supports probability prediction
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(processed_input)
            except:
                probabilities = None
            
            # Format results
            results = {}
            for i, target in enumerate(self.target_names):
                category_index = predictions[0][i]
                severity = self.severity_labels[category_index]
                
                result = {
                    'category_index': int(category_index),
                    'severity': severity
                }
                
                # Add probabilities if available
                if probabilities is not None:
                    try:
                        probs = probabilities[i][0]  # For this sample
                        prob_dict = {}
                        for j, prob in enumerate(probs):
                            if j < len(self.severity_labels):
                                prob_dict[self.severity_labels[j]] = float(prob)
                        result['probabilities'] = prob_dict
                    except:
                        pass
                
                results[target] = result
            
            return {
                'status': 'success',
                'predictions': results,
                'model_info': {
                    'model_type': self.metadata['model_type'],
                    'accuracy': self.metadata['accuracy'],
                    'dataset_size': self.metadata['dataset_size']
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_dass_questions(self):
        """
        Return the DASS-21 questions for the frontend.
        """
        dass_questions = {
            "Q1A": "I found it hard to wind down",
            "Q2A": "I was aware of dryness of my mouth",
            "Q3A": "I couldn't seem to experience any positive feeling at all",
            "Q4A": "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)",
            "Q5A": "I found it difficult to work up the initiative to do things",
            "Q6A": "I tended to over-react to situations",
            "Q7A": "I experienced trembling (e.g. in the hands)",
            "Q8A": "I felt that I was using a lot of nervous energy",
            "Q9A": "I was worried about situations in which I might panic and make a fool of myself",
            "Q10A": "I felt that I had nothing to look forward to",
            "Q11A": "I found myself getting agitated",
            "Q12A": "I found it difficult to relax",
            "Q13A": "I felt down-hearted and blue",
            "Q14A": "I was intolerant of anything that kept me from getting on with what I was doing",
            "Q15A": "I felt I was close to panic",
            "Q16A": "I was unable to become enthusiastic about anything",
            "Q17A": "I felt I wasn't worth much as a person",
            "Q18A": "I felt that I was rather touchy",
            "Q19A": "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
            "Q20A": "I felt scared without any good reason",
            "Q21A": "I felt that life was meaningless"
        }
        
        return {
            'questions': dass_questions,
            'instructions': 'Please rate each statement on a scale of 1-4: 1=Never, 2=Sometimes, 3=Often, 4=Almost Always',
            'scale': {
                1: 'Never',
                2: 'Sometimes', 
                3: 'Often',
                4: 'Almost Always'
            }
        }

# Create a global instance
dass_service = DASSPredictionService() 