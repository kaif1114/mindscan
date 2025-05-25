# MindScan - AI-Powered Mental Health Assessment Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

MindScan is an AI-powered mental health assessment platform that provides conversational DASS-21 (Depression, Anxiety, and Stress Scale) assessments using OpenAI's GPT models. The platform combines machine learning predictions with empathetic conversational AI to deliver comprehensive mental health evaluations.

## 🌟 Features

- **Conversational DASS-21 Assessment**: Interactive mental health evaluation using natural language
- **AI-Powered Predictions**: Machine learning model trained on clinical data for accurate assessments
- **OpenAI Integration**: GPT-4 powered conversational interface for empathetic interactions
- **Persistent Storage**: PostgreSQL database for conversation history and assessment results
- **Real-time Analytics**: Dashboard for monitoring assessment trends and user engagement
- **RESTful API**: Comprehensive API for integration with other healthcare systems
- **Modern UI**: React-based frontend with Tailwind CSS for responsive design
- **Monorepo Architecture**: Turborepo-based structure for efficient development

## 🏗️ Project Structure

```
mindscan/
├── apps/
│   ├── backend/                 # FastAPI Python backend
│   │   ├── main.py             # Main FastAPI application
│   │   ├── train_model.py      # ML model training script
│   │   ├── models.py           # Pydantic data models
│   │   ├── config.py           # Configuration and environment variables
│   │   ├── database.py         # Database connection and ORM
│   │   ├── init_database.py    # Database initialization script
│   │   ├── requirements.txt    # Python dependencies
│   │   ├── services/           # Business logic services
│   │   ├── src/               # Core application modules
│   │   ├── model/             # Trained ML models and metadata
│   │   └── data/              # Training datasets
│   └── frontend/               # React TypeScript frontend
│       ├── src/               # React components and pages
│       ├── package.json       # Node.js dependencies
│       └── vite.config.ts     # Vite build configuration
├── packages/
│   ├── ui/                    # Shared UI components
│   ├── typescript-config/     # Shared TypeScript configurations
│   └── eslint-config/         # Shared ESLint configurations
├── package.json               # Root package.json for monorepo
├── turbo.json                 # Turborepo configuration
└── README.md                  # This file
```

## 🧠 How It Works

### 1. Assessment Flow

- Users initiate a conversation with the AI assistant
- The AI conducts a natural, empathetic DASS-21 assessment
- Responses are collected and processed through the ML model
- Results provide depression, anxiety, and stress severity levels

### 2. Machine Learning Pipeline

- **Model**: Multi-output Random Forest Classifier
- **Training Data**: Clinical DASS-21 dataset with severity categories
- **Features**: 21 DASS questionnaire responses (1-4 scale)
- **Outputs**: Depression, Anxiety, Stress severity levels (Normal, Mild, Moderate, Severe, Extremely Severe)

### 3. Conversational AI

- OpenAI GPT-4 integration for natural language processing
- Context-aware conversation management
- Empathetic response generation
- Progress tracking through assessment phases

## 📊 Dataset Information

### Primary Training Dataset

The MindScan platform uses a comprehensive clinical dataset for training the DASS-21 mental health assessment model.

#### Dataset Overview

| Attribute    | Details                            |
| ------------ | ---------------------------------- |
| **File**     | `apps/backend/data/dataset.csv`    |
| **Size**     | ~42 MB, 57,775 samples             |
| **Features** | 172 columns (before preprocessing) |
| **Source**   | Clinical DASS-21 research data     |
| **Format**   | CSV with headers                   |

#### Dataset Structure

The dataset contains multiple categories of features:

##### 1. DASS-21 Questionnaire Responses

- **Columns**: `Q1A` through `Q21A`
- **Scale**: 1-4 (Never, Sometimes, Often, Almost Always)
- **Purpose**: Core assessment questions for depression, anxiety, and stress

##### 2. Demographic Features

- **Age**: Participant age (13-100 years)
- **Gender**: Gender identity
- **Education**: Education level
- **Race**: Racial/ethnic background
- **Religion**: Religious affiliation
- **Married**: Marital status

##### 3. Personality Indicators (TIPI)

- **Columns**: `TIPI1` through `TIPI10`
- **Purpose**: Ten-Item Personality Inventory scores
- **Type**: Standardized personality assessment scores

##### 4. Additional Contextual Features

- **Country**: Geographic location
- **Family Size**: Number of family members
- **Orientation**: Sexual orientation
- **Voted**: Voting behavior
- **English Native**: Native English speaker status
- **Handedness**: Left/right-handed preference

#### Target Variables

The dataset includes three primary target categories:

| Category       | Score Calculation                 | Severity Levels                                                                  |
| -------------- | --------------------------------- | -------------------------------------------------------------------------------- |
| **Depression** | Questions: 3,5,10,13,16,17,21 × 2 | 0-9: Normal, 10-13: Mild, 14-20: Moderate, 21-27: Severe, 28+: Extremely Severe  |
| **Anxiety**    | Questions: 2,4,7,9,15,19,20 × 2   | 0-7: Normal, 8-9: Mild, 10-14: Moderate, 15-19: Severe, 20+: Extremely Severe    |
| **Stress**     | Questions: 1,6,8,11,12,14,18 × 2  | 0-14: Normal, 15-18: Mild, 19-25: Moderate, 26-33: Severe, 34+: Extremely Severe |

#### Data Preprocessing Pipeline

1. **Score Calculation**: Convert DASS-21 responses to clinical scores
2. **Categorization**: Map scores to severity levels (0-4 scale)
3. **Feature Selection**: Extract relevant features for ML training
4. **Missing Value Handling**: Impute missing values using median/mode
5. **Categorical Encoding**: Label encode categorical variables
6. **Feature Scaling**: StandardScaler normalization for numerical features

#### Dataset Distribution

The dataset provides representation across severity levels:

```
Depression Distribution:
  Normal: ~45%
  Mild: ~15%
  Moderate: ~20%
  Severe: ~12%
  Extremely Severe: ~8%

Anxiety Distribution:
  Normal: ~40%
  Mild: ~10%
  Moderate: ~25%
  Severe: ~15%
  Extremely Severe: ~10%

Stress Distribution:
  Normal: ~50%
  Mild: ~12%
  Moderate: ~20%
  Severe: ~10%
  Extremely Severe: ~8%
```

### User Response Dataset

#### Real-time Data Collection

| Attribute   | Details                                     |
| ----------- | ------------------------------------------- |
| **File**    | `apps/backend/data/user_responses_dass.csv` |
| **Size**    | ~1.5 KB (growing with usage)                |
| **Purpose** | Store real user assessment responses        |
| **Updates** | Real-time during conversations              |

#### Schema

```csv
Q1A,Q2A,...,Q21A,timestamp,Depression_Category,Depression_Severity,Anxiety_Category,Anxiety_Severity,Stress_Category,Stress_Severity
```

#### Features

- **DASS Responses**: User answers to all 21 questions
- **Timestamp**: Assessment completion time
- **Predictions**: Model-generated severity categories and labels
- **Persistence**: Stored for analytics and model improvement

### Data Privacy & Ethics

#### Privacy Measures

- **No Personal Identifiers**: No names, emails, or direct identifiers stored
- **Anonymized Data**: Only assessment responses and basic demographics
- **Secure Storage**: PostgreSQL with proper access controls
- **Data Retention**: Configurable retention policies

#### Ethical Considerations

- **Informed Consent**: Users informed about data collection
- **Clinical Validation**: Dataset based on validated DASS-21 instrument
- **Bias Mitigation**: Diverse demographic representation
- **Transparency**: Open about model limitations and intended use

#### Data Usage

- **Model Training**: Improve assessment accuracy
- **Research**: Mental health pattern analysis (anonymized)
- **Quality Assurance**: Monitor model performance
- **Analytics**: Usage patterns and effectiveness metrics

### Dataset Access & Updates

#### For Developers

```bash
# Access training dataset
cd apps/backend/data

# View user responses
cat user_responses_dass.csv

# Preprocess dataset
python src/preprocess_new_dataset.py
```

#### Model Retraining

The dataset supports continuous learning:

- **Automatic Updates**: New user responses added to training data
- **Retraining Pipeline**: Scheduled model updates with expanded dataset
- **Version Control**: Model versioning with dataset snapshots
- **Performance Monitoring**: Track accuracy improvements over time

## 🚀 Quick Start

### Prerequisites

- **Node.js** >= 18.0.0
- **Python** >= 3.8
- **PostgreSQL** >= 12
- **OpenAI API Key**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mindscan.git
cd mindscan
```

### 2. Install Dependencies

```bash
# Install root dependencies
npm install

# Install backend dependencies
cd apps/backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

### 3. Environment Setup

Create a `.env` file in the `apps/backend/` directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini-2024-07-18

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/mindscan

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_CONVERSATION_LENGTH=50
CONVERSATION_MEMORY_LIMIT=100

# Model Configuration
DASS_MODEL_PATH=model/dass_model.pkl
```

### 4. Database Setup

```bash
# Create PostgreSQL database
createdb mindscan_db

# Initialize database tables
cd apps/backend
python init_database.py
```

### 5. Train the ML Model

```bash
cd apps/backend
python train_model.py
```

### 6. Start Development Servers

```bash
# From the root directory
npm run dev
```

This will start:

- Backend API server on `http://localhost:8000`
- Frontend development server on `http://localhost:5173`

## 📋 Environment Variables

### Required Variables

| Variable         | Description                        | Example                                    |
| ---------------- | ---------------------------------- | ------------------------------------------ |
| `OPENAI_API_KEY` | OpenAI API key for GPT integration | `sk-...`                                   |
| `DATABASE_URL`   | PostgreSQL connection string       | `postgresql://user:pass@localhost:5432/db` |

### Optional Variables

| Variable                    | Description                   | Default                  |
| --------------------------- | ----------------------------- | ------------------------ |
| `OPENAI_MODEL`              | OpenAI model to use           | `gpt-4o-mini-2024-07-18` |
| `ENVIRONMENT`               | Application environment       | `development`            |
| `LOG_LEVEL`                 | Logging level                 | `INFO`                   |
| `MAX_CONVERSATION_LENGTH`   | Max messages per conversation | `50`                     |
| `CONVERSATION_MEMORY_LIMIT` | Conversation memory limit     | `100`                    |
| `DASS_MODEL_PATH`           | Path to trained ML model      | `model/dass_model.pkl`   |

## 🐍 Python Virtual Environment

### Create Virtual Environment

```bash
cd apps/backend

# Create virtual environment
python -m venv mindscan_venv

# Activate virtual environment
# On Windows:
mindscan_venv\Scripts\activate
# On macOS/Linux:
source mindscan_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Deactivate Virtual Environment

```bash
deactivate
```

## 🔧 Dependencies

### Backend Dependencies

```txt
fastapi==0.104.1              # Web framework
uvicorn[standard]==0.24.0     # ASGI server
openai==1.3.5                 # OpenAI API client
pydantic==2.5.0               # Data validation
python-multipart==0.0.6       # Form data handling
python-dotenv==1.0.0          # Environment variables
pandas==2.1.3                 # Data manipulation
numpy==1.24.3                 # Numerical computing
scikit-learn==1.3.2           # Machine learning
joblib==1.3.2                 # Model serialization
python-jose[cryptography]==3.3.0  # JWT handling
passlib[bcrypt]==1.7.4        # Password hashing
sqlalchemy==2.0.23            # ORM
alembic==1.12.1               # Database migrations
asyncpg==0.29.0               # PostgreSQL async driver
aiofiles==23.2.1              # Async file operations
psycopg2-binary==2.9.9        # PostgreSQL adapter
databases[postgresql]==0.8.0  # Database abstraction
```

### Frontend Dependencies

```json
{
  "dependencies": {
    "@tanstack/react-query": "^5.77.0", // Data fetching
    "axios": "^1.9.0", // HTTP client
    "framer-motion": "^11.0.6", // Animations
    "lucide-react": "^0.344.0", // Icons
    "react": "^18.3.1", // React framework
    "react-dom": "^18.3.1", // React DOM
    "react-router-dom": "^6.22.3", // Routing
    "react-textarea-autosize": "^8.5.3" // Auto-resizing textarea
  }
}
```

## 🤖 Model Training

### Training Process

1. **Data Preparation**: Load and preprocess DASS-21 clinical dataset
2. **Feature Engineering**: Extract questionnaire responses as features
3. **Model Training**: Train Multi-output Random Forest Classifier
4. **Evaluation**: Assess model performance with cross-validation
5. **Model Persistence**: Save trained model and metadata

### Training Command

```bash
cd apps/backend
python train_model.py
```

### Model Performance

- **Model Type**: Multi-output Random Forest Classifier
- **Features**: 21 DASS questionnaire responses
- **Targets**: Depression, Anxiety, Stress severity levels
- **Accuracy**: ~85-90% (varies by category)
- **Validation**: Stratified cross-validation

### Retraining

The model can be retrained with new data:

```bash
# Via API endpoint
POST /api/model/retrain

# Or directly via script
python train_model.py
```

## 🌐 API Documentation

### Base URL

- Development: `http://localhost:8000`
- Production: `https://your-domain.com`

### Key Endpoints

#### Health Check

```http
GET /api/health
```

#### Start New Conversation

```http
POST /api/conversations/new
Content-Type: application/json

{
  "initial_message": "I'd like to take a mental health assessment"
}
```

#### Continue Conversation

```http
POST /api/conversations/continue
Content-Type: application/json

{
  "conversation_id": "uuid",
  "message": "Sometimes"
}
```

#### Direct DASS Prediction

```http
POST /api/predict/dass
Content-Type: application/json

{
  "Q1A": 2, "Q2A": 1, "Q3A": 3, ...
}
```

### Interactive API Documentation

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## 🚀 Deployment

### Production Build

```bash
# Build all applications
npm run build

# Build specific app
npm run build --filter=frontend
npm run build --filter=backend
```

### Docker Deployment

```dockerfile
# Example Dockerfile for backend
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

Ensure all environment variables are properly set in production:

```bash
# Production environment variables
export ENVIRONMENT=production
export DATABASE_URL=postgresql://prod_user:prod_pass@prod_host:5432/prod_db
export OPENAI_API_KEY=your_production_api_key
```

## 🧪 Testing

### Backend Tests

```bash
cd apps/backend
python -m pytest tests/
```

### Frontend Tests

```bash
cd apps/frontend
npm test
```

### Integration Tests

```bash
# Run all tests
npm run test

# Run specific test suite
npm run test --filter=backend
```

## 📊 Monitoring and Analytics

### Health Monitoring

- Health check endpoint: `/api/health`
- Model status monitoring
- Database connectivity checks

### Analytics Dashboard

- Conversation metrics
- Assessment completion rates
- Severity distribution analysis
- User engagement tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow TypeScript/Python type hints
- Write comprehensive tests
- Update documentation for new features
- Follow existing code style and conventions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check this README and API docs
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

## 🔮 Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] Integration with healthcare systems
- [ ] Real-time notifications
- [ ] Enhanced ML models
- [ ] Therapist dashboard

## ⚠️ Disclaimer

This application is for educational and research purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about mental health conditions.

---

**Built with ❤️ for mental health awareness and support**
