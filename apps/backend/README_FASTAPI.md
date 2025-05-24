# DASS Mental Health Assessment API v2.1

A FastAPI-based conversational mental health assessment application using the DASS-21 (Depression, Anxiety, and Stress Scale) with OpenAI integration for natural language interaction and PostgreSQL persistence.

## 🆕 What's New in v2.1

- **PostgreSQL Persistence**: All conversations, responses, and predictions stored in database
- **Conversation Memory**: Intelligent conversation context management following OpenAI guidelines
- **Analytics & Reporting**: Comprehensive analytics on conversation patterns and completion rates
- **Enhanced Data Model**: Normalized database schema for optimal performance
- **Token Usage Tracking**: Monitor OpenAI API usage per conversation
- **Processing Metrics**: Track prediction processing times and model performance
- **Fixed Pydantic v2 Compatibility**: Updated for latest Pydantic version

## 🏗️ Architecture

### Core Components

1. **FastAPI Application** (`main.py`) - Main application with all endpoints
2. **Conversation Service** (`services/conversation_service.py`) - Manages AI conversations with persistence
3. **DASS Prediction Service** (`src/dass_prediction_service.py`) - Existing ML model service
4. **Database Models** (`database.py`) - PostgreSQL models and connection management
5. **Pydantic Models** (`models.py`) - Request/response validation (v2 compatible)
6. **Configuration** (`config.py`) - Environment and settings management
7. **Database Initialization** (`init_database.py`) - Database setup and management

### Database Schema

- **conversations** - Store conversation metadata with UUIDs
- **messages** - Store individual conversation messages with roles and timestamps
- **dass_responses** - Store collected DASS question responses (Q1A-Q21A)
- **predictions** - Store DASS prediction results with model metadata
- **conversation_analytics** - Store comprehensive conversation analytics

### Key Features

- **Conversational Assessment**: Natural language DASS-21 assessment with memory
- **Persistent Storage**: All data stored in PostgreSQL for analysis and continuity
- **Memory Management**: Intelligent conversation context management to optimize tokens
- **Direct Prediction**: Traditional API endpoints for direct predictions
- **Conversation Management**: Create, continue, delete, and analyze conversations
- **Health Monitoring**: Comprehensive health checks and model status
- **Analytics Dashboard**: Detailed conversation and completion analytics
- **Auto-documentation**: Interactive API docs with Swagger UI

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **PostgreSQL 12+** running on localhost:5433
3. **OpenAI API key**
4. **Trained DASS model files** (from existing training)

### Installation

1. **Install Dependencies**

   ```bash
   pip install fastapi uvicorn openai pydantic python-dotenv psycopg2-binary sqlalchemy
   ```

   Or using requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```

2. **PostgreSQL Setup**

   Make sure PostgreSQL is running on port 5433 with your database created:

   ```sql
   CREATE DATABASE mindscan;
   ```

3. **Environment Setup**
   Create a `.env` file in the backend directory:

   ```env
   # OpenAI Configuration (REQUIRED)
   OPENAI_API_KEY=your_openai_api_key_here

   # Database Configuration (REQUIRED)
   DATABASE_URL=postgresql://postgres:your_password@localhost:5433/mindscan

   # Application Configuration
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   OPENAI_MODEL=gpt-4-1106-preview

   # Conversation Configuration
   MAX_CONVERSATION_LENGTH=50
   CONVERSATION_MEMORY_LIMIT=20

   # DASS Model Configuration
   DASS_MODEL_PATH=model/dass_model.pkl
   ```

4. **Initialize Database**

   ```bash
   python init_database.py
   ```

   This will create all necessary tables in your PostgreSQL database.

5. **Start the Server**

   ```bash
   python main.py
   ```

6. **Access the API**
   - API Documentation: http://localhost:8000/api/docs
   - Health Check: http://localhost:8000/api/health
   - Alternative Docs: http://localhost:8000/api/redoc

## 📚 API Endpoints

### Conversation Endpoints

#### Start New Conversation

```http
POST /api/conversations/new
Content-Type: application/json

{
  "initial_message": "Hello, I'd like to take the DASS assessment."
}
```

**Response:**

```json
{
  "conversation_id": "uuid-here",
  "message": "Hello! I'm here to guide you through the DASS-21 assessment..."
}
```

#### Continue Conversation

```http
POST /api/conversations/continue
Content-Type: application/json

{
  "conversation_id": "uuid-here",
  "message": "I sometimes feel anxious."
}
```

**Response:**

```json
{
  "conversation_id": "uuid-here",
  "message": "Thank you for sharing that. Let me ask about...",
  "is_assessment_complete": false,
  "predictions": null
}
```

#### Get Conversation History

```http
GET /api/conversations/{conversation_id}
```

#### Get Full Conversation Data (with responses and predictions)

```http
GET /api/conversations/{conversation_id}/full
```

#### Delete Conversation

```http
DELETE /api/conversations/{conversation_id}
```

### Analytics Endpoints

#### Get Overall Analytics

```http
GET /api/analytics
```

**Response:**

```json
{
  "total_conversations": 150,
  "completed_conversations": 120,
  "completion_rate": 0.8,
  "average_assessment_completion_rate": 0.85,
  "average_completion_time_minutes": 12.5
}
```

### Prediction Endpoints

#### Direct DASS Prediction

```http
POST /api/predict/dass
Content-Type: application/json

{
  "responses": {
    "Q1A": 2,
    "Q2A": 1,
    "Q3A": 3,
    ...
    "Q21A": 2
  }
}
```

#### Detailed Prediction (with probabilities)

```http
POST /api/predict/dass/detailed
Content-Type: application/json

{
  "responses": {
    "Q1A": 2,
    "Q2A": 1,
    ...
  }
}
```

### Information Endpoints

#### Get DASS Questions

```http
GET /api/dass/questions
```

#### Model Information

```http
GET /api/model/info
```

#### Health Check

```http
GET /api/health
```

## 🤖 Conversational Flow

The AI assistant follows this enhanced flow:

1. **Welcome**: Greets user and explains the DASS assessment
2. **Question Collection**: Asks DASS questions naturally, one or few at a time
3. **Response Extraction**: Parses user responses (numbers 1-4 or text like "sometimes")
4. **Progress Tracking**: Keeps track of collected responses with database persistence
5. **Memory Management**: Maintains conversation context while optimizing token usage
6. **Tool Calling**: When all 21 responses collected, automatically calls prediction tool
7. **Results**: Presents results in a supportive, professional manner
8. **Analytics**: Records completion metrics and conversation quality data

### Response Scale

- **1 = Never**
- **2 = Sometimes**
- **3 = Often**
- **4 = Almost Always**

## 🔧 Configuration

### Environment Variables

| Variable                    | Description                      | Default                                    |
| --------------------------- | -------------------------------- | ------------------------------------------ |
| `OPENAI_API_KEY`            | OpenAI API key (required)        | -                                          |
| `DATABASE_URL`              | PostgreSQL connection (required) | `postgresql://user:pass@localhost:5433/db` |
| `OPENAI_MODEL`              | OpenAI model to use              | `gpt-4-1106-preview`                       |
| `ENVIRONMENT`               | Environment mode                 | `development`                              |
| `LOG_LEVEL`                 | Logging level                    | `INFO`                                     |
| `MAX_CONVERSATION_LENGTH`   | Max messages per conversation    | `50`                                       |
| `CONVERSATION_MEMORY_LIMIT` | Messages kept in OpenAI context  | `20`                                       |
| `DASS_MODEL_PATH`           | Path to DASS model file          | `model/dass_model.pkl`                     |

### Memory Management

The application implements OpenAI's best practices for conversation memory:

- **System Message**: Always included first in context
- **Memory Limit**: Configurable limit (default: 20 messages) to control token usage
- **Context Summarization**: When approaching memory limits, creates summaries
- **Progress Tracking**: Maintains awareness of assessment progress across memory boundaries

## 📊 Database Management

### Initialize Database

```bash
# Create tables
python init_database.py

# Reset database (development only)
python init_database.py reset
```

### Database Queries

#### Check Active Conversations

```sql
SELECT COUNT(*) FROM conversations WHERE is_completed = FALSE;
```

#### Token Usage Analysis

```sql
SELECT
    DATE(created_at) as date,
    SUM(total_tokens_used) as daily_tokens
FROM conversation_analytics
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

#### Completion Rate Analysis

```sql
SELECT
    AVG(completion_rate) as avg_completion_rate,
    AVG(completion_time_minutes) as avg_time_minutes
FROM conversation_analytics
WHERE completion_rate > 0;
```

## 🚨 Troubleshooting

### Common Issues

#### 1. PostgreSQL Connection Error

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check if running on correct port
sudo netstat -tlnp | grep :5433
```

#### 2. Pydantic Compatibility Error

The application is updated for Pydantic v2. If you encounter regex errors, ensure you're using the latest version:

```bash
pip install pydantic>=2.0
```

#### 3. OpenAI API Key Issues

Ensure your OpenAI API key is valid and has sufficient credits:

```bash
# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

#### 4. DASS Model Not Found

Ensure the DASS model files are in the correct location:

```bash
ls -la model/
# Should show: dass_model.pkl, feature_names.pkl, model_metadata.pkl, etc.
```

## 📈 Production Deployment

### Security Considerations

1. **Database Security**: Use strong passwords, enable SSL, restrict access by IP
2. **API Security**: Implement rate limiting, add authentication if needed, use HTTPS
3. **Data Privacy**: Implement data retention policies, consider encryption at rest

### Performance Optimization

1. **Database Indexes**: Add indexes for frequently queried columns
2. **Connection Pooling**: Implement connection pooling for high-traffic scenarios
3. **Memory Management**: Tune `CONVERSATION_MEMORY_LIMIT` based on usage patterns

### Backup Strategy

```bash
# Daily database backups
pg_dump -h localhost -p 5433 -U postgres mindscan > backup_$(date +%Y%m%d).sql
```

## 📝 Development

### Adding New Features

1. **Database Changes**: Update `database.py` models and run migrations
2. **API Endpoints**: Add to `main.py` with proper validation
3. **Conversation Logic**: Modify `conversation_service.py` for new conversation features
4. **Frontend Integration**: Update models in `models.py` for API contracts

### Testing

```bash
# Test database connection
python init_database.py

# Test health endpoint
curl http://localhost:8000/api/health

# Test conversation flow
curl -X POST http://localhost:8000/api/conversations/new \
  -H "Content-Type: application/json" \
  -d '{"initial_message": "Hello"}'
```

This FastAPI application provides a robust, scalable platform for conversational mental health assessments with comprehensive data persistence and analytics capabilities.
