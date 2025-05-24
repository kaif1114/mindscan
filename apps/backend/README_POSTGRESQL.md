# DASS Mental Health Assessment API v2.1 - PostgreSQL Integration

A FastAPI-based conversational mental health assessment application with persistent conversation memory using PostgreSQL and OpenAI integration following best practices for conversation management.

## 🆕 What's New in v2.1

- **PostgreSQL Persistence**: All conversations, responses, and predictions stored in database
- **Conversation Memory**: Intelligent conversation context management following OpenAI guidelines
- **Analytics & Reporting**: Comprehensive analytics on conversation patterns and completion rates
- **Enhanced Data Model**: Normalized database schema for optimal performance
- **Token Usage Tracking**: Monitor OpenAI API usage per conversation
- **Processing Metrics**: Track prediction processing times and model performance

## 🏗️ Database Architecture

### Database Schema

The application uses a normalized PostgreSQL schema with the following tables:

#### 1. **conversations**

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_completed BOOLEAN DEFAULT FALSE,
    user_ip VARCHAR(45),
    user_agent VARCHAR(500)
);
```

#### 2. **messages**

```sql
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER
);
```

#### 3. **dass_responses**

```sql
CREATE TABLE dass_responses (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    question_id VARCHAR(10) NOT NULL, -- Q1A, Q2A, etc.
    response_value INTEGER NOT NULL CHECK (response_value BETWEEN 1 AND 4),
    extracted_from_message_id INTEGER REFERENCES messages(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 4. **predictions**

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    depression_category INTEGER NOT NULL,
    depression_severity VARCHAR(50) NOT NULL,
    anxiety_category INTEGER NOT NULL,
    anxiety_severity VARCHAR(50) NOT NULL,
    stress_category INTEGER NOT NULL,
    stress_severity VARCHAR(50) NOT NULL,
    model_accuracy FLOAT,
    model_type VARCHAR(100),
    dataset_size INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER,
    raw_prediction_data JSON
);
```

#### 5. **conversation_analytics**

```sql
CREATE TABLE conversation_analytics (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    total_messages INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    completion_time_minutes FLOAT,
    questions_asked_count INTEGER DEFAULT 0,
    responses_collected_count INTEGER DEFAULT 0,
    completion_rate FLOAT DEFAULT 0.0,
    user_satisfaction_score INTEGER,
    assessment_abandoned BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🚀 Setup Instructions

### Prerequisites

1. **PostgreSQL 12+** running on localhost:5433
2. **Python 3.8+**
3. **OpenAI API Key**

### 1. PostgreSQL Setup

#### Install PostgreSQL (if not already installed):

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

**macOS:**

```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download from [PostgreSQL official website](https://www.postgresql.org/download/windows/)

#### Configure PostgreSQL:

1. **Create user and database:**

```sql
-- Connect to PostgreSQL as superuser
sudo -u postgres psql

-- Create user (if not exists)
CREATE USER postgres WITH PASSWORD 'password';

-- Create database
CREATE DATABASE dass_conversations OWNER postgres;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE dass_conversations TO postgres;
```

2. **Configure PostgreSQL to run on port 5433:**

```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/13/main/postgresql.conf

# Change port line to:
port = 5433

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### 2. Application Setup

#### Install Dependencies:

```bash
pip install fastapi uvicorn openai pydantic python-dotenv psycopg2-binary sqlalchemy
```

#### Environment Configuration:

Create `.env` file:

```env
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5433/dass_conversations

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

#### Initialize Database:

```bash
python init_database.py
```

#### Start the Application:

```bash
python main.py
```

## 🧠 Conversation Memory Management

The application implements OpenAI's best practices for conversation memory management:

### Memory Strategy

1. **System Message**: Always included first in context
2. **Memory Limit**: Configurable limit (default: 20 messages) to manage token usage
3. **Context Summarization**: When approaching memory limits, creates summaries of earlier conversation
4. **Progress Tracking**: Maintains awareness of assessment progress across message limits

### Implementation Details

```python
def _get_conversation_context(self, conversation_id: str, db: Session) -> List[Dict[str, str]]:
    # Get recent messages with limit to manage token usage
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp).limit(settings.CONVERSATION_MEMORY_LIMIT).all()

    # Always include system message first
    # Add context summary if approaching memory limits
    # Return optimized message list for OpenAI API
```

### Benefits

- **Token Efficiency**: Prevents excessive token usage in long conversations
- **Context Preservation**: Maintains conversation continuity while managing memory
- **Progress Awareness**: AI assistant remembers assessment progress despite memory limits
- **Cost Control**: Optimizes OpenAI API costs through intelligent memory management

## 📊 Analytics & Reporting

### Available Analytics

#### Conversation Metrics:

- Total conversations initiated
- Completion rates
- Average completion times
- Token usage statistics
- Assessment abandonment rates

#### Quality Metrics:

- Response collection efficiency
- Message-to-completion ratios
- Model prediction accuracy tracking
- Processing time analysis

### Analytics Endpoints

#### Get Overall Analytics:

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

#### Get Full Conversation Data:

```http
GET /api/conversations/{conversation_id}/full
```

**Response:**

```json
{
  "conversation": {
    "id": "uuid-here",
    "created_at": "2024-01-01T10:00:00",
    "updated_at": "2024-01-01T10:15:00",
    "is_completed": true
  },
  "messages": [...],
  "dass_responses": [
    {
      "question_id": "Q1A",
      "response_value": 2,
      "timestamp": "2024-01-01T10:02:00"
    }
  ],
  "predictions": {
    "depression": {"category": 0, "severity": "Normal"},
    "anxiety": {"category": 1, "severity": "Mild"},
    "stress": {"category": 0, "severity": "Normal"},
    "processing_time_ms": 150,
    "timestamp": "2024-01-01T10:15:00"
  }
}
```

## 🔧 Advanced Configuration

### Database Connection Tuning

For production deployments, consider these PostgreSQL optimizations:

```sql
-- Connection pooling
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB

-- Performance tuning
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
```

### Conversation Memory Tuning

Adjust memory limits based on your use case:

```env
# Conservative (lower token usage)
CONVERSATION_MEMORY_LIMIT=15

# Aggressive (more context, higher token usage)
CONVERSATION_MEMORY_LIMIT=30

# Maximum conversation length before auto-termination
MAX_CONVERSATION_LENGTH=100
```

### OpenAI Model Configuration

```env
# For better conversation quality
OPENAI_MODEL=gpt-4-1106-preview

# For cost optimization
OPENAI_MODEL=gpt-3.5-turbo-1106

# For specific capabilities
OPENAI_MODEL=gpt-4-turbo-preview
```

## 🔍 Monitoring & Debugging

### Database Queries for Monitoring

#### Active Conversations:

```sql
SELECT COUNT(*) FROM conversations WHERE is_completed = FALSE;
```

#### Token Usage by Day:

```sql
SELECT
    DATE(created_at) as date,
    SUM(total_tokens_used) as daily_tokens
FROM conversation_analytics
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

#### Completion Rate Analysis:

```sql
SELECT
    AVG(completion_rate) as avg_completion_rate,
    AVG(completion_time_minutes) as avg_time_minutes
FROM conversation_analytics
WHERE completion_rate > 0;
```

### Log Analysis

The application logs important events:

```python
# Conversation creation
logger.info(f"New conversation created: {conversation_id}")

# Assessment completion
logger.info(f"Assessment completed: {conversation_id}, time: {completion_time}min")

# Error tracking
logger.error(f"Prediction failed for {conversation_id}: {error}")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Failed

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check if running on correct port
sudo netstat -tlnp | grep :5433
```

#### 2. Table Creation Errors

```bash
# Reset database
python init_database.py reset

# Check permissions
sudo -u postgres psql -c "\du"
```

#### 3. Memory Limit Issues

```bash
# Check current memory usage
SELECT conversation_id, COUNT(*) as message_count
FROM messages
GROUP BY conversation_id
ORDER BY message_count DESC;
```

#### 4. OpenAI Token Limits

- Monitor token usage in conversation_analytics table
- Adjust CONVERSATION_MEMORY_LIMIT to control usage
- Implement conversation auto-termination for runaway conversations

### Performance Optimization

#### Database Indexes:

```sql
-- Optimize conversation queries
CREATE INDEX idx_conversations_created_at ON conversations(created_at);
CREATE INDEX idx_messages_conversation_timestamp ON messages(conversation_id, timestamp);
CREATE INDEX idx_dass_responses_conversation ON dass_responses(conversation_id);

-- Optimize analytics queries
CREATE INDEX idx_analytics_timestamp ON conversation_analytics(timestamp);
```

#### Connection Pooling:

Consider implementing connection pooling for high-traffic scenarios:

```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

## 📈 Production Deployment

### Security Considerations

1. **Database Security:**

   - Use strong passwords
   - Enable SSL connections
   - Restrict database access by IP
   - Regular security updates

2. **API Security:**

   - Implement rate limiting
   - Add authentication if needed
   - Use HTTPS in production
   - Validate all inputs

3. **Data Privacy:**
   - Implement data retention policies
   - Consider encryption at rest
   - Anonymize analytics data
   - GDPR compliance measures

### Backup Strategy

```bash
# Daily database backups
pg_dump -h localhost -p 5433 -U postgres dass_conversations > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/dass"
pg_dump -h localhost -p 5433 -U postgres dass_conversations | gzip > "$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql.gz"
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete
```

This comprehensive PostgreSQL integration provides a robust foundation for persistent conversation memory while following OpenAI's best practices for conversation management and token optimization.
