"""
Database configuration and models for DASS conversation persistence
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from config import settings

# Use DATABASE_URL from settings/config instead of hardcoded value
DATABASE_URL = settings.DATABASE_URL

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Conversation(Base):
    """Store conversation metadata"""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_completed = Column(Boolean, default=False)
    user_ip = Column(String(45), nullable=True)  # For basic user tracking
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    dass_responses = relationship("DASSResponse", back_populates="conversation", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="conversation", cascade="all, delete-orphan")
    analytics = relationship("ConversationAnalytics", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    """Store individual messages in conversations"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    token_count = Column(Integer, nullable=True)  # For OpenAI usage tracking
    
    # Relationship
    conversation = relationship("Conversation", back_populates="messages")

class DASSResponse(Base):
    """Store collected DASS responses"""
    __tablename__ = "dass_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    question_id = Column(String(10), nullable=False)  # Q1A, Q2A, etc.
    response_value = Column(Integer, nullable=False)  # 1-4
    extracted_from_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="dass_responses")

class Prediction(Base):
    """Store DASS prediction results"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    
    # Prediction results
    depression_category = Column(Integer, nullable=False)
    depression_severity = Column(String(50), nullable=False)
    anxiety_category = Column(Integer, nullable=False)
    anxiety_severity = Column(String(50), nullable=False)
    stress_category = Column(Integer, nullable=False)
    stress_severity = Column(String(50), nullable=False)
    
    # Model information
    model_accuracy = Column(Float, nullable=True)
    model_type = Column(String(100), nullable=True)
    dataset_size = Column(String(100), nullable=True)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Raw prediction data (JSON)
    raw_prediction_data = Column(JSON, nullable=True)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="predictions")

class ConversationAnalytics(Base):
    """Store analytics about conversations"""
    __tablename__ = "conversation_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    
    # Conversation metrics
    total_messages = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    completion_time_minutes = Column(Float, nullable=True)
    
    # Assessment metrics
    questions_asked_count = Column(Integer, default=0)
    responses_collected_count = Column(Integer, default=0)
    completion_rate = Column(Float, default=0.0)  # responses_collected / 21
    
    # Quality metrics
    user_satisfaction_score = Column(Integer, nullable=True)  # 1-5 if collected
    assessment_abandoned = Column(Boolean, default=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="analytics")

# Database dependency
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """Drop all tables (for development/testing)"""
    Base.metadata.drop_all(bind=engine) 