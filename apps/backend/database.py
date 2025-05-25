"""Database configuration and models for DASS conversation persistence"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_completed = Column(Boolean, default=False)
    user_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    dass_responses = relationship("DASSResponse", back_populates="conversation", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="conversation", cascade="all, delete-orphan")
    analytics = relationship("ConversationAnalytics", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    token_count = Column(Integer, nullable=True)
    
    conversation = relationship("Conversation", back_populates="messages")

class DASSResponse(Base):
    __tablename__ = "dass_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    question_id = Column(String(10), nullable=False)
    response_value = Column(Integer, nullable=False)
    extracted_from_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="dass_responses")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    
    depression_category = Column(Integer, nullable=False)
    depression_severity = Column(String(50), nullable=False)
    anxiety_category = Column(Integer, nullable=False)
    anxiety_severity = Column(String(50), nullable=False)
    stress_category = Column(Integer, nullable=False)
    stress_severity = Column(String(50), nullable=False)
    
    model_accuracy = Column(Float, nullable=True)
    model_type = Column(String(100), nullable=True)
    dataset_size = Column(String(100), nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer, nullable=True)
    
    raw_prediction_data = Column(JSON, nullable=True)
    
    conversation = relationship("Conversation", back_populates="predictions")

class ConversationAnalytics(Base):
    __tablename__ = "conversation_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    
    total_messages = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    completion_time_minutes = Column(Float, nullable=True)
    
    questions_asked_count = Column(Integer, default=0)
    responses_collected_count = Column(Integer, default=0)
    completion_rate = Column(Float, default=0.0)
    
    user_satisfaction_score = Column(Integer, nullable=True)
    assessment_abandoned = Column(Boolean, default=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="analytics")

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)

def drop_tables():
    Base.metadata.drop_all(bind=engine)