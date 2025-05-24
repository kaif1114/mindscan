from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None

class ConversationRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ConversationResponse(BaseModel):
    conversation_id: str
    message: str
    is_assessment_complete: bool = False
    predictions: Optional[Dict[str, Any]] = None

class NewConversationRequest(BaseModel):
    initial_message: Optional[str] = "Hello, I'd like to take the DASS assessment."

class NewConversationResponse(BaseModel):
    conversation_id: str
    message: str

class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    is_completed: bool
    message_count: int
    last_message: str
    last_message_timestamp: str
    has_predictions: bool

class ConversationsListResponse(BaseModel):
    conversations: List[ConversationSummary]
    total: int
    limit: int
    offset: int

class DASSQuestionResponse(BaseModel):
    question_id: str = Field(..., pattern=r"^Q\d{1,2}A$")
    response: int = Field(..., ge=1, le=4)

class DASSPredictionRequest(BaseModel):
    responses: Dict[str, int] = Field(..., description="Dictionary with keys Q1A-Q21A and values 1-4")

class PredictionResult(BaseModel):
    category_index: int
    severity: str
    probabilities: Optional[Dict[str, float]] = None

class DASSPredictionResponse(BaseModel):
    status: str
    predictions: Optional[Dict[str, PredictionResult]] = None
    model_info: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    dass_model: str
    accuracy: Optional[float] = None
    timestamp: datetime

class ModelInfoResponse(BaseModel):
    status: str
    model_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    details: Optional[Dict[str, Any]] = None 