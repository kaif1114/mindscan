import uuid
import json
import re
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import desc

from config import settings
from database import (
    get_db, Conversation, Message, DASSResponse, Prediction, 
    ConversationAnalytics, SessionLocal
)
from src.dass_prediction_service import dass_service

class ConversationService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            max_retries=0  # Disable automatic retries to prevent multiple API calls
        )
        
    def create_conversation(self, initial_message: str = None, user_ip: str = None, user_agent: str = None) -> Tuple[str, str]:
        """Create a new conversation and return conversation_id and initial response."""
        db = SessionLocal()
        try:
            # Create conversation record
            conversation = Conversation(
                user_ip=user_ip,
                user_agent=user_agent
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            
            conversation_id = str(conversation.id)
            
            # Add system message
            system_message = Message(
                conversation_id=conversation.id,
                role="system",
                content=settings.DASS_SYSTEM_PROMPT
            )
            db.add(system_message)
            
            # Add user's initial message if provided
            if initial_message:
                user_message = Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=initial_message
                )
                db.add(user_message)
            
            db.commit()
            
            # Get initial assistant response
            response = self._get_assistant_response(conversation_id, db)
            
            return conversation_id, response
            
        finally:
            db.close()
    
    def continue_conversation(self, conversation_id: str, user_message: str) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
        """
        Continue an existing conversation.
        Returns: (assistant_response, is_complete, predictions)
        """
        db = SessionLocal()
        try:
            # Verify conversation exists
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                raise ValueError("Conversation not found")
            
            # Add user message
            user_msg = Message(
                conversation_id=conversation.id,
                role="user",
                content=user_message
            )
            db.add(user_msg)
            db.commit()
            db.refresh(user_msg)
            
            # Extract any DASS responses from the user message
            self._extract_and_store_dass_responses(conversation_id, user_message, user_msg.id, db)
            
            # Check if assessment is complete
            is_complete = self._is_assessment_complete(conversation_id, db)
            predictions = None
            
            if is_complete:
                # Make DASS prediction
                try:
                    predictions = self._make_dass_prediction(conversation_id, db)
                    
                    # Mark conversation as completed
                    conversation.is_completed = True
                    conversation.updated_at = datetime.utcnow()
                    db.commit()
                    
                    # Add prediction results to the conversation context
                    prediction_summary = self._format_prediction_summary(predictions)
                    response = f"Thank you for completing the DASS-21 assessment. {prediction_summary}"
                    
                    # Store assistant response
                    assistant_msg = Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=response
                    )
                    db.add(assistant_msg)
                    
                    # Update analytics
                    self._update_conversation_analytics(conversation_id, db)
                    
                except Exception as e:
                    response = f"I've collected all your responses, but there was an issue processing the assessment: {str(e)}"
                    assistant_msg = Message(
                        conversation_id=conversation.id,
                        role="assistant", 
                        content=response
                    )
                    db.add(assistant_msg)
            else:
                # Continue conversation
                response = self._get_assistant_response(conversation_id, db)
            
            db.commit()
            return response, is_complete, predictions
            
        finally:
            db.close()
    
    def _get_assistant_response(self, conversation_id: str, db: Session) -> str:
        """Get response from OpenAI assistant with conversation memory and tool calling support."""
        try:
            # Get conversation messages with memory limit (following OpenAI best practices)
            messages = self._get_conversation_context(conversation_id, db)
            
            # Add current progress to help the assistant
            progress_info = self._get_progress_info(conversation_id, db)
            if progress_info:
                messages.append({
                    "role": "system",
                    "content": f"Current progress: {progress_info}"
                })
            
            
            
            # Define tool for making DASS predictions
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "make_dass_prediction",
                        "description": "Make a DASS prediction when all 21 responses have been collected",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "responses": {
                                    "type": "object",
                                    "description": "Dictionary with keys Q1A-Q21A and integer values 1-4",
                                    "additionalProperties": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 4
                                    }
                                }
                            },
                            "required": ["responses"]
                        }
                    }
                }
            ]
            
            # Make API call - let OpenAI SDK handle retries automatically
            start_time = time.time()
            
            try:
                response = self.client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=1000,
                    temperature=0.7
                )
                
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    # Rate limit hit - return friendly message
                    error_response = "I'm temporarily experiencing high demand. Please try again in a few moments."
                    assistant_msg = Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=error_response
                    )
                    db.add(assistant_msg)
                    return error_response
                else:
                    # Different error, re-raise
                    raise e
            
            # Calculate token usage and log it
            token_count = response.usage.total_tokens if response.usage else None
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            
            print(f"💰 TOKEN USAGE: Input: {input_tokens}, Output: {output_tokens}, Total: {token_count}")
            if settings.OPENAI_MODEL.startswith("gpt-4"):
                estimated_cost = (input_tokens * 0.03 + output_tokens * 0.06) / 1000
                print(f"💰 ESTIMATED COST: ${estimated_cost:.4f}")
            elif settings.OPENAI_MODEL.startswith("gpt-3.5"):
                estimated_cost = (input_tokens * 0.001 + output_tokens * 0.002) / 1000
                print(f"💰 ESTIMATED COST: ${estimated_cost:.4f}")
            
            # Handle tool calls
            choice = response.choices[0]
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if tool_call.function.name == "make_dass_prediction":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            predictions = self._make_dass_prediction_from_args(
                                conversation_id, args["responses"], db
                            )
                            prediction_summary = self._format_prediction_summary(predictions)
                            assistant_response = f"Thank you for completing the DASS-21 assessment. {prediction_summary}"
                        except Exception as e:
                            assistant_response = f"I've collected your responses, but encountered an issue processing the assessment: {str(e)}"
            else:
                # Regular response
                assistant_response = choice.message.content
            
            # Store assistant response
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_response,
                token_count=token_count
            )
            db.add(assistant_msg)
            
            return assistant_response
            
        except Exception as e:
            error_response = "I apologize, but I'm having trouble responding right now. Please try again in a moment."
            
            # Store error response
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=error_response
            )
            db.add(assistant_msg)
            
            return error_response
    
    def _get_conversation_context(self, conversation_id: str, db: Session) -> List[Dict[str, str]]:
        """
        Get conversation context for OpenAI API, implementing memory management.
        Following OpenAI best practices for conversation memory.
        """
        # Get recent messages with limit to manage token usage
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).limit(settings.CONVERSATION_MEMORY_LIMIT).all()
        
        # Always include system message
        formatted_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = {
                    "role": msg.role,
                    "content": msg.content
                }
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Ensure system message is first
        result = []
        if system_message:
            result.append(system_message)
        
        # Add context summary if we're hitting memory limits
        if len(formatted_messages) >= settings.CONVERSATION_MEMORY_LIMIT - 5:
            context_summary = self._create_context_summary(conversation_id, db)
            if context_summary:
                result.append({
                    "role": "system",
                    "content": f"Previous conversation summary: {context_summary}"
                })
        
        result.extend(formatted_messages)
        return result
    
    def _create_context_summary(self, conversation_id: str, db: Session) -> str:
        """Create a summary of previous conversation context."""
        # Get collected responses so far
        responses = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).all()
        
        collected_count = len(responses)
        remaining_count = 21 - collected_count
        
        if collected_count > 0:
            collected_questions = [r.question_id for r in responses]
            return f"So far, you've collected responses for {collected_count} questions: {', '.join(collected_questions)}. {remaining_count} questions remaining."
        
        return "Assessment just started - no responses collected yet."
    
    def _extract_and_store_dass_responses(self, conversation_id: str, user_message: str, message_id: int, db: Session):
        """Extract DASS responses from user message and store them in database."""
        # Pattern to match responses like "1", "2", "3", "4" or "Never", "Sometimes", etc.
        number_patterns = re.findall(r'\b([1-4])\b', user_message)
        text_patterns = re.findall(r'\b(never|sometimes|often|almost always)\b', user_message.lower())
        
        # Convert text responses to numbers
        text_to_number = {
            'never': 1,
            'sometimes': 2,
            'often': 3,
            'almost always': 4
        }
        
        responses = []
        
        # Add number responses
        responses.extend([int(n) for n in number_patterns])
        
        # Add converted text responses
        responses.extend([text_to_number[t] for t in text_patterns])
        
        # Get currently collected responses to determine next question numbers
        existing_responses = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).order_by(DASSResponse.question_id).all()
        
        # Create set of already answered questions
        answered_questions = {r.question_id for r in existing_responses}
        
        # Assign responses to next unanswered questions
        for response_value in responses:
            # Find next unanswered question
            for i in range(1, 22):
                question_id = f"Q{i}A"
                if question_id not in answered_questions:
                    # Store this response
                    dass_response = DASSResponse(
                        conversation_id=conversation_id,
                        question_id=question_id,
                        response_value=response_value,
                        extracted_from_message_id=message_id
                    )
                    db.add(dass_response)
                    answered_questions.add(question_id)
                    break
    
    def _is_assessment_complete(self, conversation_id: str, db: Session) -> bool:
        """Check if all 21 DASS responses have been collected."""
        response_count = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).count()
        return response_count >= 21
    
    def _get_progress_info(self, conversation_id: str, db: Session) -> str:
        """Get current progress information."""
        collected = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).count()
        
        remaining = 21 - collected
        
        if collected == 0:
            return "No responses collected yet. Start with the first question."
        elif remaining > 0:
            return f"Collected {collected}/21 responses. Need {remaining} more responses."
        else:
            return "All 21 responses collected. Ready to make prediction."
    
    def _make_dass_prediction(self, conversation_id: str, db: Session) -> Dict[str, Any]:
        """Make DASS prediction using collected responses and store results."""
        # Get all responses for this conversation
        responses = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).all()
        
        # Convert to expected format
        response_dict = {r.question_id: r.response_value for r in responses}
        
        # Ensure we have all 21 responses
        for i in range(1, 22):
            question_id = f"Q{i}A"
            if question_id not in response_dict:
                raise ValueError(f"Missing response for {question_id}")
        
        return self._make_dass_prediction_from_args(conversation_id, response_dict, db)
    
    def _make_dass_prediction_from_args(self, conversation_id: str, responses: Dict[str, int], db: Session) -> Dict[str, Any]:
        """Make DASS prediction from response dictionary and store results."""
        start_time = time.time()
        
        # Validate responses
        for i in range(1, 22):
            question_id = f"Q{i}A"
            if question_id not in responses:
                raise ValueError(f"Missing response for {question_id}")
            if not isinstance(responses[question_id], int) or responses[question_id] < 1 or responses[question_id] > 4:
                raise ValueError(f"Invalid response for {question_id}: must be integer 1-4")
        
        # Add timestamp
        prediction_data = responses.copy()
        prediction_data['timestamp'] = datetime.now().isoformat()
        
        # Make prediction using DASS service
        result = dass_service.predict(prediction_data)
        
        if result['status'] != 'success':
            raise Exception(result.get('message', 'Prediction failed'))
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Convert numpy types to native Python types for database storage
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Clean the result to remove numpy types
        cleaned_result = convert_numpy_types(result)
        
        # Store prediction results
        prediction_record = Prediction(
            conversation_id=conversation_id,
            depression_category=cleaned_result['predictions']['Depression']['category_index'],
            depression_severity=cleaned_result['predictions']['Depression']['severity'],
            anxiety_category=cleaned_result['predictions']['Anxiety']['category_index'],
            anxiety_severity=cleaned_result['predictions']['Anxiety']['severity'],
            stress_category=cleaned_result['predictions']['Stress']['category_index'],
            stress_severity=cleaned_result['predictions']['Stress']['severity'],
            model_accuracy=float(cleaned_result['model_info']['accuracy']),  # Ensure it's a Python float
            model_type=str(cleaned_result['model_info']['model_type']),  # Ensure it's a Python string
            dataset_size=cleaned_result['model_info'].get('dataset_size'),
            processing_time_ms=processing_time_ms,
            raw_prediction_data=cleaned_result  # Store the cleaned result
        )
        db.add(prediction_record)
        db.commit()
        
        return result
    
    def _format_prediction_summary(self, predictions: Dict[str, Any]) -> str:
        """Format prediction results into a readable summary."""
        if not predictions or 'predictions' not in predictions:
            return "The assessment results are not available."
        
        results = predictions['predictions']
        summary_parts = []
        
        for target, result in results.items():
            severity = result.get('severity', 'Unknown')
            summary_parts.append(f"{target}: {severity}")
        
        summary = "Here are your assessment results:\n" + "\n".join(summary_parts)
        summary += "\n\nPlease remember that this is a screening tool and not a diagnosis. If you're experiencing distress, consider speaking with a mental health professional."
        
        return summary
    
    def _update_conversation_analytics(self, conversation_id: str, db: Session):
        """Update conversation analytics."""
        # Count messages and tokens
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).all()
        
        total_messages = len([m for m in messages if m.role != 'system'])
        total_tokens = sum(m.token_count for m in messages if m.token_count)
        
        # Calculate completion time
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        completion_time_minutes = None
        if conversation:
            time_diff = conversation.updated_at - conversation.created_at
            completion_time_minutes = time_diff.total_seconds() / 60
        
        # Count responses
        responses_collected = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).count()
        
        # Create or update analytics
        analytics = ConversationAnalytics(
            conversation_id=conversation_id,
            total_messages=total_messages,
            total_tokens_used=total_tokens,
            completion_time_minutes=completion_time_minutes,
            responses_collected_count=responses_collected,
            completion_rate=responses_collected / 21.0
        )
        db.add(analytics)
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history."""
        db = SessionLocal()
        try:
            messages = db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.role != 'system'  # Exclude system messages
            ).order_by(Message.timestamp).all()
            
            if not messages:
                raise ValueError("Conversation not found")
            
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "token_count": msg.token_count
                }
                for msg in messages
            ]
        finally:
            db.close()
    
    def get_conversation_with_responses(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation with DASS responses and predictions."""
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                raise ValueError("Conversation not found")
            
            messages = self.get_conversation(conversation_id)
            
            responses = db.query(DASSResponse).filter(
                DASSResponse.conversation_id == conversation_id
            ).order_by(DASSResponse.question_id).all()
            
            predictions = db.query(Prediction).filter(
                Prediction.conversation_id == conversation_id
            ).first()
            
            return {
                "conversation": {
                    "id": str(conversation.id),
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at,
                    "is_completed": conversation.is_completed
                },
                "messages": messages,
                "dass_responses": [
                    {
                        "question_id": r.question_id,
                        "response_value": r.response_value,
                        "timestamp": r.timestamp
                    }
                    for r in responses
                ],
                "predictions": {
                    "depression": {
                        "category": predictions.depression_category,
                        "severity": predictions.depression_severity
                    },
                    "anxiety": {
                        "category": predictions.anxiety_category,
                        "severity": predictions.anxiety_severity
                    },
                    "stress": {
                        "category": predictions.stress_category,
                        "severity": predictions.stress_severity
                    },
                    "processing_time_ms": predictions.processing_time_ms,
                    "timestamp": predictions.timestamp
                } if predictions else None
            }
        finally:
            db.close()
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all related data."""
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if conversation:
                db.delete(conversation)  # Cascade will delete related records
                db.commit()
                return True
            return False
        finally:
            db.close()
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics and metrics."""
        db = SessionLocal()
        try:
            # Get overall conversation statistics
            total_conversations = db.query(Conversation).count()
            completed_conversations = db.query(Conversation).filter(
                Conversation.is_completed == True
            ).count()
            
            completion_rate = completed_conversations / total_conversations if total_conversations > 0 else 0
            
            # Get analytics data
            analytics_data = db.query(ConversationAnalytics).all()
            
            if analytics_data:
                avg_completion_rate = sum(a.completion_rate for a in analytics_data) / len(analytics_data)
                completion_times = [a.completion_time_minutes for a in analytics_data if a.completion_time_minutes]
                avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
            else:
                avg_completion_rate = 0
                avg_completion_time = 0
            
            return {
                "total_conversations": total_conversations,
                "completed_conversations": completed_conversations,
                "completion_rate": completion_rate,
                "average_assessment_completion_rate": avg_completion_rate,
                "average_completion_time_minutes": avg_completion_time
            }
            
        finally:
            db.close()

    def get_conversations_list(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get list of conversations with summary info (without messages) for sidebar."""
        db = SessionLocal()
        try:
            conversations = db.query(Conversation).order_by(
                desc(Conversation.updated_at)
            ).limit(limit).offset(offset).all()
            
            result = []
            for conv in conversations:
                # Get the last non-system message for preview
                last_message = db.query(Message).filter(
                    Message.conversation_id == conv.id,
                    Message.role != 'system'
                ).order_by(desc(Message.timestamp)).first()
                
                # Get message count (excluding system messages)
                message_count = db.query(Message).filter(
                    Message.conversation_id == conv.id,
                    Message.role != 'system'
                ).count()
                
                # Check if has predictions
                has_predictions = db.query(Prediction).filter(
                    Prediction.conversation_id == conv.id
                ).count() > 0
                
                # Generate title from first user message or use default
                first_user_message = db.query(Message).filter(
                    Message.conversation_id == conv.id,
                    Message.role == 'user'
                ).order_by(Message.timestamp).first()
                
                if first_user_message and len(first_user_message.content) > 10:
                    title = first_user_message.content[:50].strip()
                    if len(first_user_message.content) > 50:
                        title += "..."
                else:
                    title = f"Chat {str(conv.id)[:8]}"
                
                conversation_data = {
                    "id": str(conv.id),
                    "title": title,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "is_completed": conv.is_completed,
                    "message_count": message_count,
                    "last_message": last_message.content if last_message else "No messages",
                    "last_message_timestamp": last_message.timestamp.isoformat() if last_message else conv.created_at.isoformat(),
                    "has_predictions": has_predictions
                }
                
                result.append(conversation_data)
            
            return result
            
        finally:
            db.close()

# Global instance
conversation_service = ConversationService() 