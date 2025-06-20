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
            max_retries=0  
        )
        
    def create_conversation(self, initial_message: str = None, user_ip: str = None, user_agent: str = None) -> Tuple[str, str]:
        db = SessionLocal()
        try:
            conversation = Conversation(
                user_ip=user_ip,
                user_agent=user_agent
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            
            conversation_id = str(conversation.id)
            
            system_message = Message(
                conversation_id=conversation.id,
                role="system",
                content=settings.DASS_SYSTEM_PROMPT
            )
            db.add(system_message)
            
            if initial_message:
                user_message = Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=initial_message
                )
                db.add(user_message)
            
            db.commit()
            response = self._get_assistant_response(conversation_id, db)
            return conversation_id, response
            
        finally:
            db.close()
    
    def continue_conversation(self, conversation_id: str, user_message: str) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
        """Continue conversation and return (response, is_complete, predictions)"""
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                raise ValueError("Conversation not found")
            
            user_msg = Message(
                conversation_id=conversation.id,
                role="user",
                content=user_message
            )
            db.add(user_msg)
            db.commit()
            db.refresh(user_msg)
            
            stored_responses_count = self._extract_and_store_dass_responses(conversation_id, user_message, user_msg.id, db)
            
            is_complete = self._is_assessment_complete(conversation_id, db)
            predictions = None
            
            existing_prediction = db.query(Prediction).filter(
                Prediction.conversation_id == conversation_id
            ).first()
            
            if existing_prediction:
                predictions = existing_prediction.raw_prediction_data
                response = self._get_assistant_response(conversation_id, db)
                
            elif not is_complete and stored_responses_count == 0:
                progress_info = self._get_progress_info(conversation_id, db)
                next_question_marker = "🎯 NEXT QUESTION TO ASK: "
                if next_question_marker in progress_info:
                    question_to_ask = progress_info.split(next_question_marker)[1].split(". ⚠️")[0]
                else:
                    question_texts = self._get_question_texts()
                    question_to_ask = question_texts.get(1, "the first question")

                response = f"I'm sorry, but that doesn't seem to be a valid answer. Please respond with 'Never', 'Sometimes', 'Often', or 'Almost Always', or a number from 1 to 4. Let's try again: {question_to_ask}"
                assistant_msg = Message(
                    conversation_id=conversation.id,
                    role="assistant", 
                    content=response
                )
                db.add(assistant_msg)

            elif is_complete:
                try:
                    predictions = self._make_dass_prediction(conversation_id, db)
                    
                    conversation.is_completed = True
                    conversation.updated_at = datetime.utcnow()
                    db.commit()
                    
                    self._update_conversation_analytics(conversation_id, db)
                    response = self._get_assistant_response(conversation_id, db)
                    
                except Exception as e:
                    response = f"I've collected all your responses, but there was an issue processing the assessment: {str(e)}"
                    assistant_msg = Message(
                        conversation_id=conversation.id,
                        role="assistant", 
                        content=response
                    )
                    db.add(assistant_msg)
            else:
                response = self._get_assistant_response(conversation_id, db)
            
            db.commit()
            return response, is_complete, predictions
            
        finally:
            db.close()
    
    def _get_assistant_response(self, conversation_id: str, db: Session) -> str:
        """Get response from OpenAI assistant with conversation memory and tool calling support."""
        try:
            messages = self._get_conversation_context(conversation_id, db)
            
            existing_prediction = db.query(Prediction).filter(
                Prediction.conversation_id == conversation_id
            ).first()
            
            last_assistant_message = db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.role == "assistant"
            ).order_by(Message.timestamp.desc()).first()
            
            completion_already_acknowledged = False
            if last_assistant_message and existing_prediction:
                completion_keywords = ["assessment", "completed", "results", "dass"]
                completion_already_acknowledged = any(
                    keyword in last_assistant_message.content.lower() 
                    for keyword in completion_keywords
                )
            
            if existing_prediction and not completion_already_acknowledged:
                results_context = self._format_results_context(existing_prediction)
                messages.append({
                    "role": "system",
                    "content": f"The user has completed their DASS-21 assessment. Results: {results_context}. You are now in post-assessment therapeutic conversation mode. Respond naturally to the user's messages and engage in supportive dialogue about their mental health."
                })
            elif not existing_prediction:
                progress_info = self._get_progress_info(conversation_id, db)
                if progress_info:
                    messages.insert(1, {
                        "role": "system",
                        "content": f"🔍 CRITICAL TRACKING INFO: {progress_info}"
                    })
            
            tools = []
            if not existing_prediction:
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

            try:
                if tools:
                    response = self.client.chat.completions.create(
                        model=settings.OPENAI_MODEL,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=1000,
                        temperature=0.7
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=settings.OPENAI_MODEL,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    error_response = "I'm temporarily experiencing high demand. Please try again in a few moments."
                    assistant_msg = Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=error_response
                    )
                    db.add(assistant_msg)
                    return error_response
                else:
                    raise e
            
            token_count = response.usage.total_tokens if response.usage else None
        
            choice = response.choices[0]
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if tool_call.function.name == "make_dass_prediction":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            predictions = self._make_dass_prediction_from_args(
                                conversation_id, args["responses"], db
                            )
                            assistant_response = self._get_natural_completion_response(conversation_id, db)
                        except Exception as e:
                            assistant_response = f"I've collected your responses, but encountered an issue processing the assessment: {str(e)}"
            else:
                assistant_response = choice.message.content
            
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
            
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=error_response
            )
            db.add(assistant_msg)
            
            return error_response
    
    def _get_natural_completion_response(self, conversation_id: str, db: Session) -> str:
        """Get a natural AI response after assessment completion."""
        try:
            messages = self._get_conversation_context(conversation_id, db)
            
            messages.append({
                "role": "system",
                "content": "The user has just completed their DASS-21 assessment. Provide a natural, empathetic response acknowledging the completion and offering to discuss the results. Be supportive and encourage ongoing conversation."
            })
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            return assistant_response
            
        except Exception as e:
            return "Thank you for completing the assessment. I'm here to discuss your results and provide support. How are you feeling about what we've covered?"
    
    def _get_conversation_context(self, conversation_id: str, db: Session) -> List[Dict[str, str]]:
        """Get conversation context for OpenAI API, implementing memory management."""
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).limit(settings.CONVERSATION_MEMORY_LIMIT).all()
        
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
        
        result = []
        if system_message:
            result.append(system_message)
        
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
        cleaned_message = user_message.strip().lower()
        
        print(f"DEBUG: Processing message: '{user_message}' (cleaned: '{cleaned_message}')")
        
        number_patterns = re.findall(r'\b([1-4])\b', user_message)
        
        text_patterns = []
        if 'never' in cleaned_message and 'sometimes' not in cleaned_message:
            text_patterns.append('never')
        elif 'sometimes' in cleaned_message:
            text_patterns.append('sometimes')
        elif 'often' in cleaned_message and 'almost' not in cleaned_message:
            text_patterns.append('often')
        elif 'almost always' in cleaned_message or ('almost' in cleaned_message and 'always' in cleaned_message):
            text_patterns.append('almost always')
        
        print(f"DEBUG: Found number patterns: {number_patterns}")
        print(f"DEBUG: Found text patterns: {text_patterns}")
        
        text_to_number = {
            'never': 1,
            'sometimes': 2,
            'often': 3,
            'almost always': 4
        }
        
        responses = []
        responses.extend([int(n) for n in number_patterns])
        responses.extend([text_to_number[t] for t in text_patterns])
        
        print(f"DEBUG: Final extracted responses: {responses}")
        
        existing_responses = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).order_by(DASSResponse.question_id).all()
        
        answered_questions = {r.question_id for r in existing_responses}
        print(f"DEBUG: Already answered questions: {sorted(answered_questions)}")
        
        stored_count = 0
        if responses:
            for response_value in responses:
                for i in range(1, 22):
                    question_id = f"Q{i}A"
                    if question_id not in answered_questions:
                        existing_check = db.query(DASSResponse).filter(
                            DASSResponse.conversation_id == conversation_id,
                            DASSResponse.question_id == question_id
                        ).first()
                        
                        if not existing_check:
                            dass_response = DASSResponse(
                                conversation_id=conversation_id,
                                question_id=question_id,
                                response_value=response_value,
                                extracted_from_message_id=message_id
                            )
                            db.add(dass_response)
                            answered_questions.add(question_id)
                            stored_count += 1
                            print(f"DEBUG: Stored response {response_value} for {question_id} from message: '{user_message[:50]}...'")
                        else:
                            print(f"DEBUG: Question {question_id} already answered, skipping duplicate")
                        break
            
            if stored_count > 0:
                db.commit()
                print(f"DEBUG: Successfully stored {stored_count} new responses")
            else:
                print(f"DEBUG: No new responses stored (all questions already answered)")
        else:
            print(f"DEBUG: No valid responses found in message: '{user_message[:50]}...'")
        
        db.commit()
        return stored_count
    
    def _is_assessment_complete(self, conversation_id: str, db: Session) -> bool:
        """Check if all 21 DASS responses have been collected."""
        response_count = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).count()
        return response_count >= 21
    
    def _get_progress_info(self, conversation_id: str, db: Session) -> str:
        """Get current progress information with detailed question tracking."""
        responses = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).order_by(DASSResponse.question_id).all()
        
        collected = len(responses)
        remaining = 21 - collected
        
        if collected == 0:
            return "🔍 ASSESSMENT STATUS: Starting assessment. 🎯 Ask question Q1A: 'I found it hard to wind down'"
        elif remaining > 0:
            answered_questions = {r.question_id for r in responses}
            
            next_question_num = None
            for i in range(1, 22):
                question_id = f"Q{i}A"
                if question_id not in answered_questions:
                    next_question_num = i
                    break
            
            question_texts = {
                1: "I found it hard to wind down",
                2: "I was aware of dryness of my mouth", 
                3: "I couldn't seem to experience any positive feeling at all",
                4: "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)",
                5: "I found it difficult to work up the initiative to do things",
                6: "I tended to over-react to situations",
                7: "I experienced trembling (e.g. in the hands)",
                8: "I felt that I was using a lot of nervous energy",
                9: "I was worried about situations in which I might panic and make a fool of myself",
                10: "I felt that I had nothing to look forward to",
                11: "I found myself getting agitated",
                12: "I found it difficult to relax",
                13: "I felt down-hearted and blue",
                14: "I was intolerant of anything that kept me from getting on with what I was doing",
                15: "I felt I was close to panic",
                16: "I was unable to become enthusiastic about anything",
                17: "I felt I wasn't worth much as a person",
                18: "I felt that I was rather touchy",
                19: "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
                20: "I felt scared without any good reason",
                21: "I felt that life was meaningless"
            }
            
            answered_nums = sorted([int(q[1:-1]) for q in answered_questions])
            last_answered_num = max(answered_nums) if answered_nums else 0
            last_answered_text = question_texts.get(last_answered_num, "Unknown")
            
            if next_question_num:
                next_question_text = question_texts.get(next_question_num, "Unknown question")
                
                if last_answered_num > 0:
                    return f"🔍 ASSESSMENT STATUS: Collected {collected}/21 responses. ✅ LAST ANSWERED: Q{last_answered_num}A: '{last_answered_text}'. ✅ All answered questions: {answered_nums}. 🎯 NEXT QUESTION TO ASK: Q{next_question_num}A: '{next_question_text}'. ⚠️ IMPORTANT: Acknowledge the user's response to Q{last_answered_num}A first, then ask Q{next_question_num}A."
                else:
                    return f"🔍 ASSESSMENT STATUS: Collected {collected}/21 responses. ✅ Answered questions: {answered_nums}. 🎯 NEXT QUESTION TO ASK: Q{next_question_num}A: '{next_question_text}'. ⚠️ IMPORTANT: Do NOT ask questions that are already in the answered list above. ASK ONLY THE NEXT QUESTION SPECIFIED."
            else:
                return "🔍 ASSESSMENT STATUS: All 21 responses collected. 🎯 Ready to make prediction."
        else:
            return "🔍 ASSESSMENT STATUS: All 21 responses collected. 🎯 Ready to make prediction."
    
    def _make_dass_prediction(self, conversation_id: str, db: Session) -> Dict[str, Any]:
        """Make DASS prediction using collected responses and store results."""
        responses = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).all()
        
        response_dict = {r.question_id: r.response_value for r in responses}
        
        for i in range(1, 22):
            question_id = f"Q{i}A"
            if question_id not in response_dict:
                raise ValueError(f"Missing response for {question_id}")
        
        return self._make_dass_prediction_from_args(conversation_id, response_dict, db)
    
    def _make_dass_prediction_from_args(self, conversation_id: str, responses: Dict[str, int], db: Session) -> Dict[str, Any]:
        """Make DASS prediction from response dictionary and store results."""
        start_time = time.time()
        
        for i in range(1, 22):
            question_id = f"Q{i}A"
            if question_id not in responses:
                raise ValueError(f"Missing response for {question_id}")
            if not isinstance(responses[question_id], int) or responses[question_id] < 1 or responses[question_id] > 4:
                raise ValueError(f"Invalid response for {question_id}: must be integer 1-4")
        
        prediction_data = responses.copy()
        prediction_data['timestamp'] = datetime.now().isoformat()
        
        result = dass_service.predict(prediction_data)
        
        if result['status'] != 'success':
            raise Exception(result.get('message', 'Prediction failed'))
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
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
        
        cleaned_result = convert_numpy_types(result)
        
        prediction_record = Prediction(
            conversation_id=conversation_id,
            depression_category=cleaned_result['predictions']['Depression']['category_index'],
            depression_severity=cleaned_result['predictions']['Depression']['severity'],
            anxiety_category=cleaned_result['predictions']['Anxiety']['category_index'],
            anxiety_severity=cleaned_result['predictions']['Anxiety']['severity'],
            stress_category=cleaned_result['predictions']['Stress']['category_index'],
            stress_severity=cleaned_result['predictions']['Stress']['severity'],
            model_accuracy=float(cleaned_result['model_info']['accuracy']),
            model_type=str(cleaned_result['model_info']['model_type']),
            dataset_size=cleaned_result['model_info'].get('dataset_size'),
            processing_time_ms=processing_time_ms,
            raw_prediction_data=cleaned_result
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
    
    def _format_results_context(self, prediction_record: 'Prediction') -> str:
        """Format prediction results for AI context when discussing completed assessments."""
        context_parts = []
        context_parts.append(f"Depression: {prediction_record.depression_severity}")
        context_parts.append(f"Anxiety: {prediction_record.anxiety_severity}")
        context_parts.append(f"Stress: {prediction_record.stress_severity}")
        context_parts.append(f"Model accuracy: {prediction_record.model_accuracy:.2f}")
        
        return "; ".join(context_parts)
    
    def _update_conversation_analytics(self, conversation_id: str, db: Session):
        """Update conversation analytics."""
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).all()
        
        total_messages = len([m for m in messages if m.role != 'system'])
        total_tokens = sum(m.token_count for m in messages if m.token_count)
        
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        completion_time_minutes = None
        if conversation:
            time_diff = conversation.updated_at - conversation.created_at
            completion_time_minutes = time_diff.total_seconds() / 60
        
        responses_collected = db.query(DASSResponse).filter(
            DASSResponse.conversation_id == conversation_id
        ).count()
        
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
                Message.role != 'system'
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
                db.delete(conversation)
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            print(f"Error deleting conversation {conversation_id}: {e}")
            return False
        finally:
            db.close()
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics and metrics."""
        db = SessionLocal()
        try:
            total_conversations = db.query(Conversation).count()
            completed_conversations = db.query(Conversation).filter(
                Conversation.is_completed == True
            ).count()
            
            completion_rate = completed_conversations / total_conversations if total_conversations > 0 else 0
            
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
        """Get list of conversations with summary info for sidebar."""
        db = SessionLocal()
        try:
            conversations = db.query(Conversation).order_by(
                desc(Conversation.updated_at)
            ).limit(limit).offset(offset).all()
            
            result = []
            for conv in conversations:
                last_message = db.query(Message).filter(
                    Message.conversation_id == conv.id,
                    Message.role != 'system'
                ).order_by(desc(Message.timestamp)).first()
                
                message_count = db.query(Message).filter(
                    Message.conversation_id == conv.id,
                    Message.role != 'system'
                ).count()
                
                has_predictions = db.query(Prediction).filter(
                    Prediction.conversation_id == conv.id
                ).count() > 0
                
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

    def _get_question_texts(self) -> Dict[int, str]:
        return {
            1: "I found it hard to wind down",
            2: "I was aware of dryness of my mouth", 
            3: "I couldn't seem to experience any positive feeling at all",
            4: "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)",
            5: "I found it difficult to work up the initiative to do things",
            6: "I tended to over-react to situations",
            7: "I experienced trembling (e.g. in the hands)",
            8: "I felt that I was using a lot of nervous energy",
            9: "I was worried about situations in which I might panic and make a fool of myself",
            10: "I felt that I had nothing to look forward to",
            11: "I found myself getting agitated",
            12: "I found it difficult to relax",
            13: "I felt down-hearted and blue",
            14: "I was intolerant of anything that kept me from getting on with what I was doing",
            15: "I felt I was close to panic",
            16: "I was unable to become enthusiastic about anything",
            17: "I felt I wasn't worth much as a person",
            18: "I felt that I was rather touchy",
            19: "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
            20: "I felt scared without any good reason",
            21: "I felt that life was meaningless"
        }

conversation_service = ConversationService() 