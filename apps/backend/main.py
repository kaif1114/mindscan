from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import Dict, Any

from config import settings
from database import create_tables, get_db
from models import (
    ConversationRequest, ConversationResponse,
    NewConversationRequest, NewConversationResponse,
    ConversationsListResponse,
    DASSPredictionRequest, DASSPredictionResponse,
    HealthCheckResponse, ModelInfoResponse, ErrorResponse
)
from services.conversation_service import conversation_service
from src.dass_prediction_service import dass_service

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DASS Mental Health Assessment API",
    description="Conversational DASS-21 mental health assessment with OpenAI integration and PostgreSQL persistence",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        create_tables()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "DASS Mental Health Assessment API",
        "version": "2.1.0",
        "docs": "/api/docs",
        "health": "/api/health",
        "features": [
            "Conversational DASS assessment",
            "PostgreSQL persistence", 
            "OpenAI integration",
            "Analytics and reporting"
        ]
    }

@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        test_data = {f"Q{i}A": 2 for i in range(1, 22)}
        result = dass_service.predict(test_data)
        
        from database import SessionLocal
        db = SessionLocal()
        try:
            db.execute("SELECT 1")
            db_status = "operational"
        except Exception as e:
            db_status = f"error: {str(e)}"
        finally:
            db.close()
        
        if result['status'] == 'success' and db_status == "operational":
            return HealthCheckResponse(
                status="healthy",
                dass_model="operational",
                accuracy=dass_service.metadata['accuracy'],
                timestamp=datetime.now()
            )
        else:
            return JSONResponse(
                status_code=503,
                content=HealthCheckResponse(
                    status="degraded",
                    dass_model="not responding correctly",
                    timestamp=datetime.now()
                ).dict()
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content=HealthCheckResponse(
                status="unhealthy",
                dass_model="error",
                timestamp=datetime.now()
            ).dict()
        )

@app.get("/api/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    try:
        return ModelInfoResponse(
            status="success",
            model_info={
                "name": "DASS-21 Mental Health Assessment Model",
                "type": dass_service.metadata['model_type'],
                "accuracy": dass_service.metadata['accuracy'],
                "features": dass_service.metadata['features'],
                "dataset_size": dass_service.metadata['dataset_size'],
                "targets": dass_service.target_names,
                "severity_levels": dass_service.severity_labels,
                "description": "Clinically validated Depression, Anxiety & Stress Scale with 21 items",
                "conversation_enabled": True,
                "openai_model": settings.OPENAI_MODEL,
                "persistent_storage": True,
                "database": "PostgreSQL"
            }
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations/new", response_model=NewConversationResponse)
async def create_new_conversation(request: NewConversationRequest, req: Request):
    try:
        user_ip = req.client.host if req.client else None
        user_agent = req.headers.get("user-agent")
        
        conversation_id, response = conversation_service.create_conversation(
            request.initial_message,
            user_ip=user_ip,
            user_agent=user_agent
        )
        
        return NewConversationResponse(
            conversation_id=conversation_id,
            message=response
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations/continue", response_model=ConversationResponse)
async def continue_conversation(request: ConversationRequest):
    try:
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id is required")
        
        response, is_complete, predictions = conversation_service.continue_conversation(
            request.conversation_id,
            request.message
        )
        
        return ConversationResponse(
            conversation_id=request.conversation_id,
            message=response,
            is_assessment_complete=is_complete,
            predictions=predictions
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error continuing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations", response_model=ConversationsListResponse)
async def get_conversations_list(limit: int = 50, offset: int = 0):
    try:
        conversations = conversation_service.get_conversations_list(limit=limit, offset=offset)
        return ConversationsListResponse(
            conversations=conversations,
            total=len(conversations),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error getting conversations list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history (excluding system messages)."""
    try:
        messages = conversation_service.get_conversation(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "message_count": len(messages)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}/full")
async def get_conversation_with_responses(conversation_id: str):
    """Get complete conversation data including responses and predictions."""
    try:
        conversation_data = conversation_service.get_conversation_with_responses(conversation_id)
        return conversation_data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting conversation with responses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        success = conversation_service.delete_conversation(conversation_id)
        if success:
            return {"message": "Conversation deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def get_conversation_analytics():
    """Get overall conversation analytics."""
    try:
        analytics = conversation_service.get_conversation_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/dass", response_model=DASSPredictionResponse)
async def predict_dass(request: DASSPredictionRequest):
    """Make DASS prediction directly (without conversation)."""
    try:
        # Validate that we have all 21 responses
        for i in range(1, 22):
            question_id = f"Q{i}A"
            if question_id not in request.responses:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing response for {question_id}"
                )
        
     
        prediction_data = request.responses.copy()
        prediction_data['timestamp'] = datetime.now().isoformat()
        
    
        result = dass_service.predict(prediction_data)
        
        if result['status'] == 'success':
            return DASSPredictionResponse(
                status=result['status'],
                predictions={
                    k: {
                        "category_index": v['category_index'],
                        "severity": v['severity']
                    }
                    for k, v in result['predictions'].items()
                },
                model_info=result['model_info']
            )
        else:
            return DASSPredictionResponse(
                status=result['status'],
                message=result.get('message', 'Prediction failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making DASS prediction: {e}")
        return DASSPredictionResponse(
            status="error",
            message=f"Prediction failed: {str(e)}"
        )

@app.post("/api/predict/dass/detailed", response_model=DASSPredictionResponse)
async def predict_dass_detailed(request: DASSPredictionRequest):
    """Make DASS prediction with probabilities (if available)."""
    try:
        # Validate that we have all 21 responses
        for i in range(1, 22):
            question_id = f"Q{i}A"
            if question_id not in request.responses:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing response for {question_id}"
                )
        
        
        prediction_data = request.responses.copy()
        prediction_data['timestamp'] = datetime.now().isoformat()
        
     
        result = dass_service.predict_with_probabilities(prediction_data)
        
        if result['status'] == 'success':
            return DASSPredictionResponse(
                status=result['status'],
                predictions={
                    k: {
                        "category_index": v['category_index'],
                        "severity": v['severity'],
                        "probabilities": v.get('probabilities')
                    }
                    for k, v in result['predictions'].items()
                },
                model_info=result['model_info']
            )
        else:
            return DASSPredictionResponse(
                status=result['status'],
                message=result.get('message', 'Detailed prediction failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making detailed DASS prediction: {e}")
        return DASSPredictionResponse(
            status="error",
            message=f"Detailed prediction failed: {str(e)}"
        )

@app.get("/api/dass/questions")
async def get_dass_questions():
    """Get all DASS-21 questions for reference."""
    try:
        return dass_service.get_questions()
    except Exception as e:
        logger.error(f"Error getting DASS questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/retrain")
async def retrain_model():
    """Retrain the DASS model using conversation response data."""
    try:
        from database import SessionLocal, DASSResponse
        import pandas as pd
        import subprocess
        import os
        
        logger.info("Starting model retraining process...")
        
       
        db = SessionLocal()
        try:
            responses = db.query(DASSResponse).all()
            if len(responses) < 50:  
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient data for retraining. Need at least 50 responses, got {len(responses)}"
                )
            
           
            training_data = []
            conversation_responses = {}
            
            for response in responses:
                conv_id = str(response.conversation_id)
                if conv_id not in conversation_responses:
                    conversation_responses[conv_id] = {}
                conversation_responses[conv_id][response.question_id] = response.response_value
            
      
            complete_conversations = []
            for conv_id, responses_dict in conversation_responses.items():
                if len(responses_dict) == 21:  
                    complete_conversations.append(responses_dict)
            
            if len(complete_conversations) < 20:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient complete conversations for retraining. Need at least 20, got {len(complete_conversations)}"
                )
            
          
            df_data = []
            for conv_responses in complete_conversations:
                row = {}
            
                for i in range(1, 22):
                    question_key = f"Q{i}A"
                    row[question_key] = conv_responses.get(question_key, 0)
                
    
                depression_score = sum(conv_responses.get(f"Q{i}A", 0) for i in [3, 5, 10, 13, 16, 17, 21])
                anxiety_score = sum(conv_responses.get(f"Q{i}A", 0) for i in [2, 4, 7, 9, 15, 19, 20])
                stress_score = sum(conv_responses.get(f"Q{i}A", 0) for i in [1, 6, 8, 11, 12, 14, 18])
                
          
                row['Depression_Category'] = min(4, depression_score // 5)
                row['Anxiety_Category'] = min(4, anxiety_score // 4)
                row['Stress_Category'] = min(4, stress_score // 6)
                
                df_data.append(row)
            
         
            new_df = pd.DataFrame(df_data)
            backup_path = './data/dataset_backup.csv'
            new_data_path = './data/conversation_training_data.csv'
            
            # Backup existing dataset
            if os.path.exists('./data/dataset.csv'):
                import shutil
                shutil.copy('./data/dataset.csv', backup_path)
            
           
            new_df.to_csv(new_data_path, index=False)
            
        
            if os.path.exists('./data/dataset.csv'):
                existing_df = pd.read_csv('./data/dataset.csv')
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv('./data/dataset.csv', index=False)
            else:
                new_df.to_csv('./data/dataset.csv', index=False)
            
            logger.info(f"Prepared training data with {len(df_data)} new samples")
            
            # Run training script
            result = subprocess.run(
                ['python', 'train_model.py'],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=300  
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                raise HTTPException(status_code=500, detail=f"Model training failed: {result.stderr}")
            
    
            dass_service.load_model()
            
            logger.info("Model retraining completed successfully")
            
            return {
                "status": "success",
                "message": "Model retrained successfully",
                "training_samples": len(df_data),
                "complete_conversations": len(complete_conversations),
                "total_responses": len(responses),
                "new_accuracy": dass_service.metadata.get('accuracy', 'Unknown'),
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            message="Endpoint not found",
            details={
                "available_endpoints": {
                    "health": "/api/health",
                    "model_info": "/api/model/info",
                    "new_conversation": "/api/conversations/new",
                    "continue_conversation": "/api/conversations/continue",
                    "conversation_history": "/api/conversations/{id}",
                    "full_conversation": "/api/conversations/{id}/full",
                    "analytics": "/api/analytics",
                    "predict_dass": "/api/predict/dass",
                    "predict_dass_detailed": "/api/predict/dass/detailed",
                    "dass_questions": "/api/dass/questions",
                    "docs": "/api/docs"
                }
            }
        ).dict()
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(message="Internal server error").dict()
    )

if __name__ == "__main__":
    import uvicorn
    
  
  
  
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    ) 
