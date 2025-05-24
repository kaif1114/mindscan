// API Request/Response Types
export interface CreateConversationRequest {
  initial_message: string;
}

export interface CreateConversationResponse {
  conversation_id: string;
  message: string;
}

export interface ContinueConversationRequest {
  conversation_id: string;
  message: string;
}

export interface ContinueConversationResponse {
  conversation_id: string;
  message: string;
  is_assessment_complete: boolean;
  predictions: DASSPrediction | null;
}

export interface ConversationMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
}

export interface Conversation {
  id: string;
  created_at: string;
  updated_at: string;
  is_completed: boolean;
  title?: string | null;
}

export interface FullConversationResponse {
  conversation: Conversation;
  messages: ConversationMessage[] | null;
  responses: DASSResponse[] | null;
  predictions: DASSPrediction[] | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  is_completed: boolean;
  message_count: number;
  last_message: string;
  last_message_timestamp: string;
  has_predictions: boolean;
}

export interface ConversationsListResponse {
  conversations: ConversationSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface DASSResponse {
  id: string;
  conversation_id: string;
  question_id: string;
  response_value: number;
  created_at: string;
}

export interface DASSPrediction {
  id: string;
  conversation_id: string;
  depression_score: number;
  anxiety_score: number;
  stress_score: number;
  depression_level: string;
  anxiety_level: string;
  stress_level: string;
  has_disorder: boolean;
  disorder_type: string | null;
  confidence: number;
  model_version: string;
  created_at: string;
}

export interface AnalyticsResponse {
  total_conversations: number;
  completed_conversations: number;
  completion_rate: number;
  average_assessment_completion_rate: number;
  average_completion_time_minutes: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  model_status: {
    dass_model_loaded: boolean;
    model_version: string;
  };
  database_status: {
    connected: boolean;
    active_conversations: number;
  };
}

// Error types
export interface APIError {
  detail: string;
}
