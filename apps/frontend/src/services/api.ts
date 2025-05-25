import axios, { AxiosInstance, AxiosError } from "axios";
import {
  CreateConversationRequest,
  CreateConversationResponse,
  ContinueConversationRequest,
  ContinueConversationResponse,
  FullConversationResponse,
  ConversationsListResponse,
  AnalyticsResponse,
  HealthResponse,
  APIError,
} from "../types/api";

class APIService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: "http://localhost:8000/api",
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(
          `Making ${config.method?.toUpperCase()} request to ${config.url}`
        );
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        return response;
      },
      (error: AxiosError<APIError>) => {
        if (error.response?.data?.detail) {
          throw new Error(error.response.data.detail);
        }
        throw new Error(error.message || "An unexpected error occurred");
      }
    );
  }

  // Conversation endpoints
  async createConversation(
    data: CreateConversationRequest
  ): Promise<CreateConversationResponse> {
    const response = await this.client.post<CreateConversationResponse>(
      "/conversations/new",
      data
    );
    return response.data;
  }

  async continueConversation(
    data: ContinueConversationRequest
  ): Promise<ContinueConversationResponse> {
    const response = await this.client.post<ContinueConversationResponse>(
      "/conversations/continue",
      data
    );
    return response.data;
  }

  async getConversationsList(
    limit: number = 50,
    offset: number = 0
  ): Promise<ConversationsListResponse> {
    const response = await this.client.get<ConversationsListResponse>(
      `/conversations?limit=${limit}&offset=${offset}`
    );
    return response.data;
  }

  async getConversation(
    conversationId: string
  ): Promise<FullConversationResponse> {
    const response = await this.client.get<FullConversationResponse>(
      `/conversations/${conversationId}/full`
    );
    return response.data;
  }

  async deleteConversation(conversationId: string): Promise<void> {
    await this.client.delete(`/conversations/${conversationId}`);
  }

  // Analytics endpoints
  async getAnalytics(): Promise<AnalyticsResponse> {
    const response = await this.client.get<AnalyticsResponse>("/analytics");
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>("/health");
    return response.data;
  }

  // Model retraining
  async retrainModel(): Promise<{
    status: string;
    message: string;
    training_samples: number;
    complete_conversations: number;
    total_responses: number;
    new_accuracy: string | number;
    timestamp: string;
  }> {
    const response = await this.client.post("/model/retrain");
    return response.data;
  }
}

export const apiService = new APIService();
