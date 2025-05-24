import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiService } from "../services/api";
import {
  CreateConversationRequest,
  ContinueConversationRequest,
  FullConversationResponse,
} from "../types/api";

// Query keys
export const conversationKeys = {
  all: ["conversations"] as const,
  lists: () => [...conversationKeys.all, "list"] as const,
  list: (limit: number, offset: number) =>
    [...conversationKeys.lists(), { limit, offset }] as const,
  conversation: (id: string) => ["conversations", id] as const,
  analytics: ["analytics"] as const,
  health: ["health"] as const,
};

// Create new conversation
export const useCreateConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateConversationRequest) =>
      apiService.createConversation(data),
    onSuccess: () => {
      // Invalidate conversations list and analytics
      queryClient.invalidateQueries({ queryKey: conversationKeys.lists() });
      queryClient.invalidateQueries({ queryKey: conversationKeys.analytics });
    },
    onError: (error) => {
      console.error("Failed to create conversation:", error);
    },
  });
};

// Continue conversation
export const useContinueConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ContinueConversationRequest) =>
      apiService.continueConversation(data),
    onSuccess: (data) => {
      // Invalidate the specific conversation to refetch updated data
      queryClient.invalidateQueries({
        queryKey: conversationKeys.conversation(data.conversation_id),
      });

      // Invalidate conversations list to update last message/timestamp
      queryClient.invalidateQueries({ queryKey: conversationKeys.lists() });

      // Invalidate analytics if assessment is complete
      if (data.is_assessment_complete) {
        queryClient.invalidateQueries({ queryKey: conversationKeys.analytics });
      }
    },
    onError: (error) => {
      console.error("Failed to continue conversation:", error);
    },
  });
};

// Get conversations list for sidebar
export const useConversationsList = (
  limit: number = 50,
  offset: number = 0
) => {
  return useQuery({
    queryKey: conversationKeys.list(limit, offset),
    queryFn: () => apiService.getConversationsList(limit, offset),
    staleTime: 1000 * 30, // 30 seconds
    refetchOnWindowFocus: false,
  });
};

// Get conversation details
export const useConversation = (
  conversationId: string,
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: conversationKeys.conversation(conversationId),
    queryFn: () => apiService.getConversation(conversationId),
    enabled: enabled && !!conversationId,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60, // 1 minute
  });
};

// Delete conversation
export const useDeleteConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (conversationId: string) =>
      apiService.deleteConversation(conversationId),
    onSuccess: (_, conversationId) => {
      // Remove the specific conversation from cache
      queryClient.removeQueries({
        queryKey: conversationKeys.conversation(conversationId),
      });

      // Invalidate conversations list
      queryClient.invalidateQueries({ queryKey: conversationKeys.lists() });

      // Invalidate analytics
      queryClient.invalidateQueries({ queryKey: conversationKeys.analytics });
    },
    onError: (error) => {
      console.error("Failed to delete conversation:", error);
    },
  });
};

// Get analytics
export const useAnalytics = () => {
  return useQuery({
    queryKey: conversationKeys.analytics,
    queryFn: () => apiService.getAnalytics(),
    staleTime: 1000 * 60 * 5, // 5 minutes
    refetchOnWindowFocus: false,
  });
};

// Health check
export const useHealthCheck = () => {
  return useQuery({
    queryKey: conversationKeys.health,
    queryFn: () => apiService.healthCheck(),
    staleTime: 1000 * 30, // 30 seconds
    refetchInterval: 1000 * 60, // Refetch every minute
    refetchOnWindowFocus: false,
  });
};
