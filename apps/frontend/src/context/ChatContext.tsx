import React, { createContext, useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import {
  useCreateConversation,
  useContinueConversation,
  useDeleteConversation,
  useConversation,
  useConversationsList,
} from "../hooks/useConversations";
import {
  ConversationMessage,
  FullConversationResponse,
  DASSPrediction,
  ConversationSummary,
} from "../types/api";

export interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messages: Message[];
  analysis?: {
    hasDisorder: boolean;
    disorderType?: string;
    confidence: number;
    summary: string;
    recommendations: string[];
    depressionScore?: number;
    anxietyScore?: number;
    stressScore?: number;
    depressionLevel?: string;
    anxietyLevel?: string;
    stressLevel?: string;
  };
}

interface ChatContextType {
  currentSession: ChatSession | null;
  sessions: ChatSession[];
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  createNewSession: (initialMessage?: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  setCurrentSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => Promise<void>;
  loadSession: (sessionId: string) => Promise<void>;
  clearError: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const useChatContext = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return context;
};

const formatDASSToPrediction = (prediction: DASSPrediction) => {
  const levelDescriptions: { [key: string]: string } = {
    Normal: "within normal range",
    Mild: "mild levels detected",
    Moderate: "moderate levels detected",
    Severe: "severe levels detected",
    "Extremely Severe": "extremely severe levels detected",
  };

  const getRecommendations = (prediction: DASSPrediction): string[] => {
    const recommendations = [];

    if (prediction.depression_level !== "Normal") {
      recommendations.push(
        "Consider speaking with a mental health professional about depression"
      );
      recommendations.push(
        "Maintain regular sleep patterns and physical activity"
      );
    }

    if (prediction.anxiety_level !== "Normal") {
      recommendations.push(
        "Practice relaxation techniques such as deep breathing or meditation"
      );
      recommendations.push("Consider anxiety management strategies");
    }

    if (prediction.stress_level !== "Normal") {
      recommendations.push(
        "Identify and address sources of stress in your life"
      );
      recommendations.push("Consider stress management techniques");
    }

    if (prediction.has_disorder) {
      recommendations.push("Seek professional mental health support");
    } else {
      recommendations.push("Continue monitoring your mental wellbeing");
    }

    recommendations.push(
      "Maintain healthy lifestyle habits including regular exercise and social connections"
    );

    return recommendations;
  };

  let summary = "Based on the DASS-21 assessment: ";
  const parts = [];

  if (prediction.depression_level !== "Normal") {
    parts.push(
      `Depression ${
        levelDescriptions[prediction.depression_level] ||
        prediction.depression_level.toLowerCase()
      }`
    );
  }

  if (prediction.anxiety_level !== "Normal") {
    parts.push(
      `Anxiety ${
        levelDescriptions[prediction.anxiety_level] ||
        prediction.anxiety_level.toLowerCase()
      }`
    );
  }

  if (prediction.stress_level !== "Normal") {
    parts.push(
      `Stress ${
        levelDescriptions[prediction.stress_level] ||
        prediction.stress_level.toLowerCase()
      }`
    );
  }

  if (parts.length === 0) {
    summary += "All measures are within normal ranges.";
  } else {
    summary += parts.join(", ") + ".";
  }

  return {
    hasDisorder: prediction.has_disorder,
    disorderType: prediction.disorder_type || undefined,
    confidence: prediction.confidence,
    summary,
    recommendations: getRecommendations(prediction),
    depressionScore: prediction.depression_score,
    anxietyScore: prediction.anxiety_score,
    stressScore: prediction.stress_score,
    depressionLevel: prediction.depression_level,
    anxietyLevel: prediction.anxiety_level,
    stressLevel: prediction.stress_level,
  };
};

const convertAPIMessagesToMessages = (
  apiMessages: ConversationMessage[]
): Message[] => {
  if (!apiMessages || !Array.isArray(apiMessages)) {
    return [];
  }

  return apiMessages
    .filter((msg) => msg.role !== "system")
    .map((msg) => ({
      id: msg.id,
      content: msg.content,
      sender: msg.role === "user" ? ("user" as const) : ("ai" as const),
      timestamp: new Date(msg.timestamp),
    }));
};

const convertFullConversationToSession = (
  data: FullConversationResponse
): ChatSession => {
  if (!data.conversation) {
    throw new Error("Conversation data is missing");
  }

  const messages = convertAPIMessagesToMessages(data.messages || []);
  const lastMessage =
    messages.length > 0
      ? messages[messages.length - 1].content
      : "New conversation";

  // Get the latest prediction if available
  const predictions = data.predictions || [];
  const latestPrediction =
    predictions.length > 0 ? predictions[predictions.length - 1] : null;

  return {
    id: data.conversation.id,
    title:
      data.conversation.title || `Chat ${data.conversation.id.slice(0, 8)}`,
    lastMessage,
    timestamp: new Date(data.conversation.updated_at || new Date()),
    messages,
    analysis: latestPrediction
      ? formatDASSToPrediction(latestPrediction)
      : undefined,
  };
};

export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(
    null
  );
  const [localSessions, setLocalSessions] = useState<Map<string, ChatSession>>(
    new Map()
  );
  const [error, setError] = useState<string | null>(null);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const navigate = useNavigate();

  // Fetch conversations list for sidebar
  const {
    data: conversationsListData,
    isLoading: isConversationsListLoading,
    error: conversationsListError,
  } = useConversationsList();

  const createConversationMutation = useCreateConversation();
  const continueConversationMutation = useContinueConversation();
  const deleteConversationMutation = useDeleteConversation();

  // Only use this hook when we need to load a specific session from backend
  const {
    data: conversationData,
    isLoading: isConversationLoading,
    error: conversationError,
  } = useConversation(loadingSessionId || "", !!loadingSessionId);

  const isLoading =
    createConversationMutation.isPending ||
    continueConversationMutation.isPending ||
    deleteConversationMutation.isPending ||
    isConversationLoading;

  // Convert API conversation summaries to ChatSessions for UI
  const sessions: ChatSession[] = React.useMemo(() => {
    if (!conversationsListData?.conversations) return [];

    return conversationsListData.conversations.map(
      (summary: ConversationSummary) => {
        // Check if we have local data for this conversation
        const localSession = localSessions.get(summary.id);

        if (localSession) {
          // Use local data but update metadata from API
          return {
            ...localSession,
            lastMessage: summary.last_message,
            timestamp: new Date(summary.last_message_timestamp),
          };
        }

        // Create session from API summary
        return {
          id: summary.id,
          title: summary.title,
          lastMessage: summary.last_message,
          timestamp: new Date(summary.last_message_timestamp),
          messages: [], // Will be loaded when needed
          analysis: summary.has_predictions
            ? {
                hasDisorder: false,
                confidence: 0,
                summary: "",
                recommendations: [],
              }
            : undefined,
        };
      }
    );
  }, [conversationsListData, localSessions]);

  // Handle loaded conversation data
  React.useEffect(() => {
    if (conversationData && loadingSessionId) {
      try {
        const updatedSession =
          convertFullConversationToSession(conversationData);

        // Store in local sessions map
        setLocalSessions(
          (prev) => new Map(prev.set(updatedSession.id, updatedSession))
        );

        // Set as current session
        setCurrentSession(updatedSession);

        // Clear loading state
        setLoadingSessionId(null);
      } catch (error) {
        console.error("Error converting conversation data:", error);
        setError("Failed to process conversation data");
        setLoadingSessionId(null);
      }
    }
  }, [conversationData, loadingSessionId]);

  // Handle errors
  React.useEffect(() => {
    if (createConversationMutation.error) {
      setError(createConversationMutation.error.message);
    } else if (continueConversationMutation.error) {
      setError(continueConversationMutation.error.message);
    } else if (deleteConversationMutation.error) {
      setError(deleteConversationMutation.error.message);
    } else if (conversationError) {
      setError(conversationError.message);
      setLoadingSessionId(null);
    } else if (conversationsListError) {
      setError(conversationsListError.message);
    }
  }, [
    createConversationMutation.error,
    continueConversationMutation.error,
    deleteConversationMutation.error,
    conversationError,
    conversationsListError,
  ]);

  const createNewSession = async (
    initialMessage: string = "Hello, I'd like to take the DASS assessment."
  ) => {
    try {
      setError(null);
      const response = await createConversationMutation.mutateAsync({
        initial_message: initialMessage,
      });

      // Create session with initial messages
      const newSession: ChatSession = {
        id: response.conversation_id,
        title: `New Chat`,
        lastMessage: response.message,
        timestamp: new Date(),
        messages: [
          {
            id: `user-${Date.now()}`,
            content: initialMessage,
            sender: "user",
            timestamp: new Date(),
          },
          {
            id: `ai-${Date.now()}`,
            content: response.message,
            sender: "ai",
            timestamp: new Date(),
          },
        ],
      };

      setLocalSessions((prev) => new Map(prev.set(newSession.id, newSession)));
      setCurrentSession(newSession);
    } catch (error) {
      console.error("Failed to create conversation:", error);
    }
  };

  const sendMessage = async (content: string) => {
    if (!currentSession || !content.trim()) return;

    try {
      setError(null);

      // Add user message to local state immediately
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        content,
        sender: "user",
        timestamp: new Date(),
      };

      // Update local state with user message
      const updatedSession = {
        ...currentSession,
        messages: [...currentSession.messages, userMessage],
        lastMessage: content,
        timestamp: new Date(),
      };

      setCurrentSession(updatedSession);
      setLocalSessions(
        (prev) => new Map(prev.set(currentSession.id, updatedSession))
      );

      // Send to backend
      const response = await continueConversationMutation.mutateAsync({
        conversation_id: currentSession.id,
        message: content,
      });

      // Add AI response to local state
      const aiMessage: Message = {
        id: `ai-${Date.now()}`,
        content: response.message,
        sender: "ai",
        timestamp: new Date(),
      };

      const finalSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, aiMessage],
        lastMessage: response.message,
        timestamp: new Date(),
      };

      // If assessment is complete, add analysis
      if (response.is_assessment_complete && response.predictions) {
        finalSession.analysis = formatDASSToPrediction(response.predictions);
      }

      setCurrentSession(finalSession);
      setLocalSessions(
        (prev) => new Map(prev.set(currentSession.id, finalSession))
      );

      // Navigate to results if assessment is complete
      if (response.is_assessment_complete && response.predictions) {
        navigate(`/results/${currentSession.id}`);
      }
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  };

  const selectSession = (sessionId: string) => {
    // Check if we have the session with messages in local cache
    const localSession = localSessions.get(sessionId);
    if (localSession && localSession.messages.length > 0) {
      setCurrentSession(localSession);
    } else {
      // Load session from backend if not in local cache or no messages
      loadSession(sessionId);
    }
  };

  const loadSession = async (sessionId: string) => {
    setLoadingSessionId(sessionId);
  };

  const deleteSession = async (sessionId: string) => {
    try {
      setError(null);
      await deleteConversationMutation.mutateAsync(sessionId);

      setLocalSessions((prev) => {
        const newSessions = new Map(prev);
        newSessions.delete(sessionId);
        return newSessions;
      });

      if (currentSession && currentSession.id === sessionId) {
        setCurrentSession(null);
      }
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  };

  const clearError = () => {
    setError(null);
  };

  const messages = currentSession?.messages || [];

  return (
    <ChatContext.Provider
      value={{
        currentSession,
        sessions,
        messages,
        isLoading,
        error,
        createNewSession,
        sendMessage,
        setCurrentSession: selectSession,
        deleteSession,
        loadSession,
        clearError,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};
