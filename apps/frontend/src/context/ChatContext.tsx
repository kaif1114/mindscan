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

const convertRawPredictionToDASSPrediction = (
  rawPrediction: any
): DASSPrediction => {
  if (!rawPrediction || !rawPrediction.predictions) {
    throw new Error("Invalid prediction data");
  }

  const predictions = rawPrediction.predictions;

  const depressionScore = (predictions.Depression?.category_index || 0) * 7;
  const anxietyScore = (predictions.Anxiety?.category_index || 0) * 7;
  const stressScore = (predictions.Stress?.category_index || 0) * 7;

  const hasDisorder =
    predictions.Depression?.severity !== "Normal" ||
    predictions.Anxiety?.severity !== "Normal" ||
    predictions.Stress?.severity !== "Normal";

  let disorderType = null;
  if (hasDisorder) {
    const scores = [
      {
        type: "Depression",
        score: depressionScore,
        severity: predictions.Depression?.severity,
      },
      {
        type: "Anxiety",
        score: anxietyScore,
        severity: predictions.Anxiety?.severity,
      },
      {
        type: "Stress",
        score: stressScore,
        severity: predictions.Stress?.severity,
      },
    ];
    const highestScore = scores.reduce((prev, current) =>
      current.score > prev.score ? current : prev
    );
    if (highestScore.severity !== "Normal") {
      disorderType = highestScore.type;
    }
  }

  return {
    id: `pred-${Date.now()}`,
    conversation_id: "",
    depression_score: depressionScore,
    anxiety_score: anxietyScore,
    stress_score: stressScore,
    depression_level: predictions.Depression?.severity || "Normal",
    anxiety_level: predictions.Anxiety?.severity || "Normal",
    stress_level: predictions.Stress?.severity || "Normal",
    has_disorder: hasDisorder,
    disorder_type: disorderType,
    confidence: rawPrediction.model_info?.accuracy || 0.8,
    model_version: rawPrediction.model_info?.model_type || "unknown",
    created_at: new Date().toISOString(),
  };
};

const convertSimplifiedPredictionToDASSPrediction = (
  simplifiedPrediction: any
): DASSPrediction => {
  if (!simplifiedPrediction) {
    throw new Error("Invalid prediction data");
  }

  const severityToScore = {
    Normal: 0,
    Mild: 7,
    Moderate: 14,
    Severe: 21,
    "Extremely Severe": 28,
  };

  const depressionScore =
    severityToScore[
      simplifiedPrediction.depression?.severity as keyof typeof severityToScore
    ] || 0;
  const anxietyScore =
    severityToScore[
      simplifiedPrediction.anxiety?.severity as keyof typeof severityToScore
    ] || 0;
  const stressScore =
    severityToScore[
      simplifiedPrediction.stress?.severity as keyof typeof severityToScore
    ] || 0;

  const hasDisorder =
    simplifiedPrediction.depression?.severity !== "Normal" ||
    simplifiedPrediction.anxiety?.severity !== "Normal" ||
    simplifiedPrediction.stress?.severity !== "Normal";

  let disorderType = null;
  if (hasDisorder) {
    const scores = [
      {
        type: "Depression",
        score: depressionScore,
        severity: simplifiedPrediction.depression?.severity,
      },
      {
        type: "Anxiety",
        score: anxietyScore,
        severity: simplifiedPrediction.anxiety?.severity,
      },
      {
        type: "Stress",
        score: stressScore,
        severity: simplifiedPrediction.stress?.severity,
      },
    ];
    const highestScore = scores.reduce((prev, current) =>
      current.score > prev.score ? current : prev
    );
    if (highestScore.severity !== "Normal") {
      disorderType = highestScore.type;
    }
  }

  return {
    id: `pred-${Date.now()}`,
    conversation_id: "",
    depression_score: depressionScore,
    anxiety_score: anxietyScore,
    stress_score: stressScore,
    depression_level: simplifiedPrediction.depression?.severity || "Normal",
    anxiety_level: simplifiedPrediction.anxiety?.severity || "Normal",
    stress_level: simplifiedPrediction.stress?.severity || "Normal",
    has_disorder: hasDisorder,
    disorder_type: disorderType,
    confidence: 0.8,
    model_version: "unknown",
    created_at: simplifiedPrediction.timestamp || new Date().toISOString(),
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

  let latestPrediction: DASSPrediction | null = null;

  if (data.predictions) {
    try {
      if (Array.isArray(data.predictions)) {
        if (data.predictions.length > 0) {
          const lastPred = data.predictions[data.predictions.length - 1];
          latestPrediction = lastPred;
        }
      } else {
        const predictions = data.predictions as any;
        if (
          predictions.depression &&
          predictions.anxiety &&
          predictions.stress
        ) {
          latestPrediction =
            convertSimplifiedPredictionToDASSPrediction(predictions);
          latestPrediction.conversation_id = data.conversation.id;
        } else {
          latestPrediction = convertRawPredictionToDASSPrediction(predictions);
          latestPrediction.conversation_id = data.conversation.id;
        }
      }
    } catch (error) {
      console.error("Error converting prediction data:", error);
      latestPrediction = null;
    }
  }

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

  const {
    data: conversationsListData,
    isLoading: isConversationsListLoading,
    error: conversationsListError,
  } = useConversationsList();

  const createConversationMutation = useCreateConversation();
  const continueConversationMutation = useContinueConversation();
  const deleteConversationMutation = useDeleteConversation();

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

  const sessions: ChatSession[] = React.useMemo(() => {
    if (!conversationsListData?.conversations) return [];

    return conversationsListData.conversations.map(
      (summary: ConversationSummary) => {
        const localSession = localSessions.get(summary.id);

        if (localSession) {
          return {
            ...localSession,
            lastMessage: summary.last_message,
            timestamp: new Date(summary.last_message_timestamp),
          };
        }

        return {
          id: summary.id,
          title: summary.title,
          lastMessage: summary.last_message,
          timestamp: new Date(summary.last_message_timestamp),
          messages: [],
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

  React.useEffect(() => {
    if (conversationData && loadingSessionId) {
      try {
        const updatedSession =
          convertFullConversationToSession(conversationData);

        setLocalSessions(
          (prev) => new Map(prev.set(updatedSession.id, updatedSession))
        );

        setCurrentSession(updatedSession);

        setLoadingSessionId(null);
      } catch (error) {
        console.error("Error converting conversation data:", error);
        setError("Failed to process conversation data");
        setLoadingSessionId(null);
      }
    }
  }, [conversationData, loadingSessionId]);

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

      const userMessage: Message = {
        id: `user-${Date.now()}`,
        content,
        sender: "user",
        timestamp: new Date(),
      };

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

      const response = await continueConversationMutation.mutateAsync({
        conversation_id: currentSession.id,
        message: content,
      });

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

      if (response.is_assessment_complete && response.predictions) {
        const dassPrediction = convertRawPredictionToDASSPrediction(
          response.predictions
        );
        dassPrediction.conversation_id = currentSession.id;
        finalSession.analysis = formatDASSToPrediction(dassPrediction);
      } else if (currentSession.analysis) {
        finalSession.analysis = currentSession.analysis;
      }

      setCurrentSession(finalSession);
      setLocalSessions(
        (prev) => new Map(prev.set(currentSession.id, finalSession))
      );
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  };

  const selectSession = (sessionId: string) => {
    const localSession = localSessions.get(sessionId);
    if (localSession && localSession.messages.length > 0) {
      setCurrentSession(localSession);
    } else {
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
