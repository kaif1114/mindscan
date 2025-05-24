import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Trash2,
  Search,
  Calendar,
  MessageCircle,
  Eye,
  Loader2,
  BarChart3,
  AlertCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useChatContext } from "../context/ChatContext";
import { useAnalytics } from "../hooks/useConversations";
import { formatDate } from "../utils/helpers";

const HistoryPage: React.FC = () => {
  const {
    sessions,
    deleteSession,
    setCurrentSession,
    isLoading,
    error,
    clearError,
  } = useChatContext();
  const { data: analytics, isLoading: analyticsLoading } = useAnalytics();
  const [searchTerm, setSearchTerm] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const navigate = useNavigate();

  const filteredSessions = sessions.filter(
    (session) =>
      session.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.lastMessage.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleViewChat = (sessionId: string) => {
    setCurrentSession(sessionId);
    navigate("/");
  };

  const handleViewResults = (sessionId: string) => {
    navigate(`/results/${sessionId}`);
  };

  const handleDeleteSession = async (sessionId: string) => {
    setDeletingId(sessionId);
    try {
      await deleteSession(sessionId);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="p-4">
      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg mb-4 p-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-red-700">
            <AlertCircle size={16} />
            <span className="text-sm">{error}</span>
          </div>
          <button
            onClick={clearError}
            className="text-red-500 hover:text-red-700 text-sm font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">Conversation History</h1>
        <p className="text-gray-600">
          View and manage your DASS-21 assessment conversations.
        </p>
      </div>

      {/* Analytics Section */}
      {analytics && (
        <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
          <div className="flex items-center mb-4">
            <BarChart3 className="h-5 w-5 text-primary-600 mr-2" />
            <h2 className="text-lg font-semibold">Analytics Overview</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary-600">
                {analytics.total_conversations}
              </div>
              <div className="text-sm text-gray-600">Total Conversations</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {analytics.completed_conversations}
              </div>
              <div className="text-sm text-gray-600">Completed Assessments</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {Math.round(analytics.completion_rate * 100)}%
              </div>
              <div className="text-sm text-gray-600">Completion Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(analytics.average_completion_time_minutes)}m
              </div>
              <div className="text-sm text-gray-600">Avg. Time</div>
            </div>
          </div>
        </div>
      )}

      <div className="mb-6 max-w-md">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search size={18} className="text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search by conversation title or content"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10"
          />
        </div>
      </div>

      {isLoading && sessions.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center gap-3 text-primary-600">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span>Loading conversations...</span>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <AnimatePresence>
            {filteredSessions.length > 0 ? (
              filteredSessions.map((session) => (
                <motion.div
                  key={session.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="card card-hover"
                >
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="font-semibold text-lg">{session.title}</h3>
                    <button
                      onClick={() => handleDeleteSession(session.id)}
                      disabled={deletingId === session.id}
                      className="p-1 text-gray-400 hover:text-error-600 rounded-full hover:bg-gray-100 disabled:opacity-50"
                      aria-label="Delete conversation"
                    >
                      {deletingId === session.id ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <Trash2 size={16} />
                      )}
                    </button>
                  </div>

                  <div className="flex items-center text-sm text-gray-500 mb-3">
                    <Calendar size={14} className="mr-1" />
                    <span>{formatDate(session.timestamp)}</span>
                    <span className="mx-2">•</span>
                    <span>{session.messages.length} messages</span>
                  </div>

                  <p className="text-gray-700 mb-4 line-clamp-2">
                    {session.lastMessage}
                  </p>

                  {session.analysis && (
                    <div className="mb-4">
                      <div
                        className={`text-sm p-2 rounded-md mb-2 ${
                          session.analysis.hasDisorder
                            ? "bg-warning-50 text-warning-700"
                            : "bg-success-50 text-success-700"
                        }`}
                      >
                        {session.analysis.hasDisorder
                          ? `${session.analysis.disorderType} indicators detected`
                          : "No significant concerns detected"}
                      </div>

                      {/* Score summary */}
                      {(session.analysis.depressionScore !== undefined ||
                        session.analysis.anxietyScore !== undefined ||
                        session.analysis.stressScore !== undefined) && (
                        <div className="text-xs text-gray-600 space-y-1">
                          {session.analysis.depressionScore !== undefined && (
                            <div>
                              Depression: {session.analysis.depressionScore} (
                              {session.analysis.depressionLevel})
                            </div>
                          )}
                          {session.analysis.anxietyScore !== undefined && (
                            <div>
                              Anxiety: {session.analysis.anxietyScore} (
                              {session.analysis.anxietyLevel})
                            </div>
                          )}
                          {session.analysis.stressScore !== undefined && (
                            <div>
                              Stress: {session.analysis.stressScore} (
                              {session.analysis.stressLevel})
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="flex space-x-2 mt-auto pt-2 border-t border-gray-100">
                    <button
                      onClick={() => handleViewChat(session.id)}
                      className="btn btn-outline flex-1 flex items-center justify-center gap-1 py-1.5"
                    >
                      <MessageCircle size={16} />
                      <span>Open</span>
                    </button>

                    {session.analysis ? (
                      <button
                        onClick={() => handleViewResults(session.id)}
                        className="btn btn-outline flex-1 flex items-center justify-center gap-1 py-1.5"
                      >
                        <Eye size={16} />
                        <span>Results</span>
                      </button>
                    ) : (
                      <button
                        onClick={() => handleViewChat(session.id)}
                        className="btn btn-outline flex-1 flex items-center justify-center gap-1 py-1.5"
                      >
                        <MessageCircle size={16} />
                        <span>Continue</span>
                      </button>
                    )}
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="col-span-full text-center py-12">
                <MessageCircle
                  size={48}
                  className="mx-auto mb-4 text-gray-300"
                />
                <h3 className="text-lg font-medium text-gray-700 mb-2">
                  No conversations found
                </h3>
                <p className="text-gray-500 mb-6">
                  {searchTerm
                    ? "Try a different search term or clear your search"
                    : "Use the 'New Assessment' button in the sidebar to start your first DASS-21 assessment"}
                </p>
                <button
                  onClick={() => navigate("/")}
                  className="btn btn-primary"
                >
                  Start New Assessment
                </button>
              </div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

export default HistoryPage;
