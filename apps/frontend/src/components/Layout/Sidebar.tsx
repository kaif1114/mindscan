import React, { useState } from "react";
import {
  PlusCircle,
  MessageCircle,
  Trash2,
  Search,
  Loader2,
  Eye,
  AlertCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useChatContext } from "../../context/ChatContext";
import { useConversationsList } from "../../hooks/useConversations";
import { formatDate, truncateText } from "../../utils/helpers";
import { useNavigate } from "react-router-dom";

interface SidebarProps {
  onNewChat: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onNewChat }) => {
  const {
    currentSession,
    setCurrentSession,
    deleteSession,
    error,
    clearError,
  } = useChatContext();
  const {
    data: conversationsData,
    isLoading,
    error: listError,
  } = useConversationsList();
  const [searchTerm, setSearchTerm] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const navigate = useNavigate();

  const sessions = conversationsData?.conversations || [];

  const filteredSessions = sessions.filter(
    (session) =>
      session.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.last_message.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSessionClick = (sessionId: string) => {
    setCurrentSession(sessionId);
    navigate("/"); // Navigate to chat page
  };

  const handleDeleteSession = async (
    sessionId: string,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();
    setDeletingId(sessionId);
    try {
      await deleteSession(sessionId);
    } finally {
      setDeletingId(null);
    }
  };

  const handleViewResults = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigate(`/results/${sessionId}`);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-gray-200">
        <button
          onClick={onNewChat}
          className="btn btn-primary w-full flex items-center justify-center gap-2"
        >
          <PlusCircle size={18} />
          <span>New Assessment</span>
        </button>

        <div className="mt-4 relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search size={16} className="text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search conversations"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10"
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Error Banner */}
      {(error || listError) && (
        <div className="mx-3 mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs">
          <div className="flex items-center gap-1 text-red-700">
            <AlertCircle size={12} />
            <span>{error || listError?.message}</span>
          </div>
          <button
            onClick={clearError}
            className="mt-1 text-red-600 hover:text-red-800 underline"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto scrollbar-hide p-3 space-y-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
          </div>
        ) : (
          <AnimatePresence>
            {filteredSessions.length > 0 ? (
              filteredSessions.map((session) => (
                <motion.div
                  key={session.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.2 }}
                >
                  <div
                    className={`sidebar-item cursor-pointer ${
                      currentSession?.id === session.id ? "active" : ""
                    }`}
                    onClick={() => handleSessionClick(session.id)}
                  >
                    <MessageCircle size={18} className="mr-3 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-start">
                        <p className="font-medium truncate">{session.title}</p>
                        <span className="text-xs text-gray-500 ml-2 flex-shrink-0">
                          {formatDate(new Date(session.last_message_timestamp))}
                        </span>
                      </div>
                      <p className="text-sm text-gray-500 truncate">
                        {truncateText(session.last_message, 30)}
                      </p>

                      {/* Status indicators */}
                      <div className="flex items-center mt-1 gap-2">
                        <span className="text-xs text-gray-400">
                          {session.message_count} messages
                        </span>
                        {session.is_completed && (
                          <span className="text-xs text-green-600 font-medium">
                            Completed
                          </span>
                        )}
                        {session.has_predictions && (
                          <span className="text-xs text-blue-600 font-medium">
                            Analyzed
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center ml-2 gap-1">
                      {session.has_predictions && (
                        <button
                          onClick={(e) => handleViewResults(session.id, e)}
                          className="p-1 text-gray-400 hover:text-blue-600 rounded-full hover:bg-gray-100"
                          title="View Results"
                        >
                          <Eye size={14} />
                        </button>
                      )}
                      <button
                        onClick={(e) => handleDeleteSession(session.id, e)}
                        disabled={deletingId === session.id}
                        className="p-1 text-gray-400 hover:text-error-600 rounded-full hover:bg-gray-100 disabled:opacity-50"
                        title="Delete Conversation"
                      >
                        {deletingId === session.id ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <Trash2 size={14} />
                        )}
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <MessageCircle size={40} className="mx-auto mb-2 opacity-20" />
                <p className="text-sm">
                  {searchTerm
                    ? "No conversations found"
                    : "No conversations yet"}
                </p>
                {!searchTerm && (
                  <p className="text-xs mt-1">
                    Start a new assessment to begin
                  </p>
                )}
              </div>
            )}
          </AnimatePresence>
        )}
      </div>

      <div className="p-4 border-t border-gray-200">
        <div className="text-xs text-gray-500 text-center">
          <p>MindScan AI v2.1</p>
          <p className="mt-1">©2025 Mental Health AI</p>
          {conversationsData && (
            <p className="mt-1 text-gray-400">
              {conversationsData.total} conversations
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
