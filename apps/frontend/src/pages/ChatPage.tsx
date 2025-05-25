import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  AlertCircle,
  Loader2,
  Brain,
  MessageCircle,
  CheckCircle,
  Eye,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import TextareaAutosize from "react-textarea-autosize";
import { useChatContext } from "../context/ChatContext";
import { formatDate } from "../utils/helpers";
import { useNavigate } from "react-router-dom";

const ChatPage: React.FC = () => {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const navigate = useNavigate();
  const {
    currentSession,
    messages,
    sendMessage,
    isLoading,
    error,
    clearError,
  } = useChatContext();

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (currentSession) {
      inputRef.current?.focus();
    }
  }, [currentSession]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const messageToSend = input;
    setInput(""); // Clear input immediately for better UX

    await sendMessage(messageToSend);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Show welcome screen if no current session
  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-5rem)] p-8">
        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg mb-6 p-3 flex items-center justify-between max-w-md w-full">
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

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center max-w-2xl"
        >
          <div className="mb-8">
            <Brain className="w-24 h-24 text-primary-600 mx-auto mb-6" />
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Welcome to MindScan AI
            </h1>
            <p className="text-xl text-gray-600 mb-2">
              DASS-21 Mental Health Assessment
            </p>
            <p className="text-gray-500 max-w-lg mx-auto">
              Take a confidential, AI-guided assessment to evaluate your current
              mental health status. The DASS-21 scale measures depression,
              anxiety, and stress levels through a conversational interface.
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-md p-8 mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              How it works:
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center font-semibold text-sm">
                  1
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-1">
                    Start Conversation
                  </h3>
                  <p className="text-sm text-gray-600">
                    Click "New Assessment" in the sidebar to begin
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center font-semibold text-sm">
                  2
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-1">
                    Answer Questions
                  </h3>
                  <p className="text-sm text-gray-600">
                    Respond to 21 questions about your recent experiences
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center font-semibold text-sm">
                  3
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-1">
                    Get Results
                  </h3>
                  <p className="text-sm text-gray-600">
                    Receive detailed analysis and recommendations
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center">
                <MessageCircle className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-semibold text-blue-800">Ready to begin?</h3>
                <p className="text-sm text-blue-600">
                  Use the sidebar to start your assessment
                </p>
              </div>
            </div>
            <p className="text-sm text-blue-700">
              Look for the <strong>"New Assessment"</strong> button in the left
              sidebar to start your DASS-21 evaluation.
            </p>
          </div>

          <p className="text-xs text-gray-500 mt-6 max-w-md mx-auto">
            <strong>Disclaimer:</strong> This assessment is for informational
            purposes only and should not replace professional mental health
            consultation.
          </p>
        </motion.div>
      </div>
    );
  }

  // Show chat interface if session exists
  return (
    <div className="flex flex-col h-[calc(100vh-5rem)]">
      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg mx-4 mt-4 p-3 flex items-center justify-between">
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

      {/* Assessment Completed Banner */}
      {currentSession?.analysis && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-green-50 border border-green-200 rounded-lg mx-4 mt-4 p-3"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-green-700">
              <CheckCircle size={16} />
              <span className="text-sm font-medium">
                Assessment completed! Your results are ready.
              </span>
            </div>
            <button
              onClick={() => navigate(`/results/${currentSession.id}`)}
              className="flex items-center gap-1 px-3 py-1 bg-green-100 hover:bg-green-200 text-green-700 rounded-md text-sm font-medium transition-colors"
            >
              <Eye size={14} />
              View Results
            </button>
          </div>
          <p className="text-xs text-green-600 mt-1">
            You can continue our conversation to discuss your results or ask any
            questions.
          </p>
        </motion.div>
      )}

      <div className="flex-1 overflow-y-auto p-4 pb-0">
        <div className="flex flex-col space-y-4">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="flex flex-col"
              >
                <div
                  className={`chat-bubble ${
                    message.sender === "user"
                      ? "chat-bubble-user"
                      : "chat-bubble-ai"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
                <span
                  className={`text-xs mt-1 text-gray-500 ${
                    message.sender === "user" ? "ml-auto" : ""
                  }`}
                >
                  {formatDate(message.timestamp)}
                </span>
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center"
            >
              <div className="chat-bubble chat-bubble-ai flex items-center space-x-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>AI is thinking...</span>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="p-4 border-t border-gray-200 bg-white">
        <div className="relative">
          <TextareaAutosize
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isLoading ? "Please wait..." : "Type your message here..."
            }
            className="input min-h-[50px] pr-12 resize-none"
            maxRows={5}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="absolute right-2 bottom-2 p-2 text-primary-600 hover:text-primary-800 disabled:text-gray-300 disabled:cursor-not-allowed transition-colors duration-200"
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>

        <div className="mt-2 text-xs text-gray-500 text-center">
          <p>
            MindScan AI is here to help, but it's not a replacement for
            professional mental health support.
          </p>
          <p className="mt-1">
            Your assessment will complete automatically when all questions are
            answered.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
