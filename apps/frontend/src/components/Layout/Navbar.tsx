import React from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Menu,
  Brain,
  History,
  X,
  MessageCircle,
  RefreshCw,
} from "lucide-react";
import { motion } from "framer-motion";
import { useChatContext } from "../../context/ChatContext";
import { apiService } from "../../services/api";

interface NavbarProps {
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ sidebarOpen, toggleSidebar }) => {
  const location = useLocation();
  const { currentSession } = useChatContext();
  const [isRetraining, setIsRetraining] = React.useState(false);

  const getPageTitle = () => {
    const path = location.pathname;
    if (path === "/") return "Chat";
    if (path.includes("/results")) return "Analysis Results";
    if (path === "/history") return "History";
    return "MindScan AI";
  };

  const handleRetrain = async () => {
    if (isRetraining) return;

    setIsRetraining(true);
    try {
      const result = await apiService.retrainModel();
      alert(
        `Model retrained successfully!\n\nTraining samples: ${result.training_samples}\nComplete conversations: ${result.complete_conversations}\nNew accuracy: ${result.new_accuracy}`
      );
    } catch (error) {
      console.error("Retrain error:", error);
      alert(
        `Retraining failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    } finally {
      setIsRetraining(false);
    }
  };

  return (
    <header className="bg-white border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-md text-gray-500 hover:text-gray-900 focus:outline-none"
              aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            >
              {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
              className="ml-4 flex items-center"
            >
              <Brain className="h-8 w-8 text-primary-600" />
              <h1 className="ml-2 text-xl font-semibold text-gray-900">
                MindScan AI
              </h1>
            </motion.div>
          </div>

          <div className="hidden md:block">
            <h2 className="text-lg font-medium text-gray-700">
              {getPageTitle()}
            </h2>
          </div>

          <div className="flex items-center space-x-4">
            <nav className="flex space-x-2">
              <NavLink to="/" active={location.pathname === "/"}>
                <MessageCircle size={20} />
                <span className="hidden md:inline ml-1">Chat</span>
              </NavLink>

              {currentSession && (
                <NavLink
                  to={`/results/${currentSession.id}`}
                  active={location.pathname.includes("/results")}
                >
                  <Brain size={20} />
                  <span className="hidden md:inline ml-1">Results</span>
                </NavLink>
              )}

              <NavLink to="/history" active={location.pathname === "/history"}>
                <History size={20} />
                <span className="hidden md:inline ml-1">History</span>
              </NavLink>

              <button
                onClick={handleRetrain}
                disabled={isRetraining}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                  isRetraining
                    ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                }`}
                title="Retrain model with conversation data"
              >
                <RefreshCw
                  size={20}
                  className={isRetraining ? "animate-spin" : ""}
                />
                <span className="hidden md:inline ml-1">
                  {isRetraining ? "Retraining..." : "Retrain"}
                </span>
              </button>
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
};

interface NavLinkProps {
  to: string;
  active: boolean;
  children: React.ReactNode;
}

const NavLink: React.FC<NavLinkProps> = ({ to, active, children }) => {
  return (
    <Link
      to={to}
      className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
        active
          ? "bg-primary-50 text-primary-700"
          : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
      }`}
    >
      {children}
    </Link>
  );
};

export default Navbar;
