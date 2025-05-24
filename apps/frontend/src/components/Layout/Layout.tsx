import React from "react";
import { Outlet } from "react-router-dom";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import { useChatContext } from "../../context/ChatContext";
import { motion } from "framer-motion";

const Layout: React.FC = () => {
  const { currentSession, sessions, createNewSession, isLoading } =
    useChatContext();
  const [sidebarOpen, setSidebarOpen] = React.useState(true);

  const toggleSidebar = () => {
    setSidebarOpen((prev) => !prev);
  };

  const handleCreateNewSession = async () => {
    await createNewSession();
  };

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      <motion.div
        initial={{ x: -280 }}
        animate={{ x: sidebarOpen ? 0 : -280 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="fixed md:relative z-20 w-[280px] h-full bg-white border-r border-gray-200 shadow-sm"
      >
        <Sidebar onNewChat={handleCreateNewSession} />
      </motion.div>

      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar} />

        <main className="flex-1 overflow-y-auto bg-gray-50 p-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="container mx-auto max-w-5xl"
          >
            <Outlet />
          </motion.div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
