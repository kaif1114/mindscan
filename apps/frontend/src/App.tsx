import React from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout/Layout";
import ChatPage from "./pages/ChatPage";
import ResultsPage from "./pages/ResultsPage";
import HistoryPage from "./pages/HistoryPage";
import { ChatProvider } from "./context/ChatContext";
import ErrorBoundary from "./components/ErrorBoundary";

function App() {
  return (
    <ErrorBoundary>
      <ChatProvider>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<ChatPage />} />
            <Route path="results/:sessionId" element={<ResultsPage />} />
            <Route path="history" element={<HistoryPage />} />
          </Route>
        </Routes>
      </ChatProvider>
    </ErrorBoundary>
  );
}

export default App;
