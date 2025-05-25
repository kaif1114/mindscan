import React, { useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  CheckCircle,
  ArrowLeft,
  Download,
  MessageCircle,
  Loader2,
  BarChart3,
} from "lucide-react";
import { motion } from "framer-motion";
import { useChatContext } from "../context/ChatContext";

const ResultsPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { sessions, setCurrentSession, isLoading } = useChatContext();
  const navigate = useNavigate();

  const session = sessions.find((s) => s.id === sessionId);

  useEffect(() => {
    if (sessionId) {
      setCurrentSession(sessionId);
    }
  }, [sessionId, setCurrentSession]);

  if (isLoading && !session) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8">
        <Loader2 className="w-12 h-12 animate-spin text-primary-600 mb-4" />
        <h2 className="text-xl font-semibold text-gray-800 mb-2">
          Loading results...
        </h2>
        <p className="text-gray-600">
          Please wait while we retrieve your assessment results...
        </p>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8">
        <AlertTriangle size={48} className="text-warning-500 mb-4" />
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Session not found
        </h2>
        <p className="text-gray-600 mb-6">
          The conversation you're looking for doesn't exist or has been deleted.
        </p>
        <button onClick={() => navigate("/")} className="btn btn-primary">
          Return to Chat
        </button>
      </div>
    );
  }

  if (!session.analysis) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8">
        <AlertTriangle size={48} className="text-warning-500 mb-4" />
        <h2 className="text-xl font-semibold text-gray-800 mb-2">
          Assessment not completed
        </h2>
        <p className="text-gray-600 mb-6">
          This conversation hasn't been fully analyzed yet. Please complete the
          DASS assessment.
        </p>
        <button onClick={() => navigate("/")} className="btn btn-primary">
          Continue Assessment
        </button>
      </div>
    );
  }

  const { analysis } = session;

  const ScoreCard = ({
    title,
    score,
    level,
    maxScore = 21,
  }: {
    title: string;
    score?: number;
    level?: string;
    maxScore?: number;
  }) => {
    if (score === undefined || level === undefined) return null;

    const percentage = (score / maxScore) * 100;
    const getColorClass = (level: string) => {
      switch (level) {
        case "Normal":
          return "bg-green-500";
        case "Mild":
          return "bg-yellow-500";
        case "Moderate":
          return "bg-orange-500";
        case "Severe":
          return "bg-red-500";
        case "Extremely Severe":
          return "bg-red-700";
        default:
          return "bg-gray-500";
      }
    };

    return (
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-semibold text-gray-800">{title}</h4>
          <span className="text-2xl font-bold text-gray-900">{score}</span>
        </div>
        <div className="mb-2">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${getColorClass(level)}`}
              style={{ width: `${percentage}%` }}
            ></div>
          </div>
        </div>
        <p
          className={`text-sm font-medium ${
            level === "Normal"
              ? "text-green-700"
              : level === "Mild"
                ? "text-yellow-700"
                : level === "Moderate"
                  ? "text-orange-700"
                  : "text-red-700"
          }`}
        >
          {level}
        </p>
      </div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="p-4"
    >
      <div className="flex items-center mb-6">
        <button
          onClick={() => navigate("/")}
          className="mr-4 p-2 rounded-full hover:bg-gray-100"
          aria-label="Go back to chat"
        >
          <ArrowLeft size={20} />
        </button>
        <h1 className="text-2xl font-bold">DASS-21 Assessment Results</h1>
      </div>

      <div className="bg-white rounded-xl shadow-md overflow-hidden mb-6">
        <div
          className={`p-4 ${
            analysis.hasDisorder ? "bg-warning-50" : "bg-success-50"
          }`}
        >
          <div className="flex items-center">
            {analysis.hasDisorder ? (
              <AlertTriangle className="h-8 w-8 text-warning-500 mr-3" />
            ) : (
              <CheckCircle className="h-8 w-8 text-success-500 mr-3" />
            )}
            <div>
              <h2 className="text-xl font-semibold">
                {analysis.hasDisorder
                  ? `${analysis.disorderType} Indicators Detected`
                  : "No Significant Concerns Detected"}
              </h2>
              <p className="text-sm text-gray-600">
                Confidence: {Math.round(analysis.confidence * 100)}%
              </p>
            </div>
          </div>
        </div>

        <div className="p-6">
          {/* DASS Scores Section */}
          {(analysis.depressionScore !== undefined ||
            analysis.anxietyScore !== undefined ||
            analysis.stressScore !== undefined) && (
            <section className="mb-6">
              <div className="flex items-center mb-4">
                <BarChart3 className="h-5 w-5 text-primary-600 mr-2" />
                <h3 className="text-lg font-semibold">DASS-21 Scores</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <ScoreCard
                  title="Depression"
                  score={analysis.depressionScore}
                  level={analysis.depressionLevel}
                />
                <ScoreCard
                  title="Anxiety"
                  score={analysis.anxietyScore}
                  level={analysis.anxietyLevel}
                />
                <ScoreCard
                  title="Stress"
                  score={analysis.stressScore}
                  level={analysis.stressLevel}
                />
              </div>
            </section>
          )}

          <section className="mb-6">
            <h3 className="text-lg font-semibold mb-2">Summary</h3>
            <p className="text-gray-700">{analysis.summary}</p>
          </section>

          <section>
            <h3 className="text-lg font-semibold mb-2">Recommendations</h3>
            <ul className="space-y-2">
              {analysis.recommendations.map((rec, index) => (
                <li key={index} className="flex items-start">
                  <div className="h-6 w-6 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                    {index + 1}
                  </div>
                  <p className="text-gray-700">{rec}</p>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </div>

      <div className="flex flex-wrap gap-4">
        <button
          onClick={() => {
            setCurrentSession(sessionId!);
            navigate("/");
          }}
          className="btn btn-primary flex items-center gap-2"
        >
          <MessageCircle size={16} />
          <span>Continue Conversation</span>
        </button>

        <button
          onClick={() => navigate("/")}
          className="btn btn-outline flex items-center gap-2"
        >
          <MessageCircle size={16} />
          <span>Start New Assessment</span>
        </button>

        <button
          onClick={() => navigate("/history")}
          className="btn btn-outline flex items-center gap-2"
        >
          <BarChart3 size={16} />
          <span>View History</span>
        </button>

        <button
          onClick={() => {
            // Create downloadable report
            const reportData = {
              sessionId: session.id,
              timestamp: session.timestamp,
              analysis: analysis,
            };
            const blob = new Blob([JSON.stringify(reportData, null, 2)], {
              type: "application/json",
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `dass-assessment-${session.id.slice(0, 8)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          }}
          className="btn btn-outline flex items-center gap-2"
        >
          <Download size={16} />
          <span>Download Report</span>
        </button>
      </div>

      <div className="mt-8 p-4 bg-gray-50 border border-gray-200 rounded-lg">
        <p className="text-sm text-gray-500">
          <strong>Important:</strong> This DASS-21 assessment is generated by an
          AI system and should not be considered a medical diagnosis. The
          Depression, Anxiety and Stress Scale (DASS-21) is a screening tool
          only. If you're experiencing mental health concerns, please consult
          with a qualified healthcare professional for proper evaluation and
          treatment.
        </p>
      </div>
    </motion.div>
  );
};

export default ResultsPage;
