# Frontend API Integration

This document describes how the React frontend integrates with the FastAPI backend for the MindScan DASS-21 mental health assessment application.

## Architecture Overview

The frontend uses a modern React stack with:

- **TanStack Query** for server state management
- **Axios** for HTTP requests
- **TypeScript** for type safety
- **React Context** for local state management

## Key Components

### 1. API Service (`src/services/api.ts`)

- Centralized Axios client with interceptors
- Type-safe API methods
- Error handling and request/response logging
- Base URL: `http://localhost:8000/api`

### 2. TanStack Query Hooks (`src/hooks/useConversations.ts`)

- `useCreateConversation` - Start new DASS assessment
- `useContinueConversation` - Send messages and receive AI responses
- `useConversation` - Fetch conversation details
- `useDeleteConversation` - Delete conversations
- `useAnalytics` - Get assessment analytics
- `useHealthCheck` - Monitor backend health

### 3. Chat Context (`src/context/ChatContext.tsx`)

- Manages conversation state
- Integrates with TanStack Query hooks
- Handles loading states and errors
- Converts API data to UI-friendly formats

### 4. Type Definitions (`src/types/api.ts`)

- TypeScript interfaces matching FastAPI models
- Request/response types
- DASS prediction and analytics types

## API Endpoints Used

### Conversation Management

- `POST /api/conversations/new` - Create new conversation
- `POST /api/conversations/continue` - Continue conversation
- `GET /api/conversations/{id}/full` - Get conversation details
- `DELETE /api/conversations/{id}` - Delete conversation

### Analytics

- `GET /api/analytics` - Get overall analytics
- `GET /api/health` - Health check

## Data Flow

1. **Start Assessment**: User clicks "Start New Assessment"

   - Frontend calls `createConversation` with initial message
   - Backend creates conversation and returns AI response
   - UI displays conversation interface

2. **Continue Conversation**: User sends messages

   - Frontend calls `continueConversation` with user message
   - Backend processes message, updates DASS responses
   - AI generates contextual response
   - When all 21 questions answered, backend automatically generates predictions

3. **View Results**: Assessment completion

   - Backend returns `is_assessment_complete: true` with predictions
   - Frontend automatically navigates to results page
   - Results page displays DASS scores, levels, and recommendations

4. **History Management**: View past assessments
   - Frontend fetches analytics and conversation summaries
   - Users can view, continue, or delete conversations

## Error Handling

- Network errors caught by Axios interceptors
- API errors displayed in UI with dismiss functionality
- Loading states shown during API calls
- Graceful fallbacks for missing data

## State Management

- **Server State**: Managed by TanStack Query with caching
- **Local State**: React Context for current session
- **UI State**: Component-level useState for forms and interactions

## Configuration

The API base URL is configured in `src/services/api.ts`:

```typescript
baseURL: "http://localhost:8000/api";
```

For production, this should be updated to your deployed backend URL.

## Development

1. Ensure the FastAPI backend is running on `http://localhost:8000`
2. Start the frontend development server
3. The app will automatically connect to the backend
4. Check browser console for API request logs

## Features

- ✅ Real-time conversation with AI
- ✅ Automatic DASS-21 assessment completion
- ✅ Persistent conversation storage
- ✅ Analytics and reporting
- ✅ Error handling and loading states
- ✅ Responsive design
- ✅ Type-safe API integration
