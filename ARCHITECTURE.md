# System Architecture

This document provides a detailed overview of the architecture for the RAG-based University Assistant Bot. The system is designed with a modular, scalable, and resilient architecture to handle complex queries efficiently.

## 1. Architectural Diagram

The following diagram illustrates the high-level architecture and the flow of data between components:

```mermaid
graph TD
    subgraph User Interaction
        A[Telegram User] -- User Query (Text) --> B[Telegram Bot Interface];
        B -- Formatted Answer (Text) --> A;
    end

    subgraph Core Logic
        B --> C{Resilience Manager};
        C -- Wrapped Call --> D[Enhanced RAG Pipeline];
        D -- Result/Error --> C;
        C -- Result/Fallback --> B;
    end

    subgraph Pipeline Components
        D -- "1. Manages History" --> E[Conversation Manager];
        D -- "2. Retrieves Docs" --> F[Retriever];
        D -- "4. Generates Final Answer" --> G[LLM Model];
        F -- "2a. Embeds Query" --> H[Embedding Model];
    end
    
    subgraph Data Sources
        I[HTML Documents] --> J[Preprocessing Engine];
        J --> K[Vector DB (ChromaDB)];
        F -- "2b. Searches Docs" --> K;
    end

    subgraph Monitoring
        L[Performance Monitor]
        D -- Sends Metrics --> L
        C -- Sends Metrics --> L
    end

    G -- Wrapped by Resilience --> C
    H -- Wrapped by Resilience --> C

    style A fill:#cde,stroke:#333,stroke-width:2px
    style I fill:#f9f,stroke:#333,stroke-width:2px
```

## 2. Component Descriptions

### a. User Interaction Layer

-   **Telegram Bot Interface (`src/interface/bot.py`)**:
    -   Serves as the primary entry point for users.
    -   Handles incoming messages and commands from Telegram (`/start`, `/help`, `/stats`, etc.).
    -   Manages user sessions and formats the final response for the user.
    -   Protected by the **Resilience Manager** to handle Telegram API errors or pipeline failures gracefully.

### b. Core Logic Layer

-   **Enhanced RAG Pipeline (`src/pipeline/ragpipeline.py`)**:
    -   Orchestrates the entire process of generating an answer.
    -   It receives a user query, manages the conversation history, retrieves relevant documents, generates a prompt, and uses the LLM to get the final answer.
    -   Integrates query caching to return instant responses for repeated questions.

-   **Resilience Manager (`src/utils/resilience.py`)**:
    -   A crucial cross-cutting concern that wraps all critical components (LLM, Embedding Model, RAG Pipeline).
    -   Provides **Circuit Breaker**, **Retry**, and **Fallback** mechanisms to ensure the system is fault-tolerant.

### c. Data Processing & Retrieval

-   **Preprocessing Engine (`src/preprocess/`)**:
    -   **HTML Parser**: Reads raw HTML files from the `data/` directory and extracts clean text and table data.
    -   **Chunking**: Implements a hierarchical chunking strategy, splitting documents into larger parent chunks and smaller, more specific child chunks. This improves the quality of retrieved context.

-   **Embedding Model (`src/models/embedding.py`)**:
    -   Responsible for converting text (queries and document chunks) into numerical vectors (embeddings).
    -   Uses a sentence-transformer model.
    -   Highly optimized with in-memory caching, disk caching, and batch processing.

-   **Vector DB (ChromaDB) (`src/preprocess/db_manager.py`)**:
    -   A specialized database that stores the text chunks and their corresponding embeddings.
    -   Enables efficient similarity searches to find document chunks that are most relevant to a user's query embedding.

-   **Retriever (`src/retrieval/retriever.py`)**:
    -   Takes a user query, embeds it using the `Embedding Model`, and searches the `Vector DB` to find the top-k most relevant document chunks.

### d. Core Models

-   **LLM Model (`src/models/llm.py`)**:
    -   The core generative model (e.g., YandexGPT, Llama).
    -   Receives a prompt containing the user's query, conversation history, and the retrieved context.
    -   Generates a human-like answer based on the provided information.

-   **Conversation Manager (`src/models/conversation.py`)**:
    -   Tracks and stores conversation history for each user.
    -   Provides the necessary context for follow-up questions.
    -   Supports persistence to disk and has a TTL (Time-To-Live) for old conversations.

## 3. Data Flow

Here is the step-by-step data flow for a typical user query:

1.  **User Sends Query**: A user sends a message to the bot on Telegram.
2.  **Bot Receives Query**: The `Telegram Bot Interface` receives the query and the user's ID.
3.  **Pipeline Invocation**: The bot calls the `EnhancedRAGPipeline`'s `answer()` method, which is protected by the `ResilienceManager`.
4.  **History & Cache Check**: The pipeline first checks its query cache for an existing answer. If not found, it retrieves the user's conversation history from the `ConversationManager`.
5.  **Document Retrieval**:
    a. The `Retriever` embeds the user's query using the `Embedding Model`.
    b. It then uses this embedding to search `ChromaDB` for the most relevant document chunks.
6.  **Prompt Generation**: The pipeline constructs a detailed prompt containing:
    -   The original user query.
    -   The retrieved document chunks (context).
    -   Recent messages from the conversation history.
7.  **LLM Generation**: The prompt is sent to the `LLM Model`, which generates the final answer. This call is also protected by the `ResilienceManager`.
8.  **Response & Caching**:
    a. The generated answer is returned to the pipeline.
    b. The answer is cached for future use.
    c. The new user query and bot answer are added to the user's conversation history.
9.  **User Receives Answer**: The final answer is sent back to the user via the `Telegram Bot Interface`.

This architecture ensures that the system is modular, allowing individual components to be updated or replaced with minimal impact on the rest of the system. The resilience layer adds a robust safety net, making the bot reliable even when external services or internal models face issues. 