# OMNIBRAIN

1. Overall Project Understanding Project Goal 
Build an enterprise-grade Agentic Multi-Modal Retrieval-Augmented Generation (RAG) system capable of understanding PDFs, text, tables, charts, images, and SQL databases. Instead of relying on a single LLM, the system uses a LangGraph Supervisor Agent to orchestrate specialized agents (Search, Vision, SQL) that collaborate to answer complex questions with citations while minimizing hallucinations. 
Core Components 
• Document ingestion (PDF parsing, text/image/table extraction) 
• Embedding generation and Vector Database (Qdrant) 
• Multi-modal retrieval (text + image embeddings) 
• Vision-Language Model for charts and images 
• Text-to-SQL agent for structured database queries 
• LangGraph Supervisor for routing and orchestration 
• Self-RAG retry mechanism 
• Guardrails and Langfuse monitoring 
• FastAPI backend and Streamlit frontend 
Expected Workflow 
User uploads documents → Ingestion Pipeline → Vector Database → User asks a question → LangGraph Supervisor routes to Search/Vision/SQL agents → Results are synthesized → Final cited response returned through the UI.