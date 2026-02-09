# AI RAG Engine

An enterprise-grade RAG system built with Google Vertex AI and Google ADK for intelligent document search and question-answering with grounded responses.

## Tech stack

- Python 3.8+
- Google Vertex AI
- Google ADK (Agent Development Kit)
- Gemini Models
- PyPDF2 / PyMuPDF

## Features

- **Multi-Modal Document Processing**: Extract and process text, images, and tables from PDF documents
- **Intelligent Chunking**: Optimized document segmentation with overlap for better context preservation
- **Hybrid Search**: Combines semantic and keyword-based search with configurable weighting
- **LLM Reranking**: Advanced result reranking using Gemini models for improved relevance
- **Grounding Verification**: Automatic verification of response claims against source material
- **Context Management**: Multi-turn conversation support with context tracking
- **Google ADK Integration**: Modern agent framework using Google's Agent Development Kit
- **Cloud Storage Integration**: Seamless GCS integration for document management

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PDF Documents  │────▶│  RAG Ingestion   │────▶│  Vertex AI RAG  │
│                 │     │     Pipeline      │     │     Corpus      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                        ┌──────────────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  Google ADK   │
                │     Agent     │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Gemini Model  │
                │  (Flash 2.5)  │
                └───────────────┘
```

## Requirements

- Python 3.8+
- Google Cloud Platform account
- Vertex AI API enabled
- Google Cloud Storage bucket

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-rag-engine
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with the following:
```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCS_BUCKET_NAME=your-bucket-name
```

## Usage

### Basic Usage

1. Place your PDF documents in the `docs/` directory

2. Run the RAG agent:
```bash
python rag_agent.py
```

### Programmatic Usage

```python
from rag_agent import ADKRAGAgent, create_rag_corpus, import_documents_to_corpus

# Initialize the system
corpus_id = create_rag_corpus(
    corpus_name="my-knowledge-base",
    description="Enterprise documentation"
)

# Upload and import documents from local docs folder
from rag_agent import upload_file_to_gcs
import os

# Upload local PDFs to GCS
document_paths = [
    "docs/technical_manual.pdf",
    "docs/product_specs.pdf",
    "docs/user_guide.pdf"
]

gcs_uris = []
for doc_path in document_paths:
    if os.path.exists(doc_path):
        gcs_uri = upload_file_to_gcs(doc_path, os.getenv('GCS_BUCKET_NAME'))
        gcs_uris.append(gcs_uri)

# Import uploaded documents to RAG corpus
import_documents_to_corpus(corpus_id, gcs_uris)

# Create and use the agent
adk_agent = ADKRAGAgent(
    corpus_id=corpus_id,
    project_id="your-project-id",
    location="us-central1"
)

agent = adk_agent.create_agent()
response = adk_agent.chat(agent, "What are the system requirements?")
print(response)
```

## Key Components

### RAGAgent
Core RAG functionality including:
- Hybrid search with semantic and keyword matching
- Context management for multi-turn conversations
- Result reranking using LLM scoring
- Grounding verification for hallucination prevention

### ADKRAGAgent
Google ADK wrapper that provides:
- Tool-based RAG search functionality
- Native integration with Gemini models
- Automatic tool calling and response synthesis
- Session-based conversation tracking

### Document Processing
- PDF text extraction with PyPDF2
- Image extraction with PyMuPDF (fitz)
- Table content parsing and structuring
- Multi-modal embedding creation

## Configuration

### Retrieval Parameters
Adjust in `configure_retrieval_parameters()`:
- `similarity_top_k`: Number of results to retrieve (default: 10)
- `vector_distance_threshold`: Similarity threshold (default: 0.5)
- `alpha`: Hybrid search weight (default: 0.5)

### Chunking Settings
Modify in `chunk_document()`:
- `chunk_size`: Characters per chunk (default: 1000)
- `overlap`: Overlap between chunks (default: 200)

### Model Configuration
Change models in initialization:
- RAG Agent: `gemini-2.5-flash`
- ADK Agent: `gemini-2.0-flash-001`
- Embeddings: `text-embedding-004`

## Project Structure

```
ai-rag-engine/
├── rag_agent.py           # Main RAG system implementation
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
├── docs/                  # Source PDF documents
├── extracted_images/      # Extracted images from PDFs
└── README.md             # This file
```

## Features in Detail

### Hybrid Search
Combines semantic embeddings with keyword matching for optimal retrieval:
```python
results = agent.hybrid_search(
    corpus_id=corpus_id,
    query="your query",
    semantic_weight=0.7,  # 70% semantic, 30% keyword
    top_k=10
)
```

### Grounding Verification
Ensures responses are factually grounded in source documents:
```python
verification = agent.verify_grounding(
    response=answer,
    sources=retrieved_docs
)
```

### Multi-Modal Processing
Handle documents with text, images, and tables:
```python
images = agent.extract_images_from_pdf(pdf_path, output_dir)
table_data = agent.process_table_content(table_text)
embedding = agent.create_multimodal_embedding(text, image_path, table_data)
```

## Limitations

- Rate limits: 5 requests/minute for RAG retrieval operations
- Requires 3-minute wait time after corpus document import for indexing
- PDF processing only (extend for other formats as needed)

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Open an issue on GitHub
- Check Google Vertex AI documentation: https://cloud.google.com/vertex-ai/docs

## Acknowledgments

- Built with Google Vertex AI RAG Engine
- Powered by Google ADK and Gemini models
- Uses PyPDF2 and PyMuPDF for document processing
