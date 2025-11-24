# 🩺 Healthcare & First Aid Assistant

An interactive Streamlit application that functions as a healthcare & first‑aid oriented assistant. It lets you upload medical / first aid PDF documents (guidelines, manuals, protocols), builds a local semantic index with embeddings, and answers user questions conversationally. When the retrieved document context is weak or insufficient, it automatically falls back to live web search via Tavily to enrich the answer. A variant (`appWithEvaluation.py`) also auto‑generates QA examples from the ingested content and evaluates answer quality.

> Disclaimer: This tool does NOT provide personalized medical diagnosis or treatment. Always consult a qualified healthcare professional for medical decisions.

## ✅ Key Features
- PDF ingestion & parsing (via `PyPDFLoader`).
- Chunking & semantic embedding (`RecursiveCharacterTextSplitter` + `GoogleGenerativeAIEmbeddings`).
- Vector similarity search using `Chroma`.
- Conversational retrieval with memory (`ConversationBufferMemory`).
- Automatic web fallback powered by `TavilySearch` when answer quality heuristics trigger.
- Gemini 1.5 Flash (`ChatGoogleGenerativeAI`) as the LLM for answers & evaluation.
- Evaluation variant (`appWithEvaluation.py`) using `QAGenerateChain` + `QAEvalChain` to:
  - Generate synthetic QA pairs from your uploaded documents.
  - Run the conversational retrieval chain.
  - Grade predicted answers for quick quality insights.

## 🗂 Project Files
| File | Purpose |
|------|---------|
| `app.py` | Main conversational assistant with PDF upload + Tavily fallback.
| `app1.py` | Simplified earlier variant (similar logic, fewer heuristics). |
| `app2.py` | Near duplicate of `app.py` (alternative experimentation). |
| `appWithEvaluation.py` | Adds automated QA generation & evaluation UI section. |
| `requirements.txt` | Python dependencies (framework + LangChain ecosystem). |

All active assistant variants follow the same pipeline: Upload PDFs → Split → Embed → Store in Chroma → Chat with retrieval + memory → Optional web fallback.

## 🧱 Architecture Overview
1. **Upload PDFs**: User selects one or more PDF files via Streamlit.
2. **Load & Split**: Each PDF is loaded (`PyPDFLoader`) and chunked (`RecursiveCharacterTextSplitter`, size 1000, overlap 200).
3. **Filter (evaluation app)**: `appWithEvaluation.py` filters out trivial / empty chunks.
4. **Embedding**: Chunks embedded with Google Generative AI embeddings model: `models/embedding-001`.
5. **Vector Store**: Stored in a transient in‑memory / local `Chroma` instance.
6. **Retriever**: Top‐k similarity search (`k=3`).
7. **Prompting**: Custom system prompt injects: conversation history, retrieved context, user question.
8. **LLM Answer**: Gemini 1.5 Flash generates the answer.
9. **Fallback Logic**: If heuristics detect low‑quality / insufficient answers (short length, phrases like “I don’t know”), perform `TavilySearch` and append web snippet.
10. **Memory**: `ConversationBufferMemory` persists dialogue state so follow‑ups retain prior user details.
11. **Evaluation (optional)**: Generate QA pairs from document text and grade predictions.

## 🔧 Tech Stack
- **UI**: Streamlit
- **LLM & Embeddings**: Google Gemini (`langchain-google-genai`)
- **Retrieval**: Chroma vector store (`chromadb`)
- **Framework**: LangChain Classic abstraction layer (`langchain_classic`)
- **Web Search**: Tavily (`langchain-tavily` + `tavily-python`)
- **Document Loading**: `pypdf` via `PyPDFLoader`
- **Async Compatibility**: `nest-asyncio` to prevent event loop conflicts in Streamlit

### Why `langchain_classic` Imports?
Your installed LangChain distribution separates classic modules into `langchain_classic`. The original code used `from langchain.chains...` which failed (module missing). Imports were updated to:
```
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate
from langchain_classic.evaluation.qa import QAGenerateChain, QAEvalChain  # in evaluation variant
```
If you prefer older import paths, pin an earlier LangChain version (pre‑1.0 refactor) and revert these changes.

## 🛠 Setup (Windows PowerShell)
```powershell
# Clone (replace path as needed)
git clone <repo-url> "HealthcareAssistant"
cd HealthcareAssistant

# Create virtual environment (if not already present)
python -m venv .venv

# Activate
& .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env file (see sample below)
notepad .env
```

## ▶️ Run the App
Choose one variant (standard recommended):
```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py
```
Alternative variants:
```powershell
& .\.venv\Scripts\python.exe -m streamlit run app2.py
& .\.venv\Scripts\python.exe -m streamlit run app1.py
& .\.venv\Scripts\python.exe -m streamlit run appWithEvaluation.py
```
Then open the displayed Local URL (default `http://localhost:8501`).

## 📁 Using the Assistant
1. Start the app.
2. Upload one or more PDF medical references.
3. Wait for “Processing documents…” spinner to finish (chunks + embeddings build).
4. Ask a question in the chat input (e.g., “What are the steps for adult CPR?”).
5. If the model’s initial response is weak, a web supplement is appended.
6. For evaluation variant: Scroll down after a few interactions to view auto‑generated QA grading.

## 🌐 Web Fallback Heuristics
Triggers when answer is:
- Too short (< ~30 chars) or empty.
- Contains phrases like “I don’t know”, “no information”, “cannot access the internet”, etc.
If triggered, a Tavily search (`max_results=3`) runs; top result snippet is appended under “Web Supplement”.

## 🔒 Environment Variables (`.env`)
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
# Optional (for LangSmith tracing / analytics)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
# Optional custom port for Streamlit
# STREAMLIT_SERVER_PORT=8501
```
Both keys must be valid; Tavily key absence will show an error banner. Google API key must have access to Gemini 1.5 Flash and embeddings.

## 🧪 Evaluation Mode (`appWithEvaluation.py`)
Adds:
- `is_valid_chunk` filtering to avoid empty / trivial segments.
- Generates synthetic QA examples from first few document chunks.
- Runs retrieval chain per example.
- Grades predictions using `QAEvalChain`.
Output shown in a results section (reference answer vs model prediction vs grade).

## ❗ Limitations
- Not a substitute for professional medical expertise.
- Retrieval quality depends on PDF clarity and chunking.
- Web fallback gives snippet, not full real‑time verified medical guidance.
- No persistent database; embeddings rebuilt each session.

## 🔄 Possible Improvements
- Add caching / persistent Chroma directory.
- Add source document citation UI (currently suppressed by `return_source_documents=False`).
- Provide selectable model/temperature controls.
- Integrate guardrails / medical safety filters.
- Add GPU acceleration (if using heavy embedding models later).

## 🧹 Maintenance / Version Pinning
For more deterministic behavior, you may optionally pin versions in `requirements.txt` (example):
```
streamlit==1.51.0
langchain==1.0.8
langchain-community==0.4.1
langchain-google-genai==3.1.0
langchain-tavily==0.2.13
chromadb==1.3.5
python-dotenv==1.0.1
nest-asyncio==1.6.0
pypdf==6.4.0
tavily-python==0.7.13
```
If you do this and revert to legacy imports you may need to re‑test compatibility.

## 💬 Example Interaction
```
User: What should I do first when someone is unconscious?
Assistant: (Retrieves from uploaded first aid manual) First assess scene safety, then check responsiveness...
... (If weak) + Web Supplement: [Article Title] (url) snippet...
```

## ✅ Quick Start (Compressed)
```powershell
git clone <repo-url>
cd <repo>
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
echo GOOGLE_API_KEY=xxx > .env
echo TAVILY_API_KEY=yyy >> .env
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

## ⚖️ License
No explicit license included in repository snapshot. Add one (e.g., MIT) if you plan to share/distribute.

---
Feel free to request enhancements (persistent storage, multi‑PDF metadata, safety filters, etc.).
