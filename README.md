<div align="center">

# 🩺 Healthcare & First Aid Assistant

Conversational Retrieval‑Augmented Generation (RAG) app for healthcare & first‑aid reference material. Upload trusted medical / first‑aid PDFs, ask natural language questions, and get grounded answers. When local context is weak, the system enriches responses with a targeted web search (Tavily). An evaluation mode can auto‑generate QA pairs and score answer quality.

</div>

> DISCLAIMER: Educational support only. Not a source of diagnosis, treatment, or emergency decision‑making. Always consult qualified healthcare professionals.

---

## 1. Why This Project Matters
- Demonstrates end‑to‑end RAG design (ingestion → chunking → embedding → retrieval → conversation → fallback search).
- Shows practical use of Google Gemini (chat + embeddings) integrated with LangChain Classic.
- Implements quality heuristics and dynamic search fallback (Tavily) for robustness.
- Includes automated answer evaluation (synthetic QA generation + grading pipeline).
- Highlights engineering decisions: lean architecture, modularity, environment isolation, prompt design, conversational state management.

---

## 2. Core Features
- PDF ingestion & parsing (`PyPDFLoader`).
- Chunking with overlap (`RecursiveCharacterTextSplitter`, 1000 / 200).
- Embeddings via Google Generative AI (`models/embedding-001`).
- Vector similarity search (`Chroma`).
- Conversational Retrieval (`ConversationalRetrievalChain`) + dialogue memory (`ConversationBufferMemory`).
- Web fallback: heuristic triggers → Tavily search (top snippets appended).
- Evaluation mode (`appWithEvaluation.py`): synthetic QA generation + automatic grading (`QAGenerateChain`, `QAEvalChain`).
- Clean environment management (.venv ignored from git; accidental tracking removed).

---

## 3. Tech Stack
| Layer | Tools |
|-------|-------|
| UI | Streamlit |
| LLM / Embeddings | Google Gemini (Flash + Embedding) |
| Retrieval | Chroma Vector Store |
| Orchestration | LangChain Classic modules |
| Web Search | Tavily API |
| Document Parsing | PyPDF / LangChain loaders |
| Utilities | `python-dotenv`, `nest-asyncio` |

Why `langchain_classic`? Current LangChain distribution splits classic abstractions (chains, prompts, memory) into a separate namespace. Imports updated accordingly to avoid runtime import errors.

---

## 4. Architecture Flow
1. Upload PDFs → load & chunk.
2. Generate embeddings → store in Chroma.
3. User query + history → retrieval (k=3) → prompt assembly.
4. Gemini LLM answers.
5. Heuristics (answer length / low‑confidence phrases) → optional Tavily search enrichment.
6. Evaluation mode: generate QA pairs → run chain → grade responses.

Pseudo Diagram:
```
PDFs ─► Loader ─► Chunker ─► Embeddings ─► Chroma ─► Retriever ─► Prompt ─► Gemini
                                                  │                         │
                                                  └──── Memory ◄────────────┘
                                 (Heuristic Failure) ─► Tavily Search ─► Merge Snippet
                         (Evaluation) ─► QA Generator ─► Chain ─► Grader ─► Scores
```

---

## 5. Skills Demonstrated
- RAG pipeline engineering & retrieval tuning.
- Prompt composition & conversational context injection.
- Fallback strategy design (quality heuristics → external search).
- Automated evaluation scaffolding (synthetic data + grading loop).
- Dependency / environment management & refactor for breaking API changes.
- Git hygiene (removal of accidentally tracked virtual environment).

---

## 6. Repository Structure
```
app.py                 # Main Streamlit RAG assistant
appWithEvaluation.py   # Adds QA generation + grading UI section
requirements.txt       # Dependencies
README.md              # Project documentation (this file)
chroma_db/             # Local Chroma persistence (ignored if reconfigured)
data/                  # (Optional) Sample documents location
.venv/                 # Virtual environment (ignored)
```

Legacy experimental files (`app1.py`, `app2.py`) were removed to reduce noise.

---

## 7. Quick Start (Windows PowerShell)
```powershell
git clone <REPO_URL> HealthcareAssistant
cd HealthcareAssistant
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
notepad .env   # add keys (see below)
& .\.venv\Scripts\python.exe -m streamlit run app.py
```
Open the printed local URL (default http://localhost:8501).

Evaluation mode:
```powershell
& .\.venv\Scripts\python.exe -m streamlit run appWithEvaluation.py
```

---

## 8. Environment Variables (.env)
```dotenv
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
# Optional tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
```
Gemini key must have access to chat + embeddings. Tavily key required for fallback enrichment.

---

## 9. Usage Workflow
1. Launch app.
2. Upload one or more trusted medical PDFs.
3. Wait for embedding build (spinner).
4. Ask questions conversationally (follow‑ups leverage memory).
5. If answer flagged as weak, appended “Web Supplement” from Tavily.
6. (Evaluation mode) Scroll to results section for generated QA pairs and grading metrics.

Heuristic triggers (simplified): very short output OR phrases like “I don’t know”.

---

## 10. Evaluation Mode Details
Adds chunk validity filter, synthetic QA generation, retrieval run, and grading (reference vs prediction) for rapid qualitative feedback. Useful to gauge coverage and identify weak retrieval segments early.

---

## 11. Safety & Limitations
- Not a diagnostic or emergency decision system.
- Content quality depends on source PDFs (format / clarity).
- No persistent long‑term knowledge base (embeddings rebuilt each session by default).
- Web enrichment provides snippets, not vetted clinical review.

Future safety improvements could include medical disclaimer injection per answer, confidence scoring, and integration with vetted guideline APIs.

---

## 12. Roadmap / Potential Enhancements
- Persistent vector store & incremental updates.
- Source citation panel (show chunk origins).
- Adjustable retrieval depth & temperature controls.
- Lightweight guardrail / filtering layer (contraindications, hallucination flags).
- Usage metrics dashboard (session length, fallback frequency).
- Alternative embedding models for offline mode.

---

## 13. Version / Maintenance Notes
Refactored imports to `langchain_classic` due to package restructuring. Pin versions if reproducibility is critical.
Example pinned set:
```text
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

---

## 14. Example Interaction
```
User: What are the first steps in adult CPR?
Assistant: (Grounded response using uploaded PDF context...)
Web Supplement (if triggered): <title> <snippet>
```

---

## 15. Project Status
✅ Core functionality working (PDF → RAG → conversation → fallback).
⚠️ Embedding quota may hit limits on free tier (Gemini embeddings). Consider caching or alternative models.
🧪 Evaluation prototype active.

---

## 16. Contributing / License
No license currently declared. Add MIT / Apache‑2.0 before external distribution. Contributions welcome via PR once licensed.

---

## 17. Contact
For questions or collaboration: open an issue or reach out via GitHub profile.

---

If you would like a live demo, deployment guide, or expanded evaluation metrics, feel free to request them.

