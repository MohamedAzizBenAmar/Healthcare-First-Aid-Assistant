import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain_classic.prompts import PromptTemplate
import nest_asyncio  # Add this import
nest_asyncio.apply()  # Apply the patch to allow nested event loops
# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

if not os.environ.get("TAVILY_API_KEY"):
    st.error("❌ TAVILY_API_KEY is missing. Please check your .env file.")

# Streamlit app config
st.set_page_config(page_title="🩺 Health Assistant Chatbot")
st.title("🩺 Healthcare & First Aid Assistant")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3)

# Prompt template with required input variables
prompt_template = """You are an intelligent, friendly, and highly helpful medical assistant.

Important rules:
- You must always try to answer the user's question.
- If you cannot find the answer in the provided documents, use web search via Tavily (you have access to this feature).
- Never say that you do not have access to the internet or web search: you can use Tavily to search online.
- Remember everything the user tells you during the conversation (name, preferences, symptoms, etc.).
- If the user asks for personal information like "what is my name?", answer based on what they have told you previously.

Conversation history:
{chat_history}

Available documents:
{context}

Question: {question}
Helpful and professional answer (in English):
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=prompt_template,
)


# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True  
    )
memory = st.session_state.memory
# Load documents once
if "retriever" not in st.session_state:
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing documents..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    loader = PyPDFLoader(tmp_file.name)
                    all_docs.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(all_docs)
            st.session_state.split_docs = split_docs

            if not split_docs:
                st.error("No valid content found in uploaded documents.")
                st.stop()

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = Chroma.from_documents(
                split_docs,
                embedding=embeddings
            )
            st.session_state.retriever = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            st.rerun()

# Interface Chat
elif "retriever" in st.session_state:
    retriever = st.session_state.retriever
    memory = st.session_state.memory
    tavily_tool = TavilySearch(max_results=3)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
        chain_type="stuff",  # or "map_reduce", "refine", "map_rerank"
        output_key="answer"  # <-- Add this line
    )

    def should_use_web(question: str, answer: str) -> bool:
        answer_lower = answer.lower()
        return (
            not answer
            or len(answer.strip()) < 30
            or "no information" in answer_lower
            or "i don't know" in answer_lower
            or "sorry" in answer_lower
            or "i cannot access the internet" in answer_lower
            or "i cannot directly access the internet" in answer_lower
            or "i do not have access to the internet" in answer_lower
            or "i would need to perform a web search" in answer_lower
            or "i recommend checking the who website" in answer_lower
            or "i recommend searching online" in answer_lower
        )


    def search_web_fallback(query: str):
        results = tavily_tool.invoke({"query": query})
        print("result web:", results)
        if "results" in results and results["results"]:
            top = results["results"][0]
            return f"🌐 Web Result: [{top['title']}]({top['url']})\n{top['content'][:500]}..."
        return "❌ No information found online."

    def get_answer_or_fallback(question):
        if not isinstance(question, str) or not question.strip():
            return "❌ Invalid or empty question."
        result = qa_chain.invoke({"question": question})
        answer = result["answer"].strip()
        sources = result.get("source_documents", [])

        if should_use_web(question, answer):
            web_result = search_web_fallback(question)
            answer = f"{answer}\n\n🔎 Web Supplement:\n{web_result}" if answer else web_result

        # Format sources for display
        if sources:
            sources_text = "\n\n**Sources used:**\n"
            for i, doc in enumerate(sources, 1):
                # Try to show document metadata or a snippet
                snippet = doc.page_content[:200].replace('\n', ' ')
                sources_text += f"- Source {i}: {snippet}...\n"
            answer += sources_text

        return answer


    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.chat_input("Ask your medical or first aid question")
    if user_input:
        with st.spinner("Thinking..."):
            response = get_answer_or_fallback(user_input)
            st.session_state.chat_history.append((user_input, response))

    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(bot_msg)
# If no document loaded
if "retriever" not in st.session_state:
    st.info("📄 Please upload at least one PDF document to get started.")
