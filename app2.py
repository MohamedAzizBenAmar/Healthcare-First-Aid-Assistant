import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate

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
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Prompt template with required input variables
prompt_template = """Tu es un assistant médical intelligent, amical et très utile.

Règles importantes :
- Tu dois toujours essayer de répondre à la question de l'utilisateur.
- Si tu ne trouves pas la réponse dans les documents fournis, utilise la recherche web via Tavily (tu as accès à cette fonctionnalité).
- Ne dis jamais que tu n’as pas accès à internet ou à la recherche web : tu peux utiliser Tavily pour chercher en ligne.
- Souviens-toi de tout ce que l'utilisateur te dit pendant la conversation (nom, préférences, symptômes, etc.).
- Si l'utilisateur demande une information personnelle comme "quel est mon nom ?", réponds en te basant sur ce qu'il t’a dit précédemment.

Historique de la conversation :
{chat_history}

Documents disponibles :
{context}

Question : {question}
Réponse utile et professionnelle :
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=prompt_template,
)


# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = Chroma.from_documents(split_docs, embedding=embeddings)

            st.session_state.retriever = vectordb.as_retriever()
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
        return_source_documents=False
    )

    def should_use_web(question: str, answer: str) -> bool:
        answer_lower = answer.lower()
        return (
            not answer
            or len(answer.strip()) < 30
            or "aucune information" in answer_lower
            or "je ne sais pas" in answer_lower
            or "désolé" in answer_lower
            or "je ne peux pas accéder à internet" in answer_lower
            or "i cannot directly access the internet" in answer_lower
            or "je n'ai pas accès à internet" in answer_lower
            or "i do not have access to the internet" in answer_lower
            or "i would need to perform a web search" in answer_lower
        )


    def search_web_fallback(query: str):
        results = tavily_tool.invoke({"query": query})
        if "results" in results and results["results"]:
            top = results["results"][0]
            return f"🌐 Résultat Web : [{top['title']}]({top['url']})\n{top['content'][:500]}..."
        return "❌ Aucune information trouvée en ligne."

    def get_answer_or_fallback(question):
        result = qa_chain.invoke({"question": question})
        answer = result["answer"].strip()

        if should_use_web(question, answer):
            web_result = search_web_fallback(question)
            # Re-ask the LLM, adding the web result to the context
            context = f"{web_result}\n\n"
            # You may want to add the original context from your retriever as well
            prompt_input = {
                "chat_history": memory.buffer_as_str,
                "context": context,
                "question": question
            }
            final_answer = llm.invoke(prompt.format(**prompt_input))
            return final_answer
        return answer


    # Historique de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Entrée utilisateur
    user_input = st.chat_input("Posez votre question médicale ou de premiers secours")
    if user_input:
        with st.spinner("Réflexion en cours..."):
            response = get_answer_or_fallback(user_input)
            st.session_state.chat_history.append((user_input, response))

    # Affichage historique
    for user_msg, bot_msg in st.session_state.chat_history:
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(bot_msg)

# Si aucun document chargé
if "retriever" not in st.session_state:
    st.info("📄 Veuillez téléverser au moins un document PDF pour commencer.")
