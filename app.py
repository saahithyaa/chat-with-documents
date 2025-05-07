import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter
import tempfile
import os

st.sidebar.header("🔧 Configuration")
OPENAI_API_KEY = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("📁 Upload your PDF", type=["pdf"])

if not OPENAI_API_KEY:
    st.sidebar.warning("Please enter your OpenAI API key.")
    st.stop()

if not uploaded_file:
    st.sidebar.info("Upload a PDF to start chatting.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    pdf_path = tmp_file.name

loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")

template = """
You are a helpful and factual assistant that answers questions based solely on the provided context.

Instructions:
- Use only the information in the given context and chat history. Do not rely on outside knowledge or make assumptions.
- If the context does not contain enough information to fully answer the question, respond with: "I don't know."
- If needed, ask for clarification instead of guessing.
- Structure your answer in clear, numbered or bulleted points for readability.
- Be comprehensive and include all relevant details from the context.

Context:
{context}

Question:
{question}

Chat History:
{chat_history}
"""

prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

chain = (
    {
        "context": itemgetter("context") ,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | prompt
    | model
    | parser
)

if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your document assistant. I can help find something or answer a question based on your file."}
        ]

def page2():
    st.title("📄 View your PDF")
    pdf_viewer(pdf_path, height=1000)

def format_chat_history():
    return "\n".join([
        f"User: {msg['content']}" if msg["role"] == "user" else f"AI: {msg['content']}"
        for msg in st.session_state.messages
    ])

def format_documents(documents):
    formatted = []
    for doc in documents:
        page_number = doc.metadata.get('page_label') or 'Unknown'
        content = doc.page_content.strip()
        if content.startswith(str(page_number)):
            content = content[len(str(page_number)):].lstrip()

        formatted.append(f"\n:blue[Page {page_number}]\n{content}")

    return formatted

def page1():
    st.title("📄 Chat with your PDF")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
              with st.popover("_View Sources_"):
                  for source in msg["sources"]:
                      st.caption(source)
                      st.divider()


    user_input = st.chat_input("Enter your query here")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = format_chat_history()
                context = retriever.invoke(user_input)
                response = chain.invoke({"question": user_input, "context": context, "chat_history": chat_history})
                st.markdown(response)
                sources = format_documents(context)
                with st.popover("_View Sources_"):
                    for source in sources:
                        st.caption(source)
                        st.divider()
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

def main():
    pages = {
        "Navigation": [
            st.Page(page1, title="Chat"),
            st.Page(page2, title="View PDF"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()

# Cleanup temp file 
# os.unlink(pdf_path)
