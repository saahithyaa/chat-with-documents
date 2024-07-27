import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOllama

def save_file(file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

def setup_qa_chain(uploaded_files):
        # Load documents
        docs = []
        for file in uploaded_files:
            file_path = save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOllama(model="llama2"),
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        return qa_chain
    
st.title("Documents Bot")
with st.chat_message("assistant"):
        st.markdown("How can I help you?")
available = ["docx","pdf","csv"]
opt = st.sidebar.radio(
        label="Document Type",
        options=available,
        key="SELECTED_TYPE"
        )

if opt:
    uploaded_files = st.sidebar.file_uploader(label='Upload files', type=[opt], accept_multiple_files=True)
    if not uploaded_files:
        st.error("Please upload documents to continue!")
        st.stop()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    if uploaded_files and prompt:
            qa_chain = setup_qa_chain(uploaded_files)

            # display_msg(prompt, 'user')

            with st.chat_message("assistant"):
                result = qa_chain.invoke(
                    {"question":prompt},
                    # {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # to show references
                for idx, doc in enumerate(result['source_documents'],1):
                    filename = os.path.basename(doc.metadata['source'])
                    page_num = doc.metadata['page']
                    ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)
    # response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    # with st.chat_message("assistant"):
    #     st.markdown(response)
    # # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": response})