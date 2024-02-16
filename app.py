import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import huggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

#load the pdf
loader=DirectoryLoader("data\fitness_data.pdf",glob="*.pdf",loader_cls=PyPDFLoader)
documents=loader.load()

#split text into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks=text_splitter.split_documents(documents)

#create embeddings (we will use a sentence transformer from huggingface)
embeddings=huggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 model_kwargs={'device':'cpu'})

#create a vector store
vector_store=FAISS.from_documents(text_chunks,embeddings)

#create llm
llm=CTransformers(model="EleutherAI/llama-lm",
                  model_type="llama",
                  config={'max_new_tokens':128,"temperature":0.01})

memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type="stuff",
                    retriever=vector_store(search_kwargs={"k:2"}),#top 2 answers
                    memory=memory)  

st.title("Fitness Chatbot 💪🏽")