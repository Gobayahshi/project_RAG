import streamlit as st
from langchain.document_loaders import DirectoryLoader  # 문서 로딩
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트 분할
from langchain.embeddings import OpenAIEmbeddings  # 임베딩
from langchain.vectorstores import FAISS  # 벡터 저장소
from langchain.chat_models import ChatOpenAI  # LLM
from langchain.chains import ConversationalRetrievalChain
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your-api-key"

st.title("RAG 시스템")

# 1. 문서 로딩 및 처리
@st.cache_resource  # 캐싱을 통한 성능 최적화
def initialize_rag():
    # 문서 로딩
    loader = DirectoryLoader("your_documents_folder/", glob="*.txt")
    documents = loader.load()
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # 벡터 저장소 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # RAG 체인 생성
    llm = ChatOpenAI(temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain

# RAG 시스템 초기화
qa_chain = initialize_rag()

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사용자 입력
user_input = st.text_input("질문을 입력하세요:")

if user_input:
    # 질문 처리
    result = qa_chain({"question": user_input, "chat_history": []})
    
    # 응답 표시
    st.write("🤖 답변:", result["answer"])
    
    # 참조 문서 표시
    st.write("📚 참조 문서:")
    for doc in result["source_documents"]:
        st.write("- " + doc.page_content[:200] + "...")
