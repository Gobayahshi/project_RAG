import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# API 키 입력 및 설정
def set_openai_api_key():
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = ''
    
    api_key_input = st.text_input(
        "OpenAI API 키를 입력하세요:",
        type="password",
        value=st.session_state.OPENAI_API_KEY,
        key="api_key_input"
    )

    if api_key_input:
        st.session_state.OPENAI_API_KEY = api_key_input
        os.environ["OPENAI_API_KEY"] = api_key_input
        return True
    else:
        st.warning("API 키를 입력해주세요!")
        return False

# API 키 설정 확인
if not set_openai_api_key():
    st.stop()

# 1. 문서 로딩 및 처리
@st.cache_resource  # 캐싱을 통한 성능 최적
def initialize_rag():
    # 문서 로딩 경로를 'documents' 폴더로 수정
    loader = DirectoryLoader("./documents/", glob="*.txt")  # 상대 경로 사용
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
