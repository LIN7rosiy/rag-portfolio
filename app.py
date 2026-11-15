# -*- coding: utf-8 -*-
# Based on the 'chat_with_pdf' template from Shubhamsaboo/awesome-llm-apps
# Original repo: https://github.com/Shubhamsaboo/awesome-llm-apps

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

# 设置页面标题
st.title("📄 AI 文档智能问答助手 (RAG)")
st.write("上传 PDF 文档，AI 会根据文档内容为你解答问题。")

# 侧边栏输入 API Key
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[获取 OpenAI API Key](https://platform.openai.com/account/api-keys)")
    st.markdown("[获取免费的替代 Key (如使用国内模型需改代码)]")

# 初始化 Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 处理 PDF 上传与向量化
if openai_api_key:
    #os.environ["OPENAI_API_KEY"] = openai_api_key
    
    uploaded_file = st.file_uploader("请上传一个 PDF 文件", type="pdf")
    
    if uploaded_file is not None:
        if st.session_state.vectorstore is None:
            with st.spinner("正在解析文档并建立知识库，请稍候..."):
                # 将上传的文件保存为临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # 加载文档并切分
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # 建立向量数据库
                #embedding = OpenAIEmbeddings()
                # embedding = OpenAIEmbeddings(
                #     api_key=openai_api_key,
                #     base_url="https://api.deepseek.com",
                #     model="deepseek-embedding"
                # )
                embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
                st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
            st.success("知识库构建完成！")
        
        # 问答界面
        user_query = st.chat_input("请输入关于文档的问题...")
        
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            with st.spinner("AI 正在思考..."):
                #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                llm = ChatOpenAI(
                    api_key=openai_api_key,
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat", # 注意这里用官方标准的 deepseek-chat
                    temperature=0
                )

                retriever = st.session_state.vectorstore.as_retriever()
                
                system_prompt = (
                    "你是一个智能助手，请严格根据以下检索到的上下文回答用户的问题。"
                    "如果不知道答案，就说不知道。保持回答简洁。\n\n"
                    "{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # 展示聊天记录
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
else:
    st.info("请在左侧输入 OpenAI API Key 以上传文档并开始问答。")
