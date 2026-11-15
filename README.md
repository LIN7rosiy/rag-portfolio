# RAG PDF QA Application (AI 文档智能问答助手)

## 📌 项目简介
本项目是一个基于 **Streamlit + LangChain + DeepSeek + HuggingFace** 构建的文档智能问答工具。用户上传 PDF 后，系统会提取文本、进行向量化存储，并结合大模型给出基于文档内容的精准回答，有效避免大模型的“幻觉”问题。

👉 **在线体验 Demo**: [点击这里体验（部署后填入链接）]()

## 🛠️ 技术栈
- **前端界面**: Streamlit
- **核心框架**: LangChain
- **大语言模型 (LLM)**: DeepSeek API (`deepseek-chat`)
- **文本向量化 (Embedding)**: HuggingFace `BAAI/bge-small-zh-v1.5`（本地加载，无需网络 API）
- **向量数据库**: ChromaDB

## 🚀 我的核心工作
1. **工程复现与二次开发**：基于 `awesome-llm-apps` 模板，成功将原本绑定 OpenAI 的架构解耦重构。
2. **降本与本地化**：由于 OpenAI 地域限制和 DeepSeek 缺乏 Embedding 接口，我通过引入本地 `HuggingFaceEmbeddings`，彻底解决了向量化网络依赖问题，实现了 0 成本、高隐私的数据处理。
3. **完整 RAG Pipeline**：独立完成了 PDF 加载、文本切分 (Chunking)、向量化 (Vector Store)、相似度检索与 LLM 生成的全链路搭建。

## ⚠️ 遇到的挑战与解决方案（排错经验）
- **痛点 1：API 受限**。原项目使用 OpenAI 接口。我通过修改 `base_url` 和 `model`，成功接入 DeepSeek 实现对话；针对 DeepSeek 无 Embedding 接口的问题，成功切换至本地轻量级 BGE 模型。
- **痛点 2：网络墙与依赖冲突**。HuggingFace 模型下载受阻，通过配置 `HF_ENDPOINT` 国内镜像解决；遇到 LangChain 1.x 版本模块重构导致 `langchain.chains` 缺失，通过降级包版本 (`<0.4`) 和修改导入路径完成兼容。

## 💻 如何本地运行
1. `git clone` 本项目
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
