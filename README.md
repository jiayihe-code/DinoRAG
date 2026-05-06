# 🦕 DinoRAG: 智能学术文献阅读 Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![DeepSeek API](https://img.shields.io/badge/LLM-DeepSeek-black.svg)](https://deepseek.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DinoRAG** (小恐龙文献阅读 Agent) 是一个专为科研人员打造的本地文献知识库与问答智能体。
项目致力于解决传统 RAG（检索增强生成）系统在学术场景下面临的“双栏乱码劫持”、“跨语言检索遗漏”以及“单次检索信息不足”等痛点，通过引入 **CRAG (Corrective RAG) 架构** 和 **两阶段重排检索**，打造了一个具备“自我反思”与“动态可视化”能力的学术助理。

## ✨ 核心特性 (Key Features)

*   🧠 **Agentic CRAG 工作流 (自反思检索)**
    *   内置质量评估门控（Quality Gate），对检索召回的片段进行 LLM 动态打分。
    *   当检索质量不达标时，触发**跨语言查询重写 (Query Rewriting)**，结合历史失败检索词自动调优搜索策略，最高支持 3 轮自我纠偏循环。
*   🎯 **两阶段高精度检索 (Two-Stage Retrieval)**
    *   **粗排**：使用 `all-MiniLM-L6-v2` 结合 ChromaDB 进行高效的轻量级向量召回 (Top-20)。
    *   **精排**：引入 `BAAI/bge-reranker-base` (Cross-Encoder) 对召回片段进行语义重排，截取 Top-5 高质量上下文，彻底解决高频关键词导致的“语义劫持”问题。
*   📄 **学术级 PDF 块级解析 (Smart PDF Parsing)**
    *   摒弃传统按行读取的解析器，采用 `PyMuPDF (fitz)` 进行智能块级 (Block-level) 解析。
    *   完美兼容 arXiv 格式的**双栏排版论文**，自动还原被切断的英文长句，过滤无意义的公式残骸与图注杂音。
*   📊 **动态逻辑可视化 (Mermaid Graph)**
    *   自动提取 LLM 总结的关键信息，在 Streamlit 侧边栏实时渲染 Mermaid 逻辑流程图 (Flowchart)，实现“图文并茂”的文献梳理体验。
*   🏗️ **工程化解耦架构**
    *   采用 ETL 数据入库 (`ingest.py`) 与 Web 交互 (`app.py`) 分离的架构设计，提升系统并发响应速度与稳定性。

## 🛠️ 技术栈 (Tech Stack)

*   **前端交互**: Streamlit
*   **大语言模型**: DeepSeek API (`deepseek-chat` / `deepseek-reasoner`)
*   **向量数据库**: ChromaDB (本地持久化)
*   **Embedding & Reranker**: SentenceTransformers
*   **文档解析**: PyMuPDF

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
克隆本项目并安装依赖：
```bash
git clone [https://github.com/YourUsername/DinoRAG.git](https://github.com/YourUsername/DinoRAG.git)
cd DinoRAG
pip install -r requirements.txt
