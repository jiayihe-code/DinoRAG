import json
import re

import streamlit as st
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from openai import OpenAI

# ==================== 1. 基础页面配置 ====================
st.set_page_config(page_title="DeepSeek Agentic RAG", layout="wide")
st.title("小恐龙文献阅读 Agent 🦕")
st.markdown("---")


# ==================== 2. 初始化环境与模型 ====================
@st.cache_resource
def load_models():
    # 基础向量模型与重排模型
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    rerank_model = CrossEncoder('BAAI/bge-reranker-base')
    return embed_model, rerank_model


embed_model, rerank_model = load_models()

# 连接离线数据库 (由 ingest.py 生成)
DB_PATH = "./research_db"
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(name="academic_papers")

# ==================== 3. 侧边栏：配置中心 ====================
with st.sidebar:
    st.header("⚙️ Agent 配置")
    api_key = st.text_input("DeepSeek API Key", type="password")
    model_choice = st.selectbox("选择逻辑大脑", ["deepseek-chat", "deepseek-reasoner"])

    st.divider()

    # 知识库状态统计
    try:
        doc_count = collection.count()
        st.success(f"📚 知识库连接正常")
        st.metric("已存片段", doc_count)
    except:
        st.error("⚠️ 未发现数据库，请先运行 ingest.py")


# ==================== 4. Agent 内部工具函数 ====================

def retrieve_and_rerank(query, top_n=5):
    """基础检索与重排流程"""
    query_vector = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=20)

    if not results['documents'][0]:
        return [], ""

    docs = results['documents'][0]
    metas = results['metadatas'][0]

    # Rerank
    pairs = [[query, doc] for doc in docs]
    scores = rerank_model.predict(pairs)
    scored = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

    top_results = scored[:top_n]
    context = ""
    for s, d, m in top_results:
        context += f"\n【来源：{m['source']}】\n{d}\n"
    return top_results, context


def grade_context(question, context, api_key):
    """Agent 反思节点：评价检索到的内容是否相关"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = f"""你是一个科研评价员。请判断以下参考资料是否足以回答用户的问题。

    问题：{question}
    资料：{context}

    请仅返回 JSON 格式：{{"score": 分数(0-10), "reason": "简短原因", "is_sufficient": true/false}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={'type': 'json_object'}
    )
    return json.loads(response.choices[0].message.content)


def rewrite_query(question, failed_queries, api_key):
    """Agent 纠偏节点：结合失败经验，生成更专业的跨语言搜索词"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 告诉大模型之前失败的尝试，防止重复犯错
    failed_history = "\n".join([f"- {q}" for q in failed_queries]) if failed_queries else "无"

    prompt = f"""你是一个顶级的学术文献检索专家。用户的问题是：'{question}'。

    由于之前的检索效果不佳，你需要重写一个全新的搜索词。
    【重要规则】：
    1. 我们的知识库主要是全英文的学术论文。如果你判断原问题涉及专业术语（如“数据集”、“过滤”），请务必将其翻译为专业的**英文关键词**。
    2. 不要写完整的句子，只需提供用空格分隔的 3-5 个核心词汇（例如：Jina embeddings dataset filtering steps）。
    3. 避免过于绝对的数字词汇（如直接搜“three steps”可能搜不到，改搜“data preparation filtering”更容易命中）。

    【过去的失败搜索词（请避开这些方向）】：
    {failed_history}

    请直接输出改写后的英文检索词，不要包含任何其他解释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5  # 稍微提高温度，增加搜索词的多样性
    )
    return response.choices[0].message.content.strip()


# ==================== 5. Agent 核心工作流 (Workflow) ====================
def extract_mermaid_and_text(response_text):
    """分离大模型回答中的文本和 Mermaid 代码"""
    # 匹配 ```mermaid ... ``` 格式的代码块
    pattern = r'```mermaid\n(.*?)\n```'
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        mermaid_code = match.group(1)
        # 将分离出的代码块从原文中删除，保持聊天窗口清爽
        clean_text = re.sub(pattern, '', response_text, flags=re.DOTALL).strip()
        return clean_text, mermaid_code
    return response_text, None


def render_mermaid_chart(mermaid_src: str, height: int = 480) -> None:
    """用 mermaid.js 在 iframe 内渲染图表（st.markdown 中的 ```mermaid 不会被执行）。"""
    code = (mermaid_src or "").strip()
    if not code:
        return
    # JSON 嵌入避免破坏 <script>，并保留中文等 Unicode
    code_json = json.dumps(code, ensure_ascii=False)
    html_page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/></head>
<body style="margin:0;padding:8px;background:#fafafa;">
  <div id="mermaid-out" class="mermaid"></div>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
  <script>
    const src = {code_json};
    const el = document.getElementById("mermaid-out");
    mermaid.initialize({{ startOnLoad: false, theme: "neutral", securityLevel: "loose",
      flowchart: {{ useMaxWidth: true, htmlLabels: true }} }});
    el.textContent = src;
    mermaid.run({{ nodes: [el], suppressErrors: true }}).then(() => {{
      try {{
        const svg = el.querySelector("svg");
        if (svg) {{
          svg.style.maxWidth = "100%";
          svg.style.height = "auto";
        }}
      }} catch (e) {{}}
    }}).catch(() => {{
      el.innerHTML = "";
      el.textContent = "Mermaid 渲染失败，请检查语法或网络。";
    }});
  </script>
</body></html>"""
    components.html(html_page, height=height, scrolling=True)


def run_agentic_workflow(question, api_key, model):
    if not api_key:
        return "⚠️ 请先配置 API Key。", ""

    MAX_RETRIES = 3  # 最大重试次数
    current_query = question
    failed_queries = []  # 记录失败的搜索词
    best_results = []
    highest_score = -1
    process_log = []

    def trace(msg):
        process_log.append(msg)
        return msg

    with st.status("🧠 Agent 正在思考与执行...", expanded=True) as status:

        for attempt in range(MAX_RETRIES + 1):
            is_first_try = (attempt == 0)
            step_name = "初步文献检索" if is_first_try else f"第 {attempt} 次深度检索"

            status.write(trace(f"🔍 [{step_name}] 正在搜索关键词: `{current_query}`"))
            top_results, context = retrieve_and_rerank(current_query)

            if not context:
                status.write(trace("⚠️ 未检索到任何片段。"))
                score = 0
                is_sufficient = False
                reason = "无内容"
            else:
                status.write(trace("⚖️ 正在评估资料相关性..."))
                grade = grade_context(question, context, api_key)
                score = grade.get('score', 0)
                is_sufficient = grade.get('is_sufficient', False)
                reason = grade.get('reason', '无')
                status.write(trace(f"📊 评价得分：{score}/10 ({reason})"))

            # 记录历史最高分的上下文（作为保底机制）
            if score > highest_score:
                highest_score = score
                best_results = top_results

            # 评判是否跳出循环
            if is_sufficient or score >= 7:
                status.write(trace("👍 资料相关性达标，终止检索循环！"))
                break
            else:
                if attempt < MAX_RETRIES:
                    status.write(trace(f"🔄 检测到质量不足 (得分 {score})，Agent 正在反思并重写搜索策略..."))
                    failed_queries.append(current_query)
                    current_query = rewrite_query(question, failed_queries, api_key)
                    status.write(trace(f"🆕 优化后的搜索词：`{current_query}`"))
                else:
                    status.write(trace("⚠️ 已达到最大重试次数，将使用目前最优结果进行强行生成。"))

        # Step 4: 最终生成 (使用多轮中得分最高的那次 context)
        status.write(trace("✍️ 正在聚合信息、绘制思维导图并撰写学术回答..."))
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        ref_sections = []
        for i, (_s, d, m) in enumerate(best_results, start=1):
            src = m.get("source", "未知来源")
            ref_sections.append(f"【[{i}]】来源：{src}\n{d}")
        numbered_context = "\n\n---\n\n".join(ref_sections) if ref_sections else "(无检索片段)"

        final_prompt = f"""你是一位严谨的科研助理。请严格根据以下编号资料回答问题。

        【要求】：
        1. 必须根据参考资料回答。若资料不足以回答，请明确说明「根据现有资料无法得出结论」，不要编造。
        2. 引用采用温哥华（Vancouver）编号制：正文中仅在确有依据的句末或关键论断处用上标式方括号编号，如 [1]、[2]；编号须与下方参考文献表一致，勿自行增删文献号。
        3. 正文之后、流程图之前，单独一节「## 参考文献」，按编号逐条列出，格式示例：
           [1] 完整来源标识（与资料中的「来源」字段一致，可补充片段中的题名/作者信息若资料中有）。
           条目之间空一行；勿在参考文献表中放入正文复述。
        4. 引用应克制，避免堆砌编号影响可读性；无依据处不要标注。
        5. 全文最后使用 Mermaid 语法生成流程图（包含在 ```mermaid 标签内），类型 graph TD，用箭头表示逻辑。

        【编号参考资料】：
        {numbered_context}

        【问题】：{question}"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3
            )
            status.update(label="✅ 任务完成！", state="complete", expanded=False)

            with st.expander("查看 Agent 最终采用的参考片段"):
                for i, (score_val, doc, meta) in enumerate(best_results):
                    st.write(f"**片段 {i + 1} | {meta['source']} (重排得分: {score_val:.2f})**")
                    st.code(doc)

            return response.choices[0].message.content, "\n\n".join(process_log)
        except Exception as e:
            status.update(label="❌ 生成失败", state="error")
            return f"生成回答时出错: {str(e)}", "\n\n".join(process_log)


# ==================== 6. 聊天界面与动态侧边栏渲染 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_mindmap" not in st.session_state:
    st.session_state.latest_mindmap = None  # 存储最新的思维导图

# --- 动态侧边栏：渲染最新思维导图 ---
with st.sidebar:
    st.divider()
    st.header("🧠 实时知识图谱")
    if st.session_state.latest_mindmap:
        render_mermaid_chart(st.session_state.latest_mindmap, height=520)
    else:
        st.caption("向 Agent 提问后，这里将自动生成逻辑拓扑图。")

# --- 主界面：渲染聊天记录 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        proc = message.get("agent_process")
        if message["role"] == "assistant" and proc:
            with st.expander("查看 / 隐藏 Agent 执行过程", expanded=False):
                st.markdown(proc)

if prompt := st.chat_input("向 Agent 提问 (例如：总结本文关于 XX 的核心观点)"):
    # 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent 生成回答
    with st.chat_message("assistant"):
        raw_response, agent_process = run_agentic_workflow(prompt, api_key, model_choice)

        # 分离纯文本和思维导图代码
        clean_response, mermaid_code = extract_mermaid_and_text(raw_response)

        # 在主界面只显示纯文本
        st.markdown(clean_response)

        if agent_process:
            with st.expander("查看 / 隐藏 Agent 执行过程", expanded=False):
                st.markdown(agent_process)

        # 将纯文本存入历史记录（附带过程日志，便于 rerun 后仍可展开查看）
        st.session_state.messages.append(
            {"role": "assistant", "content": clean_response, "agent_process": agent_process}
        )

        # 如果生成了思维导图，更新全局状态并利用 st.rerun() 刷新侧边栏
        if mermaid_code:
            st.session_state.latest_mindmap = mermaid_code
            st.rerun()  # 触发重新渲染，让侧边栏的图表立即弹出来