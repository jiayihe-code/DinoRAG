import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ==================== 配置区 ====================
PAPER_FOLDER = "./my_papers"  # 存放 PDF 的文件夹
DB_PATH = "./research_db"  # 数据库路径

# 确保文件夹存在
os.makedirs(PAPER_FOLDER, exist_ok=True)

print("⏳ 正在初始化环境和模型...")
# 初始化模型和文本切分器
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# 初始化数据库
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(name="academic_papers")


def process_papers():
    # 获取已存在的文件名，实现增量更新
    existing_items = collection.get()
    existing_filenames = set([m['source'] for m in existing_items['metadatas']]) if existing_items[
        'metadatas'] else set()

    pdf_files = [f for f in os.listdir(PAPER_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️ 在 '{PAPER_FOLDER}' 文件夹中没有找到 PDF 文件。请先放入论文！")
        return

    print(f"📄 发现 {len(pdf_files)} 个 PDF 文件，开始处理...")

    for filename in pdf_files:
        if filename in existing_filenames:
            print(f"⏭️  跳过已存在: {filename}")
            continue

        filepath = os.path.join(PAPER_FOLDER, filename)
        print(f"🔄 正在解析: {filename}...")

        full_text_list = []
        try:
            # 使用 PyMuPDF 进行块级解析，完美解决双栏乱码
            doc = fitz.open(filepath)
            for page in doc:
                blocks = page.get_text("blocks")
                for b in blocks:
                    if b[6] == 0:  # 确保是纯文本块
                        block_text = b[4].strip()
                        # 将段落内换行替换为空格，恢复完整句子
                        clean_text = " ".join(block_text.splitlines())
                        # 过滤无意义的短字符（如单独的页码或公式残骸）
                        if len(clean_text) > 30:
                            full_text_list.append(clean_text)
            doc.close()

            text = "\n\n".join(full_text_list)

            if len(text.strip()) < 100:
                print(f"⚠️ 警告: {filename} 提取内容过少，已跳过。")
                continue

            # 切片并向量化入库
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                embedding = embed_model.encode(chunk).tolist()
                collection.add(
                    ids=[f"{filename}_{i}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": filename, "chunk_id": i}]
                )
            print(f"✅ 成功入库: {filename} (生成 {len(chunks)} 个片段)")

        except Exception as e:
            print(f"❌ 解析 {filename} 时出错: {str(e)}")

    print("\n🎉 所有论文处理完毕！向量数据库已更新。")


if __name__ == "__main__":
    process_papers()