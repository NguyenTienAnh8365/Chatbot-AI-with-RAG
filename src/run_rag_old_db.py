from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG
from src.model.api_model import get_groq_llm
from src.model.local_model import get_hf_llm
from langchain_community.llms import HuggingFaceHub
import os

def main():
    # DB đã build
    persist_dir = "D:/PRJ-Github/RAG/chroma_db"

    # 1. Load lại VectorDB đã được build từ trước
    print("[INFO] Đang load VectorDB từ thư mục đã lưu...")
    vector_db = VectorDB(
        documents=None,  
        persist_path=persist_dir,
        vector_db=Chroma,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # 2. Lấy retriever
    retriever = vector_db.get_retriever(search_kwargs={"k": 5})

    # 3. Chọn mô hình LLM
    while True:
        print("Chọn mô hình ứng dụng:")
        print("1. Model API")
        print("2. Model Local")
        print("Gõ 'exit' để thoát.")
        model_choice = input("Nhập 1 hoặc 2: ").strip().lower()
        if model_choice == "1":
            llm = get_groq_llm()
            print("Đã chọn model API.")
            break
        elif model_choice == "2":
            llm = get_hf_llm()
            print("Đã chọn model Local.")
            break
        elif model_choice == "exit":
            print("👋 Kết thúc truy vấn.")
            return
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập lại.\n")

    # 4. Khởi tạo RAG chain
    rag = Offline_RAG(llm=llm)
    chain = rag.get_chain(retriever)

    # 5. Truy vấn người dùng
    while True:
        question = input("\n❓ Nhập câu hỏi (hoặc 'exit' để thoát): ").strip()
        if question.lower() == "exit":
            print("👋 Kết thúc truy vấn.")
            break

        result = chain.invoke(question)
        print(f"\n📌 Câu trả lời:\n{result}\n")

if __name__ == "__main__":
    main()
