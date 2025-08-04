import os
from src.model.api_model import get_groq_llm
from src.model.local_model import get_hf_llm
from src.rag.main import build_rag_chain

# Tắt cảnh báo song song tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Câu hỏi RAG ===
def run_rag_query(rag_chain, question: str):
    response = rag_chain.invoke(question)
    
    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = response

    print(f"\n🧠 Question: {question}")
    print(f"💡 Answer: {answer}")
    return answer

# main
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Chỉ cần thiết trên Windows khi dùng multiprocessing

    # Model choice
    print("Chọn mô hình ứng dụng:")
    print("1. Model API")
    print("2. Model Local")
    print("Gõ 'exit' để thoát.")
    model_choice = input("Nhập 1 hoặc 2: ").strip().lower()

    if model_choice == "1":
        llm = get_groq_llm()
        print("Đã chọn model API.")
    elif model_choice == "2":
        llm = get_hf_llm()
        print("Đã chọn model Local")
    elif model_choice == "exit":
        print("👋 Exiting...")
        exit(0)
    else:
        print("Lựa chọn không hợp lệ. Vui lòng nhập lại.\n")

    data_path = "D:/PRJ-Github/RAG/data_source/generative_ai"
    rag_chain = build_rag_chain(llm, data_dir=data_path, data_type="pdf")

    # === CLI tương tác ===
    print("🟢 RAG System is ready! Type your question (or 'exit' to quit):")
    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            print("👋 Exiting...")
            break
        run_rag_query(rag_chain, question)
