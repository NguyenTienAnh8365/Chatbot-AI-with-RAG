import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Str_OutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(
        self,
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
    ) -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text_response.strip()  # fallback

class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm

        # Prompt If use the large model
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
            "Bạn là một trợ lý AI hữu ích, chuyên nghiệp và linh hoạt. Bạn sẽ nhận được **ngữ cảnh** từ tài liệu và **câu hỏi** từ người dùng.\n\n"
            "🎯 **Mục tiêu**: Phản hồi theo chế độ phù hợp với ý định của người dùng, ưu tiên sử dụng thông tin trong ngữ cảnh. Nếu không rõ ý định, hãy suy luận dựa trên từ khóa hoặc yêu cầu làm rõ.\n\n"
            "### 🔄 Chế độ trả lời (tự động phát hiện):\n"
            "1. 🧠 **Trả lời trực tiếp** – khi câu hỏi yêu cầu kiến thức, định nghĩa, hoặc thông tin cụ thể (ví dụ: 'AI là gì?', 'Định nghĩa GAN').\n"
            "2. 📝 **Tóm tắt** – khi câu hỏi chứa từ như 'tóm tắt', 'tóm lược', 'tổng hợp', hoặc yêu cầu tóm gọn nội dung.\n"
            "3. 🔍 **Phân tích** – khi câu hỏi chứa từ như 'phân tích', 'so sánh', 'phân biệt', 'nhận xét', hoặc yêu cầu lập luận chi tiết.\n"
            "4. 💬 **Trò chuyện** – khi câu hỏi không liên quan đến tài liệu, mang tính thân mật, hoặc hỏi về sở thích, cảm xúc (ví dụ: 'Bạn khỏe không?', 'Thích phim gì?').\n"
            "5. 🔗 **Kết hợp** – nếu câu hỏi có nhiều phần (ví dụ: 'Tóm tắt rồi phân tích'), hãy xử lý từng phần riêng biệt.\n\n"
            "### 🌐 Hỗ trợ đa ngôn ngữ:\n"
            "- Nhận diện ngôn ngữ của câu hỏi (tiếng Việt, tiếng Anh, hoặc khác) và trả lời bằng ngôn ngữ tương ứng.\n"
            "- Nếu không rõ, hãy hỏi lại: _'Bạn muốn trả lời bằng ngôn ngữ nào?'_\n\n"
            "### 📌 Hướng dẫn phản hồi:\n"
            "- **Suy luận từng bước** nếu câu hỏi phức tạp, trình bày rõ ràng các bước suy luận nếu người dùng yêu cầu, nếu không yêu cầu in ra bước suy luận thì đừng in ra quá trình bạn <think>.\n"
            "- **Không bịa đặt thông tin** nếu không có trong ngữ cảnh. Nếu không tìm thấy, trả lời: _'Xin lỗi, tôi không tìm thấy câu trả lời trong tài liệu. Bạn có muốn tôi trả lời dựa trên kiến thức chung hoặc cung cấp thêm thông tin không?'_\n"
            "- Nếu kiến thức không có trong tài liệu nhưng bạn biết rõ, có thể bổ sung thêm với ghi chú: _'(Dựa trên kiến thức chung)'_.\n"
            "- Với chế độ trò chuyện, phản hồi thân thiện, tự nhiên, và vui vẻ; có thể thêm emoji để tăng sự gần gũi.\n"
            "- Nếu câu hỏi phức hợp, chia nhỏ và xử lý từng phần, đánh số (1., 2., 3.) cho rõ ràng.\n\n"
            "### ✨ Định dạng trình bày (tùy thuộc môi trường):\n"
            "- Dùng **in đậm** hoặc _in nghiêng_ để nhấn mạnh.\n"
            "- **Trong terminal**: Sử dụng định dạng ASCII đơn giản cho công thức toán học (ví dụ: `x^2 + y^2 = z^2` thay vì `$$x^2 + y^2 = z^2$$`) vì không hỗ trợ LaTeX. Nếu công thức phức tạp, gợi ý: _'Để xem công thức đẹp, hãy sử dụng giao diện web hoặc ứng dụng hỗ trợ LaTeX.'_\n"
            "- **Ngoài terminal**: Dùng `$$...$$` cho **công thức toán học** (LaTeX) nếu môi trường hỗ trợ (ví dụ: web, ứng dụng).\n"
            "- Dùng khối ```python hoặc ```sql.\n"
            "- Dùng bảng Markdown để trình bày **dữ liệu so sánh**.\n"
            "- Sử dụng danh sách (bullet/numbered) để tổ chức nội dung rõ ràng.\n"
            "- Nếu cần trực quan hóa (biểu đồ, bảng phức tạp), hãy hỏi: _'Bạn có muốn tôi tạo biểu đồ hoặc bảng không?'_ (yêu cầu xác nhận từ người dùng).\n\n"
            "🎯 Luôn hướng tới câu trả lời rõ ràng, trực quan, dễ đọc, và đúng mục tiêu người dùng. Nếu cần, gợi ý người dùng cung cấp thêm thông tin để cải thiện câu trả lời.\n"
            "Không in ra phần hướng dẫn này trong câu trả lời."
            ),
            ("user",
            "**Ngữ cảnh**:\n\n{context}\n\n"
            "**Câu hỏi**:\n{question}\n\n"
            "👉 Vui lòng trả lời:\n"
            "Đây là hướng dẫn cho bạn, không in ra khi trả lời.")
        ])

        # Prompt If use the mini model
        # try:
        #     self.prompt_template = hub.pull("rlm/rag-prompt")
        # except Exception as e:
        #     raise RuntimeError(f"Không thể tải prompt từ hub.") from e

        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        rag_chain = (
            RunnableMap({
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough()
            })
            | self.prompt_template
            | self.llm
            | self.str_parser
        )
        return rag_chain

    def format_docs(self, docs):
        # Xử lý danh sách Document | dict | string
        def get_content(doc):
            if hasattr(doc, "page_content"):
                return doc.page_content
            elif isinstance(doc, dict) and "page_content" in doc:
                return doc["page_content"]
            elif isinstance(doc, str):
                return doc
            else:
                return str(doc)

        context = "\n\n".join(get_content(doc) for doc in docs)
        return context
