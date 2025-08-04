import os
from typing import Union, List, Optional
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorDB:
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        persist_path: str = "D:/PRJ-Github/RAG/chroma_db",
        vector_db: Union[Chroma, FAISS] = Chroma,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        append: bool = True,  # add data or rebuild vector db
    ) -> None:
        self.vector_db_cls = vector_db
        self.embedding = embedding
        self.persist_path = persist_path

        os.makedirs(self.persist_path, exist_ok=True)

        if documents:
            if append and os.path.exists(self.persist_path):
                print(f"[INFO] Thêm {len(documents)} documents vào VectorDB đã có.")
                self.db = self._load_persisted_db()
                self.db.add_documents(documents)
                if hasattr(self.db, "persist"):
                    self.db.persist()
            else:
                print(f"[INFO] Build lại VectorDB với {len(documents)} documents.")
                self.db = self._build_and_persist_db(documents)
        else:
            self.db = self._load_persisted_db()

    def _build_and_persist_db(self, documents):
        db = self.vector_db_cls.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_path
        )
        print(f"[INFO] VectorDB được tạo và lưu tại: {self.persist_path}")
        return db

    def _load_persisted_db(self):
        if not os.path.exists(self.persist_path):
            raise ValueError(f"[ERROR] Không tìm thấy thư mục vector store: {self.persist_path}")
        
        if self.vector_db_cls == Chroma:
            db = Chroma(
                persist_directory=self.persist_path,
                embedding_function=self.embedding
            )
        else:  # FAISS
            db = self.vector_db_cls.load_local(
                self.persist_path, self.embedding, allow_dangerous_deserialization=True
            )
        print(f"[INFO] VectorDB được tải từ thư mục: {self.persist_path}")
        return db

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 10}
    ):
        retriever = self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        return retriever