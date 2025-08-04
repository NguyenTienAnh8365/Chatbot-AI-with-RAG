from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Giữ tiếng Việt
def clean_text(text):
    try:
        return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except:
        return text

def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=False).load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs.get("workers", 1))
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="🔄 Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class TextSplitter:
    def __init__(
        self,
        separators: List[str] = ['\n\n', '\n', ' ', ''],
        chunk_size: int = 5000,
        chunk_overlap: int = 500
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)

class Loader:
    def __init__(
        self,
        file_type: str = Literal["pdf"],
        split_kwargs: dict = {
            "chunk_size": 5000,
            "chunk_overlap": 500
        }
    ) -> None:
        assert file_type in ["pdf"], "file_type must be pdf"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")

        self.doc_spltter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_spltter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers=workers)

# === Example test ===
if __name__ == "__main__":
    doc_loaded = Loader(file_type="pdf").load_dir(
        "D:/PRJ-Github/RAG/data_source/generative_ai",
        workers=2
    )
    print(f"✅ Loaded {len(doc_loaded)} chunks")
    for i, chunk in enumerate(doc_loaded, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content)
