import chromadb
import torch
import fitz  # PyMuPDF
import uuid
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import logging
logger = logging.getLogger(__name__)

class MedicalRAGIngestion:
    def __init__(self, db_path="./medcpt_db"):
        # Article encoder for ingestion
        self.article_tokenizer = AutoTokenizer.from_pretrained(
            "ncbi/MedCPT-Article-Encoder"
        )
        self.article_model = AutoModel.from_pretrained(
            "ncbi/MedCPT-Article-Encoder"
        ).to("cuda")
        self.article_model.eval()

        # ChromaDB
        self.chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma.get_or_create_collection(
            name="medical_docs",
            metadata={"hnsw:space": "cosine"}
        )

    # ─── EMBED DOCUMENTS ──────────────────────────────────
    def embed_texts(self, texts: list[str]) -> list:
        inputs = self.article_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to("cuda")

        with torch.no_grad():
            outputs = self.article_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.cpu().numpy().tolist()

    # ─── INGEST MEDRAG HUGGINGFACE CORPUS ─────────────────
    def ingest_medrag_corpus(self, corpus_name="MedRAG/textbooks"):
        """
        Loads pre-chunked MedRAG corpus directly from HuggingFace.
        corpus_name options:
            MedRAG/textbooks
            MedRAG/statpearls  (download separately from NCBI first)
            MedRAG/pubmed
            MedRAG/wikipedia
        """
        logger.info(f"Loading {corpus_name} from HuggingFace...")
        dataset = load_dataset(corpus_name, split="train")

        batch_size = 32
        total = len(dataset)
        logger.info(f"Total chunks to ingest: {total}")

        for i in range(0, total, batch_size):
            batch = dataset[i:i+batch_size]

            # MedRAG textbooks have 'title' and 'content' fields
            texts = [
                f"{t} {c}"
                for t, c in zip(batch["title"], batch["content"])
            ]
            ids = [str(uuid.uuid4()) for _ in texts]
            metadatas = [
                {"source": corpus_name, "title": t}
                for t in batch["title"]
            ]

            embeddings = self.embed_texts(texts)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            if i % 500 == 0:
                logger.info(f"  Ingested {i}/{total} chunks...")

        logger.info(f"Done ingesting {corpus_name}")

    # ─── INGEST CUSTOM PDFs ───────────────────────────────
    def extract_pdf_chunks(
        self,
        pdf_path: str,
        chunk_size: int = 400,
        overlap: int = 50
    ) -> list[dict]:
        """
        Extracts text from PDF and splits into overlapping chunks.
        chunk_size: words per chunk
        overlap: words shared between consecutive chunks
        """
        doc = fitz.open(pdf_path)
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        # Split into words and create overlapping chunks
        words = full_text.split()
        chunks = []
        source_name = Path(pdf_path).stem

        for start in range(0, len(words), chunk_size - overlap):
            chunk_words = words[start:start + chunk_size]
            if len(chunk_words) < 50:  # Skip tiny trailing chunks
                break
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": " ".join(chunk_words),
                "source": source_name,
                "chunk_index": len(chunks)
            })

        logger.info(f"Extracted {len(chunks)} chunks from {source_name}")
        return chunks

    def ingest_pdf(self, pdf_path: str, batch_size: int = 16):
        chunks = self.extract_pdf_chunks(pdf_path)
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embed_texts(texts)

            self.collection.add(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {"source": c["source"], "chunk_index": c["chunk_index"]}
                    for c in batch
                ]
            )

        logger.info(f"Ingested {total} chunks from {pdf_path}")

    def ingest_pdf_folder(self, folder_path: str):
        """Ingest all PDFs in a folder"""
        pdfs = list(Path(folder_path).glob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {folder_path}")
        for pdf in pdfs:
            self.ingest_pdf(str(pdf))

    def stats(self):
        count = self.collection.count()
        logger.info(f"Total documents in vector store: {count}")