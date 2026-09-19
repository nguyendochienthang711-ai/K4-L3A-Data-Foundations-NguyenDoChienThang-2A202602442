from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# Benchmark queries agreed by group G41 (K4-L3A)
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Sinh viên hệ chính quy được mượn tối đa bao nhiêu cuốn sách và trong thời gian bao lâu?",
        "filter": None,
        "gold_answer": "Sinh viên chính quy được mượn tối đa 05 cuốn tài liệu trong cùng một thời điểm; thời hạn mượn là 21 ngày (3 tuần) cho sách kho mở từ C2 trở đi.",
    },
    {
        "id": 2,
        "query": "Mức phạt trễ hạn mượn sách mỗi ngày là bao nhiêu tiền và nếu làm mất sách thì xử lý như thế nào?",
        "filter": None,
        "gold_answer": "Phí phạt trễ hạn là 5.000 VNĐ / cuốn / ngày trễ. Nếu làm mất sách, bồi thường sách mới tương đương + 20.000 VNĐ phí xử lý, hoặc bồi thường 100% giá bìa + 100.000 VNĐ phí xử lý kỹ thuật.",
    },
    {
        "id": 3,
        "query": "Điều kiện để tài khoản của sinh viên được kích hoạt quyền mượn tài liệu thư viện về nhà là gì?",
        "filter": {"audience": "student"},
        "gold_answer": "Sinh viên phải xuất trình thẻ sinh viên hợp lệ và hoàn thành khóa học tập huấn thư viện đầu khóa với điểm bài kiểm tra trắc nghiệm từ 80% trở lên.",
    },
    {
        "id": 4,
        "query": "Đặc quyền về số lượng sách và quyền truy cập cơ sở dữ liệu quốc tế của giảng viên và nghiên cứu sinh là gì?",
        "filter": None,
        "gold_answer": "Giảng viên được mượn 05 cuốn tiêu chuẩn (tối đa 10 cuốn khi có đề tài nghiên cứu); được cấp quyền truy cập từ xa vào các CSDL quốc tế (IEEE Xplore, ScienceDirect, Scopus) và sử dụng phòng nghiên cứu riêng.",
    },
    {
        "id": 5,
        "query": "Quy định về việc sao chụp (photocopy) tài liệu trong thư viện cho phép tối đa bao nhiêu phần trăm cuốn sách?",
        "filter": None,
        "gold_answer": "Bạn đọc chỉ được phép sao chụp phục vụ cá nhân không quá 20% tổng số trang của một cuốn sách hoặc không quá 01 chương sách; nghiêm cấm sao chép nguyên cuốn giáo trình.",
    },
]


def parse_markdown_with_frontmatter(file_path: Path) -> tuple[dict, str]:
    content = file_path.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    meta = {}
    body = content
    if len(parts) >= 3:
        frontmatter = parts[1]
        body = parts[2].strip()
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, body


def load_university_corpus(dir_path: str = "data/university", chunk_size: int = 400) -> list[Document]:
    path = Path(dir_path)
    chunker = RecursiveChunker(chunk_size=chunk_size)
    documents: list[Document] = []

    files = sorted(path.glob("*.md"))
    for file_p in files:
        meta, body = parse_markdown_with_frontmatter(file_p)
        chunks = chunker.chunk(body)
        doc_id = meta.get("doc_id", file_p.stem)
        for idx, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{doc_id}_chunk_{idx}",
                    content=chunk_text,
                    metadata={
                        **meta,
                        "chunk_idx": idx,
                        "source_file": file_p.name,
                    },
                )
            )
    return documents


def get_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def run_benchmark(output_path: str = "benchmark.txt"):
    # Force UTF-8 stdout encoding where possible
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    log_lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("      K4-L3A BENCHMARK RUNNER — NHÓM G41 (THƯ VIỆN ĐH BÁCH KHOA)")
    log("=" * 70)

    docs = load_university_corpus()
    log(f"[*] Đã nạp và chia nhỏ {len(docs)} chunks từ thư mục data/university")

    embedder = get_embedder()
    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    log(f"[*] Embedding backend: {backend_name}")

    store = EmbeddingStore(collection_name="university_benchmark", embedding_fn=embedder)
    store.add_documents(docs)
    log(f"[*] Đã index {store.get_collection_size()} chunks vào EmbeddingStore")

    agent = KnowledgeBaseAgent(store=store, llm_fn=lambda p: "[Agent] Câu trả lời sinh từ ngữ cảnh truy xuất.")

    log("\n" + "-" * 70)
    log("              KẾT QUẢ TRUY XUẤT 5 CÂU HỎI ĐÁNH GIÁ")
    log("-" * 70)

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        q_text = item["query"]
        q_filter = item["filter"]
        gold = item["gold_answer"]

        if q_filter:
            results = store.search_with_filter(q_text, top_k=3, metadata_filter=q_filter)
        else:
            results = store.search(q_text, top_k=3)

        top1 = results[0] if results else None
        score = top1["score"] if top1 else 0.0
        doc_source = top1["metadata"].get("source_file", "N/A") if top1 else "N/A"
        chunk_preview = top1["content"][:150].replace("\n", " ") if top1 else ""

        log(f"\n[Câu {q_id}]: {q_text}")
        if q_filter:
            log(f"   Bộ lọc (Filter): {q_filter}")
        log(f"   Điểm Top-1 Score: {score:.4f} | Nguồn: {doc_source}")
        log(f"   Trích đoạn Top-1: {chunk_preview}...")
        log(f"   Đáp án chuẩn (Gold): {gold}")

        # In / ghi chi tiết top 3
        log("   Top-3 chunks được xếp hạng:")
        for r_idx, res in enumerate(results, start=1):
            r_src = res["metadata"].get("source_file", "N/A")
            r_prev = res["content"][:80].replace("\n", " ")
            log(f"     {r_idx}. [score={res['score']:.4f}] {r_src} -> \"{r_prev}...\"")

    log("\n" + "=" * 70)
    log("               HOÀN TẤT ĐÁNH GIÁ BENCHMARK")
    log("=" * 70)

    # Ghi toàn bộ kết quả vào file benchmark.txt
    out_file = Path(output_path)
    out_file.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[+] Đã lưu kết quả đánh giá benchmark thành công vào file: {out_file.resolve()}")


if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark.txt"
    run_benchmark(output_path=output_file)

