# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đỗ Chiến Thắng
**Nhóm:** G41
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiệm cận 1.0) nghĩa là hai vector chỉ về cùng một hướng trong không gian embedding đa chiều, biểu thị hai đoạn văn bản có sự tương đồng rất lớn về mặt ngữ nghĩa và bối cảnh thông tin, dù cách dùng từ hoặc độ dài câu có thể khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Sinh viên phải hoàn tất đăng ký học phần trước ngày 15 tháng 9.
- Câu B: Hạn chót để sinh viên lựa chọn môn học trên hệ thống là ngày 15/09.
- Tại sao tương đồng: Cả hai câu cùng truyền tải chung một thông tin quy định về hạn cuối đăng ký môn học cho sinh viên, chỉ khác nhau về cách diễn đạt từ đồng nghĩa ("học phần" / "môn học", "trước ngày" / "hạn chót").

**Ví dụ có độ tương tự THẤP:**
- Câu A: Sinh viên phải hoàn tất đăng ký học phần trước ngày 15 tháng 9.
- Câu B: Thư viện trường mở cửa phục vụ bạn đọc từ 7h30 đến 21h00 các ngày trong tuần.
- Tại sao khác: Hai câu đề cập đến hai mảng nghiệp vụ đại học hoàn toàn tách biệt (đăng ký học phần tín chỉ và giờ giấc phục vụ của thư viện), các vector ngữ nghĩa chỉ về hai hướng độc lập nhau trong không gian đa chiều.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid phụ thuộc trực tiếp vào độ dài (độ lớn) của vector, khiến cho một câu ngắn và một đoạn văn dài dù cùng một chủ đề vẫn bị tính là xa nhau. Trong khi đó, độ tương tự cosine chuẩn hóa độ lớn và chỉ đo góc định hướng giữa hai vector, giúp nắm bắt chính xác sự tương đồng ngữ nghĩa mà không bị ảnh hưởng bởi độ dài văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Áp dụng công thức số lượng chunk: $\lceil (\text{độ dài} - \text{overlap}) / (\text{chunk\_size} - \text{overlap}) \rceil = \lceil (10000 - 50) / (500 - 50) \rceil = \lceil 9950 / 450 \rceil = \lceil 22.11 \rceil = 23$.
> *Đáp án:* 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap = 100, bước nhảy là $500 - 100 = 400$, số lượng chunk tăng lên thành $\lceil (10000 - 100) / 400 \rceil = \lceil 9900 / 400 \rceil = \lceil 24.75 \rceil = 25$ chunks (tăng 2 chunks). Ta muốn tăng độ chồng chéo để bảo toàn trọn vẹn ngữ cảnh tại các ranh giới cắt, tránh việc các câu văn hoặc thông tin quan trọng bị ngắt đôi giữa hai chunk làm hệ thống truy xuất bỏ sót.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy lookbehind `r'(?<=[.!?])\s+'` để phát hiện điểm kết thúc câu sau các dấu chấm, chấm than hoặc chấm hỏi mà vẫn giữ nguyên vẹn dấu câu. Về các trường hợp ngoại lệ (edge cases), hàm kiểm tra và xử lý văn bản rỗng hoặc chuỗi toàn khoảng trắng bằng cách trả về danh sách rỗng `[]`, loại bỏ các câu rỗng sau khi tách, và gom nhóm các câu thành các chunk theo bước nhảy `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động bằng cách duyệt qua danh sách các dấu phân cách theo thứ tự ưu tiên giảm dần `["\n\n", "\n", ". ", " ", ""]`. Nếu một đoạn sau khi tách vẫn có độ dài vượt quá `chunk_size`, thuật toán sẽ gọi đệ quy `_split` để tiếp tục chia nhỏ đoạn đó bằng các dấu phân cách cấp tiếp theo; các đoạn nhỏ hơn được gom dần lại với nhau sao cho không vượt quá `chunk_size`. Base cases gồm: (1) chuỗi rỗng trả về `[]`, (2) độ dài chuỗi $\le$ `chunk_size` giữ nguyên không chia tiếp, và (3) hết danh sách dấu phân cách thì chia cơ học theo ký tự.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents` chuẩn hóa mỗi đối tượng `Document` thành bản ghi lưu trong danh sách `self._store` gồm `id`, `content`, `metadata` (luôn có trường `doc_id`) và vector nhúng được tính trước qua `_embedding_fn`. Khi thực hiện `search`, câu truy vấn của người dùng được chuyển thành vector, tính tích vô hướng (dot product) với từng vector lưu trữ thông qua hàm `_dot()`, sau đó sắp xếp theo điểm tương đồng (`score`) giảm dần và lấy ra `top_k` kết quả cao nhất.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` sử dụng cơ chế tiền lọc (pre-filtering), lọc ra danh sách các bản ghi trong kho lưu trữ thỏa mãn đồng thời tất cả các điều kiện key-value trong `metadata_filter` trước, sau đó mới tiến hành tính toán tìm kiếm tương đồng trên tập dữ liệu đã lọc nhằm đảm bảo độ chính xác cao nhất. Với `delete_document`, hệ thống lọc bỏ các bản ghi có `id` hoặc `metadata["doc_id"]` khớp với `doc_id` cần xóa, và so sánh số lượng bản ghi trước/sau để trả về `True` nếu có bản ghi bị xóa, ngược lại trả về `False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Cấu trúc prompt được thiết kế theo mô hình RAG tiêu chuẩn gồm 3 khối: Chỉ dẫn vai trò trợ lý thông minh (System instruction), Khối ngữ cảnh thực tế được truy xuất (Context blocks nối với nhau bằng hai dấu xuống dòng `\n\n`), và Câu hỏi của người dùng (Question). Ngữ cảnh được đưa vào (inject context) bằng cách gọi `self.store.search` (hoặc `search_with_filter`), trích xuất trường `content` của từng chunk tài liệu tìm được, ghép thành chuỗi ngữ cảnh rồi truyền vào `llm_fn` để sinh câu trả lời bám sát sự thật.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.10.21, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Public\Documents\AI_BAITAP\K4-L3A-Data-Foundations-NguyenDoChienThang-2A202602442
plugins: anyio-4.15.1, langsmith-0.12.6, asyncio-1.4.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.10s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên năm cuối cần hoàn thành khóa luận tốt nghiệp đúng hạn. | Thời hạn nộp đồ án tốt nghiệp cho sinh viên khóa cuối là trước tháng 6. | cao | 0.8652 | Đúng |
| 2 | Thủ tục xin cấp lại thẻ sinh viên bị mất tại phòng công tác sinh viên. | Ký túc xá nghiêm cấm nấu ăn bằng bếp điện trong phòng ở. | thấp | 0.6137 | Đúng |
| 3 | Sinh viên được phép rút học phần trước tuần thứ tư của học kỳ. | Sinh viên không được phép rút học phần sau tuần thứ tư của học kỳ. | cao | 0.9822 | Đúng |
| 4 | Quy định về thời hạn đóng học phí của trường đại học. | Regulations on university tuition payment deadlines. | cao | 0.8332 | Đúng |
| 5 | Quy trình đăng ký mượn tài liệu tại thư viện trung tâm. | Món phở bò truyền thống có nước dùng ngọt thanh từ xương hầm. | thấp | 0.5641 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là ở Cặp 3: mặc dù hai câu mang ý nghĩa phủ định/trái ngược nhau ("được phép" so với "không được phép"), điểm tương đồng cosine thực tế đo bởi mô hình Gemini lại cao kỷ lục (0.9822). Điều này chỉ ra rằng embedding học sâu biểu diễn rất xuất sắc chủ đề và ngữ cảnh từ vựng xung quanh, nhưng thường gặp khó khăn với các từ phủ định ngắn ("không", "chưa"). Ngoài ra, Cặp 4 (Tiếng Việt và Tiếng Anh) đạt điểm rất cao (0.8332), minh chứng khả năng biểu diễn ngữ nghĩa đa ngôn ngữ (cross-lingual) vượt trội của mô hình nhúng hiện đại.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên hệ chính quy được mượn tối đa bao nhiêu cuốn sách và trong thời gian bao lâu? | `hcmut-lib-borrowing-student` (Mục 2: Sinh viên chính quy mượn tối đa 5 cuốn, thời hạn 21 ngày, gia hạn 1 lần 21 ngày) | 0.8583 | Có | Sinh viên chính quy được mượn tối đa 05 cuốn tài liệu trong 21 ngày và được gia hạn thêm 01 lần 21 ngày. |
| 2 | Mức phạt trễ hạn mượn sách mỗi ngày là bao nhiêu tiền và nếu làm mất sách thì xử lý như thế nào? | `hcmut-lib-policy-adjustment` (Mục 1 & 2: Phạt 5.000đ/cuốn/ngày; mất sách đền bản mới + 20.000đ phí hoặc 100% giá bìa + 100.000đ phí xử lý) | 0.8701 | Có | Phí phạt trễ hạn là 5.000 VNĐ/cuốn/ngày. Làm mất sách cần mua bản mới đền hoặc bồi thường 100% giá bìa cộng 100.000 VNĐ phí xử lý kỹ thuật. |
| 3 | Điều kiện để tài khoản của sinh viên được kích hoạt quyền mượn tài liệu thư viện về nhà là gì? | `hcmut-lib-borrowing-student` & `hcmut-lib-information-training` (Thẻ SV hợp lệ + hoàn thành tập huấn đạt từ 80% điểm trắc nghiệm) | 0.8089 | Có | Sinh viên cần có thẻ sinh viên hợp lệ và phải hoàn thành khóa học tập huấn thư viện đầu khóa với điểm kiểm tra trắc nghiệm từ 80% trở lên. |
| 4 | Đặc quyền về số lượng sách và quyền truy cập cơ sở dữ liệu quốc tế của giảng viên và nghiên cứu sinh là gì? | `hcmut-lib-borrowing-faculty` (Mục 2: Mượn 5-10 cuốn; truy cập từ xa IEEE Xplore, ScienceDirect, Scopus, SpringerLink) | 0.7925 | Có | Giảng viên được mượn tối đa 5–10 cuốn khi có đề tài nghiên cứu và được cấp quyền truy cập từ xa vào các CSDL quốc tế (IEEE, ScienceDirect, Scopus). |
| 5 | Quy định về việc sao chụp (photocopy) tài liệu trong thư viện cho phép tối đa bao nhiêu phần trăm cuốn sách? | `hcmut-lib-document-delivery` (Mục 1: Tối đa 20% tổng số trang của sách hoặc 1 chương sách; không sao chép nguyên cuốn) | 0.8501 | Có | Bạn đọc chỉ được phép sao chụp phục vụ cá nhân tối đa không quá 20% tổng số trang hoặc 1 chương sách, không được sao chép nguyên cuốn giáo trình. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Khi nhóm cùng chạy thử nghiệm và so sánh kết quả giữa các thành viên, tôi nhận ra rằng chiến lược `RecursiveChunker` kết hợp cùng cơ chế tiền lọc `metadata_filter={"audience": "student"}` cho độ chính xác cao vượt trội so với việc chia nhỏ thô theo kích thước cố định (`FixedSizeChunker`). Lọc metadata giúp loại bỏ hoàn toàn các kết quả nhiễu từ tài liệu của giảng viên hay nội quy chung, đồng thời chunking theo ranh giới cấu trúc văn bản giúp đoạn trích dẫn trả về giữ trọn vẹn các con số và điều khoản quy định.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
