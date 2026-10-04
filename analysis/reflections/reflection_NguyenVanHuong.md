# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Văn Hưởng  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()`, `compare_strategies()` | Với threshold 0.85 trên corpus nối chung, semantic tạo 208 chunk, basic tạo 51 và structure-aware tạo 106. Semantic không mặc nhiên tạo ít chunk hơn: ngưỡng cao làm tách nhiều câu. Model được nạp một lần và tái sử dụng. |
| Hierarchical chunking | M1 + Pipeline | `chunk_hierarchical()`, `build_pipeline()`, `run_query()` | Khi xử lý từng tài liệu, 26 tài liệu có text tạo 26 parent và 124 child. Child tối đa 256 ký tự dùng cho retrieval; parent tối đa 2048 ký tự dùng làm context trả lời. ID parent có namespace theo nguồn để tránh trùng giữa tài liệu. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Tách từ bằng underthesea rồi thay `_` bằng khoảng trắng; index và query cùng chuyển về chữ thường. Dense dùng bge-m3, 1024 chiều. RRF cộng `1/(60 + rank + 1)`, không cộng trực tiếp điểm BM25 với cosine. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()`, `benchmark_reranker()` | bge-reranker-v2-m3 chấm cặp query–document. Benchmark sau warm-up với 3 ứng viên, 5 lượt: trung bình 115.02 ms, thấp nhất 111.72 ms, cao nhất 123.54 ms trên máy đang dùng. Đây là benchmark 3 ứng viên, không phải độ trễ toàn pipeline với 20 ứng viên. Pipeline chọn tối đa 3 parent khác nhau sau khi xếp hạng child. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()`, `save_report()` | Production đạt Faithfulness 0.8892, Relevancy 0.8614, Precision 0.9417, Recall 0.9167 trên 20 câu; cả bốn tăng so với Baseline. Relevancy là aggregate thấp nhất của Production. Report lưu câu hỏi, câu trả lời, đáp án chuẩn, context và điểm từng câu. Bottom-5 được chọn theo trung bình 4 metric; chẩn đoán tự động cần kiểm chứng bằng nội dung nguồn. |
| Contextual embeddings / HyQA | M5 | `_enrich_single_call()`, `contextual_prepend()`, `enrich_chunks()` | Một request gpt-4o-mini/chunk trả summary, questions, context và metadata. Production có 124 chunk cần enrichment. Context đứng trước văn bản; summary và HyQA cũng được đưa vào text đánh chỉ mục. Metadata nguồn và parent_id được giữ nguyên; API lỗi dùng fallback cục bộ. |

Các số lượng 51/208/106 ở phép so sánh M1 được tính trên corpus nối chung. Baseline thực tế cắt riêng từng tài liệu nên có 57 chunk; không dùng hai cách đếm này thay thế cho nhau. Hai PDF scan không có text layer được loader bỏ qua, nên kết quả áp dụng cho 26 tài liệu có text, không phải toàn bộ 28 file.

### Kết quả chạy từ đầu đến cuối

`main.py` chạy xong với exit code 0 trong 620.1 giây. Enrichment chiếm 367.1 giây; indexing Production 11.2 giây; nạp reranker 2.9 giây. Query Production trung bình 2631.5 ms, thấp nhất 2015.4 ms và cao nhất 3891.1 ms; số này bao gồm cả LLM. RAGAS Production chạy 80 phép chấm trong 80.9 giây.

| Metric | Naive Baseline | Production | Δ |
|--------|----------------|------------|---|
| Faithfulness | 0.8278 | 0.8892 | +0.0614 |
| Answer Relevancy | 0.7248 | 0.8614 | +0.1366 |
| Context Precision | 0.9250 | 0.9417 | +0.0167 |
| Context Recall | 0.9000 | 0.9167 | +0.0167 |

Relevancy cải thiện nhiều nhất. Các metric cao không chứng minh mọi câu đều đúng hoàn toàn: câu Senior vẫn thiếu bảng lương, bản mật khẩu cũ vẫn có lúc đứng trước bản mới, và câu tạm ứng thêm quy ước tính theo ngày chưa có trong nguồn. Chi tiết và bằng chứng được ghi trong [failure_analysis.md](../failure_analysis.md); điểm gốc nằm trong [ragas_report.json](../../reports/ragas_report.json). Vì Production khác Baseline ở nhiều bước và ở prompt trả lời, không quy toàn bộ mức tăng cho riêng enrichment hoặc reranking.

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Chạy pytest bằng sai Python

Lỗi đã gặp: `ModuleNotFoundError: No module named 'sentence_transformers'` và `ModuleNotFoundError: No module named 'pypdf'`.

Terminal có nhãn `(.venv)` nhưng log pytest cho thấy interpreter là Python 3.13 ngoài môi trường dự án. Debug bằng đường dẫn interpreter trong đầu log. Cách xử lý là chạy `.venv/bin/python -m pytest ...`, bảo đảm pytest và thư viện thuộc cùng môi trường, thay vì cài thêm dependency vào Python hệ thống.

### 2. RAGAS 0.1 và event loop trên Python 3.14

Lỗi đầu tiên: `There is no current event loop in thread 'MainThread'.` Thử áp dụng nest_asyncio trực tiếp dẫn đến lỗi tiếp theo: `Timeout should be used inside a task`.

Đọc code executor của RAGAS cho thấy nó tạo iterator `as_completed` trước khi `asyncio.run()` bắt đầu. Bản sửa dùng một generator wrapper để trì hoãn việc tạo tasks đến khi loop đang chạy, chỉ thay hàm trong phạm vi `evaluate()` rồi khôi phục. Đã kiểm tra executor hai lần liên tiếp và kiểm tra API thật trên một câu trước khi chạy cả bộ test. Không sửa các file trong site-packages hoặc thay toàn bộ global asyncio.

### 3. Cảnh báo từ dependency

Cảnh báo: `DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead`.

Hai cảnh báo xuất phát từ hai dòng trong LangChain, không phải hai lỗi độc lập của M4. Test vẫn pass. Giữ dependency theo requirements của bài lab; với dự án mới cần kiểm tra ma trận Python–RAGAS–LangChain trước khi nâng phiên bản, tránh nâng một thư viện riêng lẻ rồi làm vỡ các thư viện còn lại.

### 4. Docker daemon chưa chạy

Lỗi: `Cannot connect to the Docker daemon at unix:///Users/huongne/.docker/run/docker.sock. Is the docker daemon running?`

Mở Docker Desktop, chạy `docker compose up -d` và khởi động Qdrant trước lượt chạy chính. Phân biệt lỗi môi trường với lỗi thuật toán retrieval. Scaffold có Qdrant trong bộ nhớ để fallback, nhưng lượt thử nghiệm cuối sử dụng dịch vụ Docker của dự án.

### 5. Ghép module chưa đủ để có pipeline đúng

Khi kiểm tra phần điều phối, phát hiện parent được tạo nhưng chưa dùng khi gửi context cho LLM, và report chỉ có aggregate mà không có điểm từng câu. Bổ sung parent text vào metadata của child, chọn parent khác nhau sau reranking, và serialize `EvalResult` bằng `asdict()` vào report. Prompt Production yêu cầu ưu tiên văn bản hiện hành khi nguồn xác nhận bản mới thay thế bản cũ, giữ điều kiện phủ định và không tự thêm công thức tính.

Kiến thức cần bổ sung là đánh giá LLM-as-judge và kiểm soát chất lượng đáp án chuẩn. Ví dụ, `tam_ung.md` chỉ quy định 2%/tháng, trong khi đáp án chuẩn suy ra khoảng 50.000 đồng cho 5 ngày mà chưa có quy ước tính theo ngày trong nguồn. Cần đọc cả context và đáp án chuẩn trước khi kết luận điểm thấp luôn là lỗi mô hình; không sửa số điểm để làm đẹp báo cáo.

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý tra cứu chính sách nội bộ doanh nghiệp

Đây là đề xuất áp dụng từ bài lab; chưa có thông tin về một đồ án riêng khác. Lab được dùng làm nguyên mẫu cho dự án này.

#### 1. Hiện trạng

Pipeline nguyên mẫu sử dụng hierarchical chunking → enrichment → BM25 + Dense + RRF → Cross-Encoder → LLM → RAGAS. Corpus có tài liệu nhân sự, CNTT, tài chính và các bản chính sách cũ/mới. Các vấn đề cần tiếp tục kiểm chứng là chọn đúng phiên bản hiệu lực, tìm đủ nguồn cho câu hỏi nhiều bước, cách tính số liệu và thời gian/chi phí API khi enrichment nhiều chunk.

#### 2. Kế hoạch cải tiến

1. **Chunking:** Dùng Structure-Aware để nhận diện heading, bảng và quy định, kết hợp Hierarchical để retrieve child nhưng trả parent. Giữ tiêu đề và đường dẫn section trong metadata; bổ sung OCR cho PDF scan rồi kiểm tra độ chính xác trước khi index.
2. **Search:** Giữ Hybrid + RRF. Tách metadata phiên bản, ngày hiệu lực, trạng thái và tài liệu thay thế từ nguồn; query hỏi chính sách hiện hành cần lọc đúng phiên bản, query lịch sử vẫn được phép tìm bản cũ. Với câu hỏi nhiều bước, thử phân rã thành các câu con và hợp nhất ứng viên.
3. **Reranking:** Dùng bge-reranker-v2-m3 trên tập ứng viên nhỏ, chọn 3 parent khác nhau. Đo riêng latency retrieval, rerank và LLM; so sánh top-3 với top-5 trước khi quyết định cấu hình, thay vì tăng số context tùy ý.
4. **Evaluation:** Giữ 20 câu gốc làm bộ hồi quy; bổ sung câu để đạt ít nhất 50 ca, gồm phiên bản, phủ định, multi-hop và số liệu. Review đáp án chuẩn với chủ sở hữu chính sách. Mục tiêu nghiệm thu đề xuất: ít nhất 3 metric ≥0.75, faithfulness ≥0.85, đồng thời kiểm tra thủ công các câu quan trọng. Tách failure của evaluator/API khỏi failure của câu trả lời.
5. **Enrichment:** Giữ combined mode một request/chunk, cache theo nội dung và phiên bản tài liệu để chỉ làm giàu lại phần thay đổi. Chỉ dùng thông tin được nguồn hỗ trợ; metadata về nguồn và quyền truy cập không được LLM ghi đè. Theo dõi tỷ lệ fallback và chi phí thực tế.

#### 3. Timeline triển khai

- **Tuần 1:** Chuẩn hóa corpus và metadata phiên bản; review đáp án chuẩn; thử OCR trên hai PDF scan; triển khai chunking theo section và parent–child; chạy lại baseline trên bộ test cố định.
- **Tuần 2:** Bổ sung lọc phiên bản và retrieval nhiều bước; đo các cấu hình reranking; thêm cache enrichment; mở rộng bộ test; chạy RAGAS, kiểm tra Bottom-5 và đo latency trước khi trình diễn cho người dùng nội bộ.

Kết quả nghiệm thu sẽ căn cứ report và review nguồn thực tế. Nếu không đạt mục tiêu, ưu tiên sửa nguyên nhân đã xác định qua cây chẩn đoán rồi chạy lại cùng bộ test để so sánh.
