# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Văn Hưởng

**Khóa:** K4 - Track 3B

**Ngày thực nghiệm:** 04/10/2026

**Nguồn số liệu:** [Production report](../reports/ragas_report.json) và [Naive report](../reports/naive_baseline_report.json).

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8278 | 0.8892 | +0.0614 |
| Answer Relevancy | 0.7248 | 0.8614 | +0.1366 |
| Context Precision | 0.9250 | 0.9417 | +0.0167 |
| Context Recall | 0.9000 | 0.9167 | +0.0167 |

Lượt chạy `main.py` kết thúc với exit code 0, đánh giá đủ 20 câu cho mỗi pipeline và không có lỗi evaluator. Baseline dùng 57 chunk cơ bản; Production dùng 124 child thuộc 26 parent, enrichment một request/chunk, Hybrid Search và Cross-Encoder. Sau reranking, Production gửi tối đa 3 parent khác nhau cho LLM và dùng prompt xử lý phiên bản/ngoại lệ ở temperature 0. Cả bốn metric tăng và đều vượt 0.75. Đây là so sánh hai pipeline đầy đủ, không phải thí nghiệm cô lập tác động riêng của một module.

Corpus thực nghiệm có 26 tài liệu trích được text. `BCTC.pdf` và PDF Nghị định bảo vệ dữ liệu cá nhân là scan không có text layer, được loader bỏ qua; chưa áp dụng OCR trong lượt chạy này.

| Thời gian quan sát | Kết quả |
|--------------------|---------|
| Toàn lượt Baseline + Production | 620.1 giây |
| Enrichment 124 chunk, xử lý tuần tự | 367.1 giây |
| Production indexing BM25 + Dense | 11.2 giây |
| Nạp Cross-Encoder | 2.9 giây |
| Query Production trung bình | 2631.5 ms |
| Query Production thấp nhất / cao nhất | 2015.4 / 3891.1 ms |
| RAGAS Production, 80 phép chấm | 80.9 giây |

Query latency bao gồm retrieval, reranking và gọi LLM, không phải thời gian riêng của Cross-Encoder. Chỉ số này được lưu trong `latency` của report.

## Bottom-5 Failures

Bottom-5 được chọn đúng thứ tự trong `failures` của Production report, dựa trên trung bình bốn metric. Điểm thấp nhất không đồng nghĩa tất cả năm câu đều trả lời sai. Chẩn đoán tự động của M4 là điểm bắt đầu; các nhận định dưới đây có kiểm tra câu trả lời và context. Report giữ nguyên điểm RAGAS thực tế.

### #1 — Phí tạm ứng quá hạn: đáp án khớp chuẩn nhưng có giả định chưa được nguồn xác nhận

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn thanh toán 15 ngày; quá hạn 5 ngày; phí 2%/tháng, tương đương 300.000 đồng/tháng; đáp án chuẩn suy ra khoảng 50.000 đồng cho 5 ngày.
- **Got:** Mô hình tính `15.000.000 × 2% = 300.000`, rồi `300.000 × 5/30 = 50.000 VNĐ`, kết luận nhân viên bị phạt 50.000 đồng. Đây là trích phần tính toán; câu trả lời đầy đủ nằm trong `per_question`.
- **Scores:** Faithfulness 0.1667; Relevancy 0.8326; Precision 1.0000; Recall 0.6667; trung bình **0.6665**.
- **Worst metric:** `faithfulness`. Chẩn đoán tự động: LLM tự bịa thông tin ngoài tài liệu.
- **Câu trả lời có đúng không?** Khớp số tiền trong ground truth, nhưng không đủ bằng chứng để khẳng định cách chia theo 30 ngày là quy định công ty. Vì vậy chưa thể coi là câu trả lời hoàn toàn có căn cứ.
- **Context có đáp án không?** Context đầu tiên là [tam_ung.md](../data/tam_ung.md), có hạn 15 ngày và phí 2%/tháng. Không có quy định thu theo ngày, cách làm tròn hay tháng 30 ngày. Hai nguồn còn lại là chi phí hoàn ứng và bảo hiểm sức khỏe, không bổ sung công thức này.
- **Query có cần viết lại không?** Câu hỏi rõ. Có thể hỏi thêm “Công ty có quy định tính phí theo ngày không?” để làm rõ điều kiện còn thiếu; sửa query không tự tạo được quy định đang thiếu trong corpus.
- **Error Tree:** Output khớp chuẩn → Context hỗ trợ tỷ lệ tháng nhưng thiếu công thức theo ngày → Query rõ → Lỗi ở quy tắc tính/đáp án chuẩn và bước suy luận của LLM.
- **Root cause:** Generation tự thêm giả định pro-rata 30 ngày dù prompt đã cấm thêm công thức. Ground truth cũng chứa cùng giả định chưa được nguồn xác nhận. RAGAS không phải công cụ kiểm chứng phép toán và có thể chấm thấp các dữ kiện lấy từ query; chưa có trace claim-level để phân bổ chính xác nguyên nhân của từng phần điểm.
- **Suggested fix / Module:** Bước LLM trong `src/pipeline.py` cần trả lời tỷ lệ 2%/tháng và nêu thiếu cách quy đổi theo ngày, hoặc ghi 50.000 đồng là kết quả *có điều kiện* nếu giả sử pro-rata 30 ngày. M4 và chủ sở hữu dữ liệu cần review ground truth; chỉ bổ sung công thức vào corpus sau khi được xác nhận. Temperature đã là 0, nên chỉ “giảm temperature” theo gợi ý tự động không giải quyết được vấn đề này.

### #2 — Nghỉ phép và lương Senior: thiếu một nhánh của câu hỏi nhiều bước

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo v2024, 18 ngày phép; Senior P3–P4 có lương gross 20–35 triệu đồng/tháng.
- **Got:** Mô hình tính đúng `15 + 3 = 18` ngày phép, nhưng nói context không cung cấp mức lương Senior nên không xác định được khoảng lương.
- **Scores:** Faithfulness 1.0000; Relevancy 0.8280; Precision 0.8333; Recall 0.5000; trung bình **0.7903**.
- **Worst metric:** `context_recall`. Chẩn đoán tự động: tìm kiếm bỏ sót đoạn văn đúng.
- **Câu trả lời có đúng không?** Đúng phần nghỉ phép, chưa trả lời phần lương. Việc nêu thiếu nguồn thay vì đoán lương giữ faithfulness cao.
- **Context có đáp án không?** Ba parent là `nghi_phep_nam_v2024.md`, `nghi_phep_khong_luong.md`, `nghi_phep_nam_v2023.md`. Không có [bang_luong_2024.md](../data/bang_luong_2024.md), trong khi file này có dòng `Senior (P3-P4) | 20.000.000 - 35.000.000`.
- **Query có cần viết lại không?** Không cần người dùng hỏi lại; hệ thống có thể tự phân rã thành “9 năm thâm niên được bao nhiêu ngày phép theo quy định hiện hành?” và “Khung lương Senior P3–P4 là bao nhiêu?”.
- **Error Tree:** Output đúng một phần → Context thiếu bảng lương → Query có hai nhánh rõ ràng → Kiểm tra retrieval và lựa chọn context cho từng nhánh.
- **Root cause:** Context cuối bị tập trung vào chủ đề nghỉ phép. Report chỉ lưu các context gửi LLM, không lưu toàn bộ top-20 trước reranking, nên chưa đủ bằng chứng kết luận bảng lương bị M2 bỏ sót hay bị M3 loại khỏi top cuối.
- **Suggested fix / Module:** M2 thử query decomposition rồi RRF trên kết quả các câu con; M3/Pipeline chọn context có độ bao phủ các nhánh câu hỏi, thay vì chỉ đa dạng parent. Ghi trace top-20 và top sau reranking để phân biệt lỗi recall với lỗi chọn context. Chưa cần sửa LLM để đoán thêm thông tin.

### #3 — Hoàn chi đào tạo: đáp án đúng nhưng faithfulness bị chấm thấp

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nghỉ trước cam kết 1 năm, phải hoàn trả 100%, tức 25.000.000 đồng.
- **Got:** Mô hình dẫn quy định cam kết 1 năm và hoàn trả 100% nếu nghỉ trước hạn, kết luận **25.000.000 VNĐ**.
- **Scores:** Faithfulness 0.4000; Relevancy 0.7918; Precision 1.0000; Recall 1.0000; trung bình **0.7980**.
- **Worst metric:** `faithfulness`. Chẩn đoán tự động: LLM tự bịa câu trả lời ngoài tài liệu.
- **Câu trả lời có đúng không?** Đúng khi kết hợp dữ kiện query (25 triệu, 8 tháng) với quy định trong nguồn (cam kết 1 năm, hoàn trả 100%). Không tìm thấy lỗi số tiền hay điều kiện áp dụng trong câu trả lời đã lưu.
- **Context có đáp án không?** Parent đầu tiên là [hoan_chi_dao_tao.md](../data/hoan_chi_dao_tao.md), có đầy đủ quy tắc hoàn chi. Hai context sau là đào tạo nội bộ và nghỉ ốm, không cần thiết cho phép tính.
- **Query có cần viết lại không?** Không; số tiền và mốc thời gian đã rõ.
- **Error Tree:** Output đúng theo nguồn + dữ kiện query → Context có quy tắc cần thiết → Query rõ → Cần kiểm tra evaluator và cách trình bày phép suy luận.
- **Root cause:** Điểm thấp không đủ để kết luận hallucination. Một khả năng là judge coi các dữ kiện 25 triệu/8 tháng trong câu trả lời là không được *context* hỗ trợ, dù chúng được cung cấp trong query. Đây là giả thuyết chẩn đoán, chưa được xác nhận bằng trace claim-level của RAGAS.
- **Suggested fix / Module:** M4 bổ sung review thủ công và kiểm thử phép tính có dùng dữ kiện query. Có thể kiểm tra riêng `8 < 12` và `25.000.000 × 100%`. Generation trình bày gọn quan hệ điều kiện–kết quả. Không thêm đáp án chuẩn vào context hoặc sửa điểm RAGAS để làm đẹp report.

### #4 — MFA: câu trả lời hiện hành đúng, ground truth yêu cầu thêm thông tin lịch sử

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, v2.0 bắt buộc MFA cho email, VPN và hệ thống nội bộ; ground truth còn ghi chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên bắt buộc MFA cho các tài khoản trên theo v2.0, hiệu lực 01/07/2024.
- **Scores:** Faithfulness 0.8000; Relevancy 0.9246; Precision 1.0000; Recall 0.5000; trung bình **0.8061**.
- **Worst metric:** `context_recall`. Chẩn đoán tự động: tìm kiếm bỏ sót đoạn văn đúng.
- **Câu trả lời có đúng không?** Đúng yêu cầu hiện hành và tập trung vào câu hỏi. Không nêu lịch sử v1.0, nhưng người dùng không hỏi so sánh các phiên bản.
- **Context có đáp án không?** Parent đầu tiên là [mat_khau_v2.md](../data/mat_khau_v2.md), có yêu cầu MFA và ngày hiệu lực. Hai parent còn lại là mua sắm và làm việc từ xa. Không có v1.0 trong context cuối, nên phần lịch sử của ground truth không được bao phủ.
- **Query có cần viết lại không?** Có thể thêm “theo chính sách hiện hành” để rõ phạm vi. Nếu muốn kiểm tra cả lịch sử, cần hỏi riêng sự thay đổi giữa v1.0 và v2.0.
- **Error Tree:** Output đúng hiện hành → Context đủ cho câu hỏi hiện hành nhưng thiếu lịch sử trong reference → Query không yêu cầu lịch sử → Review phạm vi ground truth trước khi sửa retrieval.
- **Root cause:** Reference rộng hơn câu hỏi. File v1.0 cũng không nói trực tiếp “không yêu cầu MFA”; việc v1.0 không đề cập MFA chưa đủ chứng minh một quy định phủ định. Cần chủ sở hữu chính sách xác nhận mệnh đề lịch sử này. Không đủ trace để giải thích riêng faithfulness 0.8.
- **Suggested fix / Module:** M4 review/tách câu hỏi hiện hành và câu hỏi so sánh phiên bản, giữ nguyên kết quả lượt chạy này. M2 bổ sung retrieval lịch sử khi query thật sự yêu cầu. Không cần đưa bản cũ vào mọi câu hỏi chỉ để tăng context recall của một reference quá rộng.

### #5 — Đổi mật khẩu: tài liệu cũ đứng trước tài liệu hiện hành

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** V2.0 hiện hành: 120 ngày; v1.0 cũ: 90 ngày và đã bị thay thế.
- **Got:** Theo chính sách hiện hành v2.0, mật khẩu phải thay đổi **mỗi 120 ngày**.
- **Scores:** Faithfulness 1.0000; Relevancy 0.8128; Precision 0.5000; Recall 1.0000; trung bình **0.8282**.
- **Worst metric:** `context_precision`. Chẩn đoán tự động: đoạn không liên quan bị xếp đầu.
- **Câu trả lời có đúng không?** Đúng. Prompt xử lý phiên bản giúp LLM sử dụng bản mới dù thứ tự retrieval chưa phù hợp.
- **Context có đáp án không?** Có: `mat_khau_v1.md` đứng thứ nhất, `mat_khau_v2.md` đứng thứ hai, sau đó là quy trình sự cố bảo mật. V1.0 ghi rõ `ĐÃ THAY THẾ bởi v2.0`, còn v2.0 ghi 120 ngày.
- **Query có cần viết lại không?** Có thể hỏi “Theo chính sách hiện hành, bao lâu phải đổi mật khẩu?”. Tuy nhiên trợ lý chính sách nên mặc định ưu tiên quy định hiện hành khi người dùng không hỏi lịch sử.
- **Error Tree:** Output đúng → Context có đáp án nhưng bản cũ xếp trước → Query chưa chỉ rõ thời điểm → Lỗi ưu tiên phiên bản ở retrieval/reranking.
- **Root cause:** Cross-Encoder đánh giá độ liên quan ngữ nghĩa, không bảo đảm ưu tiên trạng thái hiệu lực. Metadata hiện tại chưa có bộ lọc có cấu trúc cho `effective_date`, `status`, `superseded_by`; chỉ thêm reranker không tự giải quyết xung đột phiên bản.
- **Suggested fix / Module:** M1/M5 chuẩn hóa metadata hiệu lực từ nguồn; M2 lọc hoặc điều chỉnh ưu tiên phiên bản theo ý định query; M3/Pipeline giữ chế độ tìm lịch sử khi được hỏi. Kiểm tra vị trí tài liệu hiện hành trong top context và bổ sung ca hồi quy 90/120 ngày. Generation đã đúng nên không cần sửa đáp án của lượt chạy này.

## Case Study (cho presentation)

**Question chọn phân tích:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

**Error Tree walkthrough:**
1. **Output đúng?** Đúng 18 ngày phép, thiếu khoảng lương; faithfulness đạt 1.0 nhưng câu trả lời chưa đủ.
2. **Context đúng?** Đúng phần nghỉ phép v2024; thiếu bảng lương Senior 20–35 triệu, dù corpus có bảng này.
3. **Query rewrite OK?** Query gốc có hai yêu cầu rõ. Thử phân rã thành hai truy vấn để đo recall cho từng nhánh.
4. **Fix ở bước nào?** Ghi trace top-20 trước M3 để xác định nguyên nhân, sau đó thử M2 decomposition và lựa chọn context có độ bao phủ ở M3/Pipeline. Không yêu cầu LLM bịa mức lương khi chưa tìm thấy nguồn.

**Nếu có thêm 1 giờ, sẽ optimize:**

- 20 phút: thêm trace retrieval/reranking và thử phân rã câu Senior, so sánh nguồn bảng lương có xuất hiện trước/sau M3 hay không.
- 20 phút: chuẩn hóa trạng thái hiệu lực, thử lọc phiên bản cho câu đổi mật khẩu; giữ test lịch sử để tránh loại nhầm bản cũ khi được hỏi.
- 20 phút: review reference của câu MFA và cách tính tạm ứng; bổ sung kiểm thử suy luận số liệu. Chỉ sửa reference hoặc corpus sau khi xác nhận quy định, sau đó chạy lại cùng bộ test và lưu report mới để so sánh.
