# Lab 21 — Evaluation Report

**Họ tên**: Vũ Việt Anh  **MSSV**: 2A202601107  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA Tesla T4 (15.0 GB VRAM)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 200 / 50 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*
Template Jinja của Qwen3.5 giữ nguyên cặp tag `<think>...</think>`, đảm bảo chuỗi suy luận (reasoning trace) không bị cắt bỏ hoặc làm hỏng định dạng khi nạp vào model.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | `0.4149` (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss (từ `mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3494.1 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 1035.6 |
| (c) LoRA fine-tune | 0.9375 | 0.7500 | 1.0000 | 1564.2 |

**(b) có thật sự mạnh hơn (a) không?** Có. Prompt tối ưu (b) đạt độ chính xác target 0.6875 và format 1.0000 (so với (a) chỉ đạt 0.0000 ở cả hai tiêu chí), đồng thời giảm độ trễ từ ~3.49s xuống ~1.04s.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không, giữ nguyên prompt chuẩn được cung cấp từ labkit để đảm bảo tính khách quan và liêm chính trong phép đo so sánh (SHA khớp mã định danh).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 (1e-4) | 0.6262 | 0.9375 | 1004.7 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 (1e-4) | 0.5376 | 0.9375 | 839.6 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 (1e-5) | 1.5704 | 0.0000 | 1019.4 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 (1e-4) | 0.7058 | 0.8438 | 1041.5 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế chính là Lỗi #3.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` (r=283) và `correct` (r=16) đều đạt điểm số tương đương (0.9375), tuy nhiên thứ tự theo train loss lại trái ngược: `attn_only` có loss thấp hơn (0.5376 so với 0.6262 của `correct`). Hiện tượng này cho thấy train loss thấp hơn không đồng nghĩa với khả năng tổng quát hóa thực tế tốt hơn khi đánh giá downstream task. Quan trọng hơn, việc phải tăng rank lên tới 283 ở attention module để bù đắp cho việc bỏ qua các lớp MLP/linear chứng minh rằng *vị trí gắn adapter* (all text-linear) là nhân tố cốt lõi giúp thích ứng biểu diễn ngôn ngữ hiệu quả hơn việc chỉ tập trung ép *rank* cực lớn vào riêng khối attention.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Ở run `wrong_lr`, việc áp dụng learning rate quy mô Full Fine-tuning (1e-5 thay vì 1e-4 khuyến nghị cho LoRA) khiến đường loss gần như đi ngang và dừng ở mức rất cao (1.5704 so với 0.6262), dẫn đến điểm target rớt về 0.0000 và format 0.0000 (mô hình không học được cách sinh JSON chuẩn). Nếu chỉ nhìn vào đường loss phẳng lì mà không kiểm tra learning rate, một kỹ sư có thể kết luận sai lầm rằng dữ liệu huấn luyện quá khó, số lượng epoch chưa đủ, hoặc kiến trúc LoRA không thể hội tụ trên bài toán này. Thực chất, adapter LoRA chỉ cập nhật một ma trận tham số phụ với gradient scale nhỏ, do đó cần một learning rate lớn hơn 5x đến 10x so với Full Fine-tuning để có thể dịch chuyển trọng số hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` giúp tiết kiệm đáng kể ~4.92 GB VRAM (chỉ chiếm 7.09 GB so với 12.01 GB của 16-bit LoRA), cho phép chạy vừa vặn trên các GPU cấu hình thấp. Tuy nhiên, cái giá phải trả là điểm target tụt từ 0.9375 xuống 0.8438 và thời gian huấn luyện lâu hơn một chút (1041.5s so với 1004.7s do chi phí dequantize on-the-fly). Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị của nhà phát hành: với họ model Qwen3.5 kích thước nhỏ/vừa trên các tác vụ phân loại JSON nghiêm ngặt, việc lượng tử hóa 4-bit gây mất mát thông tin nhạy cảm ở các trọng số, làm giảm độ chính xác; nếu VRAM cho phép (như T4 16GB), lựa chọn tối ưu luôn là 16-bit LoRA (fp16/bf16).

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.2500` · `regression Δ = +0.0000` · `valid_trace_rate = 0.00`

**Diễn giải:**
Bản fine-tune LoRA (`correct`) đã vượt qua cổng hồi quy một cách thuyết phục với mức cải thiện độ chính xác phân loại `target Δ = +0.2500` (tăng từ 0.6875 ở baseline prompt tối ưu lên 0.9375 ở mô hình fine-tune). Đồng thời, `regression Δ = +0.0000` (điểm các câu hỏi kiến thức phổ quát giữ nguyên ở mức 0.7500), chứng minh mô hình fine-tune không gặp phải hiện tượng "quên thảm họa" (catastrophic forgetting) đối với năng lực nền tảng. Tỷ lệ format đạt tuyệt đối 1.0 (100% tuân thủ cấu trúc JSON 4 trường hợp lệ). Do đó, quyết định phê duyệt triển khai mô hình là hoàn toàn hợp lý và có cơ sở thực nghiệm vững chắc.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, chuột không dây VN232232. Cho tôi trả lại. Gấp... | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Đúng format, phân loại đúng | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | ✅ FT thắng / chuẩn xác tuyệt đối |
| 2 | Ốp lưng điện thoại VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng...`, `tieu_cuc` | Đúng format, phân loại đúng | `hoan_tien`, `trung_binh`, `ốp lưng...`, `tieu_cuc` | ✅ FT thắng / nhận diện đúng sentiment tiêu cực |
| 3 | Cho mình hỏi, bình giữ nhiệt VN804124. Chưa thấy tiền. Khi nào tiện... | `hoan_tien`, **`thap`**, `bình giữ nhiệt`, `tich_cuc` | Nhận diện đúng `thap` | `hoan_tien`, **`trung_binh`**, `bình giữ nhiệt`, `tich_cuc` | ❌ **FT thua** (đoán sai trường `urgency`) |
| 4 | Shop ơi, nồi chiên không dầu DH249548. Thiếu phụ kiện. Khi nào tiện... | `san_pham_loi`, **`thap`**, `nồi chiên...`, `trung_tinh` | Nhận diện đúng `thap` | `san_pham_loi`, **`trung_binh`**, `nồi chiên...`, `trung_tinh` | ❌ **FT thua** (đoán sai trường `urgency`) |
| 5 | Đèn bàn LED VN339109. Vỡ khi nhận. Gấp. Shop xem giúp. | `san_pham_loi`, `cao`, `đèn bàn LED`, `trung_tinh` | Đúng format | `san_pham_loi`, `cao`, `đèn bàn LED`, `trung_tinh` | ✅ FT thắng / trích xuất đúng sản phẩm & độ khẩn |

**Có mẫu chung nào ở các ca FT thua không?**
Có. Ở các ca FT bị mất điểm (mẫu 3 và 5 trong eval target), mô hình Fine-tune có xu hướng thiên lệch (bias) dự đoán mức độ khẩn cấp là `trung_binh` thay vì `thap`, ngay cả khi câu thoại chứa cụm từ ngữ cảnh thể hiện sự không gấp như *"Khi nào tiện"*. Điều này xảy ra do phân phối nhãn trong tập train có số lượng mẫu `trung_binh` và `cao` chiếm tỷ trọng lớn hơn, khiến adapter học thiên kiến ưu tiên nhãn phổ biến hơn khi gặp các tín hiệu biên.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**
Bản fine-tune LoRA này hoàn toàn xứng đáng được cân nhắc triển khai lên môi trường thử nghiệm và phục vụ thực tế (production-candidate). Về mặt nghiệp vụ, nó nâng độ chính xác phân loại tự động lên 93.75%, tạo ra định dạng JSON 100% hợp lệ và kiểm soát tốt độ trễ. 

Qua toàn bộ chuỗi thực nghiệm, bài học rõ ràng nhất là **Learning Rate và Vị trí đặt Adapter** mới chính là đòn bẩy công nghệ thực sự, chứ không phải việc cố gắng nhồi nhét rank cực lớn hay phụ thuộc vào train loss. Đặt đúng LR (1e-4) và áp dụng LoRA lên toàn bộ các lớp linear (`text-linear`) mang lại hiệu quả vượt trội, trong khi Loss Masking (`assistant-only`) là điều kiện tiên quyết đảm bảo mô hình không học vẹt prompt đầu vào.

**Ba điều tôi học được:**
1. **Mask Proof là ranh giới sống còn:** Nếu không kiểm tra và che mask ở phần prompt/hỏi (chỉ tính loss trên phần trả lời), toàn bộ quá trình fine-tuning sẽ biến thành một mô hình học vẹt và phá hủy khả năng suy luận.
2. **Train loss không phản ánh chất lượng thực tế:** Run `attn_only` đạt train loss thấp hơn `correct` nhưng điểm target downstream lại không vượt trội; tối ưu hóa trên tập train không đồng nghĩa với năng lực tổng quát hóa.
3. **Prompting là baseline bắt buộc phải vượt qua:** Trước khi quyết định fine-tune, luôn cần đo lường cẩn thận một baseline Prompt Engineering tối ưu. Fine-tuning chỉ có giá trị khi nó chứng minh được sự vượt trội vượt bậc (+0.25 target) mà không làm suy giảm năng lực tổng quát (regression gate).

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ thử nghiệm kỹ thuật **DPO (Direct Preference Optimization)** hoặc cân bằng lại tỷ lệ phân phối nhãn cho trường `urgency=thap` trong tập dữ liệu huấn luyện, đồng thời tích hợp thêm 3% dữ liệu hội thoại đa chủ đề để củng cố độ bền vững chống suy giảm năng lực tổng quát.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
