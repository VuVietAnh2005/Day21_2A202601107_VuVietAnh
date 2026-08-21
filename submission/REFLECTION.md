# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là sự mâu thuẫn giữa `train loss` và hiệu năng thực tế `target accuracy` khi giải phẫu run `attn_only` (r=283) so với `correct` (r=16). `attn_only` đạt train loss thấp hơn rõ rệt (0.5376 so với 0.6262), nhưng độ chính xác target lại không hề vượt trội. Ngoài ra, việc QLoRA 4-bit bị sụt giảm độ chính xác rõ rệt (từ 0.9375 xuống 0.8438) trên dòng model Qwen3.5 đã chứng minh thực nghiệm rằng lượng tử hóa 4-bit không phải lúc nào cũng là giải pháp "miễn phí" như lý thuyết thường quảng bá.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở công đoạn chờ GPU sinh văn bản đánh giá (evaluation generation ở NB2 và NB5 với 3 lượt sinh) và chạy 3 run đối chứng ở NB4 (tổng cộng ~78 phút trên Colab T4). Ban đầu tôi dự đoán bước huấn luyện LoRA chính (NB3) sẽ tốn thời gian nhất, nhưng thực tế việc chạy 3 run đối chứng độc lập cùng bước sinh văn bản autoregressive để đo đạc khách quan mới là phần chiếm trọn ngân sách thời gian.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin rằng:
1. Cứ tăng rank LoRA thật lớn và tập trung vào module Attention là mô hình sẽ học tốt nhất.
2. Train loss liên tục giảm là mô hình đang tiến bộ.
3. Fine-tuning luôn là giải pháp bắt buộc và luôn vượt trội hoàn toàn so với Prompt Engineering.
Sau lab, tôi nhận ra vị trí phủ adapter (`all text-linear`) quan trọng hơn rank rất nhiều, và một prompt tối ưu chuẩn mực (baseline b) đã có thể giải quyết được gần 70% bài toán với chi phí huấn luyện bằng 0.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để định hướng luồng setup giữa Local và Colab, phân tích đối chiếu các con số thực nghiệm từ các file JSON/CSV trong `results/`, và hỗ trợ rà soát các tiêu chí Rubric. Chỗ AI chưa hoàn hảo là lúc đầu chạy kiểm thử `verify.py` trên môi trường Windows, AI chưa tính đến việc Git tự động chuyển đổi ký tự xuống dòng từ LF sang CRLF làm thay đổi mã băm SHA256 của tập dữ liệu `data/*.jsonl`, sau đó phải xử lý ép định dạng LF thì mã băm mới khớp.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi sẽ làm là **Đóng băng một tập đánh giá chuẩn (Golden Eval Set) và xây dựng baseline Prompt Engineering tối ưu nhất có thể**, kèm theo thiết lập **cơ chế Loss Masking chuẩn (`assistant-only`)**. Tôi sẽ chỉ đề xuất khách hàng bỏ chi phí hạ tầng và thời gian để fine-tune nếu và chỉ nếu baseline Prompting không thể đáp ứng được yêu cầu về độ chính xác, định dạng JSON nghiêm ngặt hoặc bài toán chi phí/độ trễ token.
