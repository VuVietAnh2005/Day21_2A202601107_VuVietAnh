# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là sự mâu thuẫn giữa `train loss` và hiệu năng thực tế `target accuracy` khi giải phẫu run `attn_only` (r=283) so với `correct` (r=16). `attn_only` đạt train loss thấp hơn rõ rệt (0.5376 so với 0.6262), nhưng độ chính xác target lại không hề vượt trội. Ngoài ra, việc QLoRA 4-bit bị sụt giảm độ chính xác rõ rệt (từ 0.9375 xuống 0.8438) trên dòng model Qwen3.5 đã chứng minh thực nghiệm rằng lượng tử hóa 4-bit không phải lúc nào cũng là giải pháp "miễn phí" như lý thuyết thường quảng bá.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở công đoạn chờ GPU sinh văn bản đánh giá (evaluation generation ở NB2 và NB5 với 3 lượt sinh) và chạy 3 run đối chứng ở NB4 (tổng cộng ~78 phút trên Colab T4). 

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước giờ mình chưa làm về phần code web chứ chưa học chính về mảng AI

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để định hướng luồng setup giữa Local và Colab, phân tích đối chiếu các con số thực nghiệm từ các file JSON/CSV trong `results/`, và hỗ trợ rà soát các tiêu chí Rubric. Chỗ AI chưa hoàn hảo là lúc đầu chạy kiểm thử `verify.py` trên môi trường Windows, AI chưa tính đến việc Git tự động chuyển đổi ký tự xuống dòng từ LF sang CRLF làm thay đổi mã băm SHA256 của tập dữ liệu `data/*.jsonl`, sau đó phải xử lý ép định dạng LF thì mã băm mới khớp.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi sẽ làm là **Đóng băng một tập đánh giá chuẩn (Golden Eval Set) và xây dựng baseline Prompt Engineering tối ưu nhất có thể**, kèm theo thiết lập **cơ chế Loss Masking chuẩn (`assistant-only`)**. Tôi sẽ chỉ đề xuất khách hàng bỏ chi phí hạ tầng và thời gian để fine-tune nếu và chỉ nếu baseline Prompting không thể đáp ứng được yêu cầu về độ chính xác, định dạng JSON nghiêm ngặt hoặc bài toán chi phí/độ trễ token.
