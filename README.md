# 🎬 Hệ Thống Gợi Ý Phim Dựa Trên Nội Dung Tóm Tắt (Content-Based Movie Recommender)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Bi--LSTM-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/LLM%20Embedding-all--MiniLM--L6--v2-FFD21E?logo=huggingface&logoColor=black)
![Gensim](https://img.shields.io/badge/Gensim-Doc2Vec-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Demo-FF4B4B?logo=streamlit&logoColor=white)

> **Đồ án môn học:** Phân tích nội dung tóm tắt cốt truyện phim để gợi ý **Top 5 phim tương đồng**, đồng thời thực nghiệm đối chiếu chất lượng và thời gian xử lý giữa 4 phương pháp biểu diễn văn bản (**TF-IDF, Doc2Vec, Bi-LSTM, LLM Embedding**) trên tập dữ liệu **MovieLens 1M**.
> - 📄 **Báo cáo đồ án (Google Docs):** [Xem chi tiết tại đây](https://docs.google.com/document/d/1pzEL7sKUjEBS_7Xv4S2NCTLpbNPovbbQYAVRzs3Jnek/edit?usp=sharing)
> - 🔗 **Nguồn dữ liệu:** [GroupLens MovieLens](https://grouplens.org/datasets/movielens/) | [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

---

## 📌 1. Giới thiệu bài toán
Hệ thống giải quyết bài toán **Gợi ý dựa trên nội dung (Content-Based Filtering)** thông qua văn bản tóm tắt cốt truyện phim (Plot Overview) nhằm khắc phục vấn đề khởi động lạnh (Cold-Start Problem):
- **Đầu vào:** Một bộ phim bất kỳ do người dùng chọn từ danh mục MovieLens 1M.
- **Xử lý:** Biểu diễn nội dung tóm tắt thành vector đặc trưng và tính độ tương đồng **Cosine Similarity** trên toàn bộ kho phim.
- **Đầu ra:** **Top 5 bộ phim có nội dung tương đồng nhất** (loại trừ phim đầu vào), kèm điểm số tương đồng, nhãn thể loại và thời gian truy xuất (Latency).

---

## 🧠 2. Các mô hình biểu diễn văn bản đối chiếu

| STT | Phương pháp | Trường phái | Cách thức hoạt động trong dự án |
| :-: | :--- | :--- | :--- |
| **1** | **TF-IDF** | Thống kê từ vựng (Baseline) | Biểu diễn vector thưa (10.000 chiều) dựa trên tần suất từ và cụm từ (`ngram_range=(1,2)`). |
| **2** | **Doc2Vec** | Học biểu diễn không giám sát | Học trực tiếp vector cấp độ đoạn văn (Paragraph Vector - 300 chiều) nắm bắt ngữ cảnh tổng thể. |
| **3** | **Bi-LSTM** | Mạng hồi quy sâu (PyTorch) | Mạng LSTM hai chiều xử lý chuỗi tuần tự để trích xuất phụ thuộc ngữ cảnh dài trong cốt truyện. |
| **4** | **LLM Embedding** | Pre-trained Transformer | Sử dụng mô hình `all-MiniLM-L6-v2` tạo vector nhúng ngữ nghĩa sâu (384 chiều). |

---

## 📊 3. Tập dữ liệu & Quy trình tiền xử lý

- **Nguồn dữ liệu:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) (`movies.dat`, `ratings.dat`, `users.dat`) kết hợp trường tóm tắt cốt truyện (`overview`) từ **TMDb / Kaggle**.
- **Quy trình tiền xử lý (Chống Data Leakage):**
  1. Đọc dữ liệu MovieLens 1M (phân tách `::`, bảng mã `latin-1`) và ánh xạ tóm tắt phim từ TMDb (thu được **2.353 phim** hợp lệ có tóm tắt $\ge 20$ từ).
  2. Chuẩn bị 2 định dạng văn bản: `raw_text` (cho Bi-LSTM, LLM) và `clean_text` (loại bỏ stopwords, ký tự đặc biệt cho TF-IDF, Doc2Vec). Tuyệt đối không đưa nhãn thể loại vào văn bản đầu vào.
  3. Chia dữ liệu theo tỷ lệ **70% Train – 15% Validation – 15% Test** bằng `MultilabelStratifiedShuffleSplit`.
  4. Chỉ học từ điển và trọng số trên tập **Train (70%)**, sau đó trích xuất vector chuẩn hóa L2 để tính Cosine Similarity.

---

## 📏 4. Độ đo đánh giá (Evaluation Metrics)

Đánh giá độc lập trên tập **Test (15%)**:
- **Quy ước liên quan (Ground Truth):** Phim gợi ý được tính là liên quan ($r_i = 1$) nếu có **chung ít nhất 1 thể loại** với phim truy vấn.
- **Chỉ số đo lường tại Top 5:** `Precision@5`, `Recall@5`, `NDCG@5` và `Latency (ms)`.

---

## 🗂️ 5. Cấu trúc thư mục dự án

```text
movielens-1m-recommender/
├── notebooks/                        # Sổ tay thực nghiệm trên Google Colab
│   ├── 01_eda_and_preprocess.ipynb   # Bước 1: Ghép dữ liệu MovieLens 1M + TMDb & Tiền xử lý
│   ├── 02_tfidf_and_doc2vec.ipynb    # Bước 2A: Huấn luyện & trích xuất TF-IDF, Doc2Vec
│   ├── 03_bilstm_and_llm.ipynb       # Bước 2B: Huấn luyện Bi-LSTM & trích xuất LLM Embedding
│   └── 04_evaluation.ipynb           # Bước 3: Tính Precision@5, Recall@5, NDCG@5 & Biểu đồ
├── src/                              # Module xử lý hệ thống
│   ├── evaluate.py                   # Hàm tính độ đo đánh giá
│   └── recommender.py                # Engine truy xuất Top 5 phim bằng Cosine Similarity
├── data/                             # Dữ liệu MovieLens 1M & catalog đã làm sạch
├── embeddings/                       # Ma trận vector (.npy / .npz)
├── app.py                            # Ứng dụng Web Demo (Streamlit)
├── requirements.txt                  # Thư viện cần thiết
└── README.md                         # Tài liệu hướng dẫn dự án
