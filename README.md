# 🎬 Hệ Thống Gợi Ý Phim Dựa Trên Nội Dung Tóm Tắt (Content-Based Movie Recommendation)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Sentence%20Transformers-all--MiniLM--L6--v2-FFD21E?logo=huggingface&logoColor=black)
![Gensim](https://img.shields.io/badge/Gensim-Doc2Vec-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Demo-FF4B4B?logo=streamlit&logoColor=white)

> **Đồ án môn học:** Phân tích nội dung tóm tắt phim để gợi ý **Top 5 phim tương đồng** với phim người dùng chọn, đồng thời đánh giá và so sánh chất lượng gợi ý cũng như thời gian xử lý giữa 4 phương pháp biểu diễn văn bản từ truyền thống đến hiện đại.
Link mô tả đồ án word:https://docs.google.com/document/d/1pzEL7sKUjEBS_7Xv4S2NCTLpbNPovbbQYAVRzs3Jnek/edit?usp=sharing
> 
---

## 📌 1. Giới thiệu bài toán
Hệ thống tập trung giải quyết bài toán **Gợi ý dựa trên nội dung (Content-Based Filtering)** thông qua văn bản tóm tắt cốt truyện phim (Plot Summary).
* **Đầu vào:** Một bộ phim bất kỳ do người dùng chọn.
* **Xử lý:** Biểu diễn nội dung tóm tắt phim thành các vector đặc trưng và tính độ tương đồng **Cosine Similarity** trên toàn bộ kho phim.
* **Đầu ra:** Danh sách **Top 5 bộ phim có nội dung gần nhất** (loại trừ phim đầu vào), kèm theo điểm tương đồng, thông tin thể loại và thời gian truy xuất thực tế.

---

## 🧠 2. Các thuật toán và Mô hình so sánh

Dự án thực nghiệm và đối chiếu 4 phương pháp biểu diễn văn bản thuộc các trường phái khác nhau:

| STT | Phương pháp | Trường phái | Cách thức hoạt động trong dự án |
| :-: | :--- | :--- | :--- |
| **1** | **TF-IDF** | Thống kê từ vựng (Lexical Baseline) | Biểu diễn văn bản dựa trên tần suất từ và cụm từ (`ngram_range=(1,2)`), dùng làm phương pháp cơ sở để đối chiếu. |
| **2** | **Doc2Vec**<br>*(Le & Mikolov, 2014)* | Học biểu diễn không giám sát | Học trực tiếp vector cấp độ văn bản (Paragraph Vector) từ tập dữ liệu huấn luyện để nắm bắt ngữ cảnh tổng thể của đoạn tóm tắt. |
| **3** | **TextCNN**<br>*(Kim, 2014)* | Deep Learning có giám sát (PyTorch) | Huấn luyện mạng CNN 1 chiều (`kernel_size = 3, 4, 5`) cho bài toán dự đoán thể loại đa nhãn (Multi-label), sau đó trích xuất vector đặc trưng trước tầng phân loại để gợi ý phim. |
| **4** | **Sentence Transformer**<br>*(`all-MiniLM-L6-v2`)* | Pre-trained Language Model | Sử dụng mô hình ngôn ngữ tiền huấn luyện tạo vector ngữ nghĩa sâu (384 chiều) cho nội dung tóm tắt phim. |

---

## 📊 3. Tập dữ liệu & Quy trình tiền xử lý:

* **Nguồn dữ liệu:** **CMU Movie Summary Corpus** (Đại học Carnegie Mellon) gồm tóm tắt cốt truyện từ Wikipedia (`plot_summaries.txt`) và thông tin siêu dữ liệu từ Freebase (`movie.metadata.tsv`).
  * 🔗 [Trang công bố tập dữ liệu](https://www.cs.cmu.edu/~ark/personas/) | [Link tải trực tiếp (.tar.gz)](https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz)
* **Quy trình xử lý chuẩn (Chống rò rỉ dữ liệu - Data Leakage):**
  1. Ghép dữ liệu theo `movie_id`, loại bỏ dữ liệu thiếu, trùng lặp và các bản tóm tắt quá ngắn/dài.
  2. Khảo sát phân bố và lọc giữ lại **Top 15 thể loại phổ biến** cùng **5.000 – 10.000 phim hợp lệ**.
  3. Làm sạch văn bản (**không** đưa nhãn thể loại vào văn bản đầu vào).
  4. Chia tập dữ liệu theo tỷ lệ **70% Train – 15% Validation – 15% Test** bằng kỹ thuật phân tầng đa nhãn (`MultilabelStratifiedShuffleSplit`).
  5. **Chỉ học tham số mô hình trên tập Train (70%)**, sau đó trích xuất vector để tính độ tương đồng Cosine.

---

## 📏 4. Độ đo đánh giá hiệu suất (Evaluation Metrics)

Các phương pháp được đánh giá công bằng trên các phim truy vấn thuộc tập **Test (15%)** với cùng một tập ứng viên (Candidate Pool):
* **Quy ước phim liên quan (Relevance):** Phim gợi ý được tính là liên quan ($r_i = 1$) nếu có **chung ít nhất 1 thể loại** với phim truy vấn trong tập thể loại đã chọn.
* **Các chỉ số đo lường ở Top 5:**
  * `Precision@5`: Tỷ lệ phim liên quan trong 5 phim được gợi ý.
  * `Recall@5`: Tỷ lệ phim liên quan tìm được so với toàn bộ phim liên quan trong kho.
  * `NDCG@5` *(Normalized Discounted Cumulative Gain)*: Đánh giá chất lượng thứ bậc xếp hạng của danh sách Top 5.
  * `Latency (ms)`: Thời gian xử lý và truy xuất kết quả gợi ý.

---

## 🗂️ 5. Cấu trúc  gợi í thư mục dự án

```text
cmu-movie-recommender/
│
├── notebooks/                        # Sổ tay thực nghiệm trên Google Colab
│   ├── 01_eda_and_preprocess.ipynb   # Bước 1: Khảo sát EDA, làm sạch & chia tập dữ liệu
│   ├── 02_tfidf_and_doc2vec.ipynb    # Bước 2A: Huấn luyện và trích xuất vector TF-IDF, Doc2Vec
│   ├── 03_textcnn_and_sbert.ipynb    # Bước 2B: Huấn luyện TextCNN và trích xuất Sentence-BERT
│   └── 04_evaluation.ipynb           # Bước 3: Chấm điểm Precision@5, Recall@5, NDCG@5 & Vẽ biểu đồ
│
├── src/                              # Các module xử lý dùng chung cho hệ thống
│   ├── evaluate.py                   # Các hàm tính độ đo đánh giá
│   └── recommender.py                # Engine truy xuất Top 5 phim bằng Cosine Similarity
│
├── data/                             # (Đã ẩn qua .gitignore - Lưu trên Google Drive nhóm)
├── embeddings/                       # (Chứa các file ma trận vector .npy / .npz tải từ Drive về)
├── app.py                            # Ứng dụng Web Demo viết bằng Streamlit
├── requirements.txt                  # Danh sách thư viện cần thiết
└── README.md                         # Tài liệu hướng dẫn dự án
