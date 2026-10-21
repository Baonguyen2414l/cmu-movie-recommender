"""Truy xuất phim theo Cosine Similarity, giữ nguyên thứ tự catalog."""

from time import perf_counter

import numpy as np


# Kiểm tra số dòng, giá trị và chuẩn hóa L2 để tính cosine bằng tích vô hướng.
def prepare_embeddings(matrix, movie_count):
    """Kiểm tra và chuẩn hóa một lần khi tải dữ liệu."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != movie_count or matrix.shape[1] == 0:
        raise ValueError("Ma trận vector phải có một dòng cho mỗi phim trong catalog.")
    if not np.isfinite(matrix).all():
        raise ValueError("Ma trận vector chứa NaN hoặc giá trị vô hạn.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Vector toàn số 0 được giữ nguyên, điểm tương đồng bằng 0.
    return matrix / np.where(norms > 0, norms, 1)


# Tính điểm với toàn bộ kho phim, bỏ phim đầu vào và lấy k phim đứng đầu.
def get_top_k_recommendations(query_idx, matrix, k=5):
    """Matrix đã chuẩn hóa L2; trả về chỉ số dòng, điểm và thời gian ms."""
    if not 0 <= query_idx < len(matrix):
        raise ValueError("Chỉ số phim không hợp lệ.")
    if k < 1:
        raise ValueError("k phải lớn hơn 0.")
    start = perf_counter()
    scores = np.clip(matrix @ matrix[query_idx], -1.0, 1.0)
    indices = np.argsort(-scores, kind="stable")
    indices = indices[indices != query_idx][:k]
    latency_ms = (perf_counter() - start) * 1000
    return indices, scores[indices], latency_ms
