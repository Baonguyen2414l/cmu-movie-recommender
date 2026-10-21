"""Chạy bằng: python -m streamlit run app.py"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.recommender import get_top_k_recommendations, prepare_embeddings


BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Gợi ý phim", page_icon="🎬", layout="wide")


# Tìm file trong thư mục dữ liệu, hoặc ngay cạnh app.py.
def find_file(filename, folder):
    for path in (BASE_DIR / folder / filename, BASE_DIR / filename):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Thiếu {filename}. Đặt file trong {folder}/ hoặc cạnh app.py.")


# Đọc và chuẩn hóa dữ liệu; file_versions giúp tải lại cache khi file thay đổi.
@st.cache_resource
def load_resources(catalog_path, model_files, file_versions):
    catalog = pd.read_csv(catalog_path)
    required = {"title", "release_date", "genres_str", "raw_text"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog thiếu cột: {', '.join(sorted(missing))}")
    if len(catalog) < 6:
        raise ValueError("Cần ít nhất 6 phim để gợi ý Top 5.")
    catalog = catalog.fillna("")
    embeddings = {}
    for name, path in model_files:
        matrix = np.load(path, allow_pickle=False)
        try:
            embeddings[name] = prepare_embeddings(matrix, len(catalog))
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    return catalog, embeddings


# Tách chuỗi thể loại của một phim thành tập hợp để tìm thể loại trùng nhau.
def genres(row):
    return {item.strip() for item in str(row["genres_str"]).split("|") if item.strip()}


# Tạo tên hiển thị trong ô chọn phim, phân biệt các phim trùng tên.
def movie_label(index):
    row = catalog.iloc[index]
    year = str(row["release_date"])[:4] or "Chưa rõ năm"
    # Chỉ số giúp phân biệt cả những phim trùng tên và năm phát hành.
    return f"{row['title']} ({year}) · #{index}"


st.title("🎬 Gợi ý phim theo nội dung tóm tắt")
st.write("Tìm Top 5 phim tương đồng và so sánh TF-IDF với Sentence-BERT.")

try:
    catalog_path = find_file("movies_catalog.csv", "data")
    tfidf_path = find_file("tfidf_embeddings.npy", "embeddings")
    try:
        sbert_path = find_file("sbert_embeddings.npy", "embeddings")
    except FileNotFoundError:
        # Chấp nhận cả tên số ít từ notebook của nhóm.
        sbert_path = find_file("sbert_embedding.npy", "embeddings")
    model_files = (("TF-IDF", tfidf_path), ("Sentence-BERT", sbert_path))
    paths = (catalog_path, tfidf_path, sbert_path)
    file_versions = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    catalog, embeddings = load_resources(catalog_path, model_files, file_versions)
except (OSError, ValueError, KeyError) as error:
    st.error(str(error))
    st.info("Cần movies_catalog.csv, tfidf_embeddings.npy và sbert_embeddings.npy từ nhóm.")
    st.stop()

with st.sidebar:
    st.header("Chọn phim")
    query_idx = st.selectbox("Tìm phim theo tên", range(len(catalog)), format_func=movie_label)
    mode = st.radio("Mô hình", ["So sánh cả hai", *embeddings])
    submitted = st.button("Gợi ý Top 5", type="primary")

query = catalog.iloc[query_idx]
query_genres = genres(query)
with st.expander("Thông tin phim đang chọn", expanded=True):
    st.subheader(str(query["title"]))
    st.write(f"Năm: {str(query['release_date'])[:4] or 'Chưa rõ'}")
    st.write("Thể loại: " + (", ".join(sorted(query_genres)) or "Chưa có"))
    st.write(str(query["raw_text"]))

if submitted:
    selected_models = list(embeddings) if mode == "So sánh cả hai" else [mode]
    for column, name in zip(st.columns(len(selected_models)), selected_models):
        with column:
            matrix = embeddings[name]
            indices, scores, latency = get_top_k_recommendations(query_idx, matrix)
            st.subheader(name)
            st.caption(f"Truy xuất: {latency:.2f} ms | Vector: {matrix.shape[1]} chiều")
            for rank, (index, score) in enumerate(zip(indices, scores), start=1):
                row = catalog.iloc[int(index)]
                matched = query_genres & genres(row)
                with st.container(border=True):
                    st.markdown(f"**#{rank}. {row['title']} ({str(row['release_date'])[:4]})**")
                    st.write(f"Cosine Similarity: {score:.4f}")
                    st.write("Thể loại: " + ", ".join(sorted(genres(row))))
                    if matched:
                        st.success("Trùng thể loại: " + ", ".join(sorted(matched)))
                    st.caption(str(row["raw_text"])[:300] + ("…" if len(str(row["raw_text"])) > 300 else ""))
else:
    st.info("Chọn phim và mô hình, sau đó bấm Gợi ý Top 5.")
