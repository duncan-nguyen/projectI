# Vietnamese Medical Information Extraction Pipeline

Hệ thống trích xuất thông tin y tế (Information Extraction - IE) từ văn bản tiếng Việt, hỗ trợ nhận diện thực thể (NER) và trích xuất quan hệ (RE). Dự án cung cấp cả thư viện Python (`src`) và giao diện demo trực quan (`app.py`).

## Tính năng chính

1.  **Tiền xử lý văn bản**:
    *   Làm sạch dữ liệu (xóa HTML tag, chuẩn hóa Unicode).
    *   Tách từ (Tokenization) hỗ trợ mô hình học máy.

2.  **Nhận diện thực thể**: Hỗ trợ 2 phương pháp:
    *   **Standard NER**: Sử dụng mô hình PhoBERT tinh chỉnh cho y tế.
    *   **GLiNER**: Sử dụng mô hình GLiNER cho khả năng zero-shot/few-shot mạnh mẽ.
    *   **Các nhãn hỗ trợ**: `PATIENT_ID`, `NAME`, `AGE`, `GENDER`, `JOB`, `LOCATION`, `ORGANIZATION`, `SYMPTOM_AND_DISEASE`, `TRANSPORTATION`, `DATE`.

3.  **Trích xuất quan hệ**:
    *   Sử dụng hệ thống dựa trên luật kết hợp từ khóa và khoảng cách ngữ cảnh.
    *   **Quan hệ hỗ trợ**:
        *   `LIVED_AT`: Bệnh nhân - Địa điểm cư trú.
        *   `HAS_SYMPTOM`: Bệnh nhân - Triệu chứng.
        *   `VISITED`: Bệnh nhân - Địa điểm đã đi qua.

4.  **Trực quan hóa**:
    *   Giao diện Web tương tác với **Streamlit**.
    *   Biểu đồ tri thức sử dụng **NetworkX** và **Matplotlib**.
    *   Bảng dữ liệu chi tiết cho thực thể và quan hệ.

## Cài đặt

### Yêu cầu
*   Python >= 3.10
*   Khuyên dùng môi trường ảo (venv, conda, hoặc uv).

### Các bước cài đặt

1.  **Clone dự án**:
    ```bash
    git clone https://github.com/your-username/projectI.git
    cd projectI
    ```

2.  **Cài đặt thư viện**:
    Dự án sử dụng file `pyproject.toml` để quản lý dependencies.

    *   Sử dụng `pip`:
        ```bash
        pip install .
        ```
    *   Sử dụng `uv` (recommended):
        ```bash
        uv sync
        ```

## Hướng dẫn sử dụng

### 1. Chạy Demo UI 
Để trải nghiệm giao diện trực quan:

```bash
streamlit run app.py
```
Giao diện sẽ mở tại `http://localhost:8501`. Tại đây bạn có thể:
*   Nhập văn bản y tế tiếng Việt.
*   Chọn mô hình NER (GLiNER hoặc Standard).
*   Xem kết quả phân tích và đồ thị tri thức.

### 2. Sử dụng Library
Bạn có thể nhúng pipeline vào dự án Python của mình:

```python
from src.pipeline import InformationExtractionPipeline

# Khởi tạo pipeline (chọn method="gliner" hoặc "standard")
pipeline = InformationExtractionPipeline(ner_method="gliner", device="cpu")

text = "Bệnh nhân Nguyễn Văn A, 45 tuổi, trú tại Hà Nội, có biểu hiện sốt cao."

# Xử lý văn bản
result = pipeline.process(text)

# Truy cập kết quả
print("Entities:", result.entities)
print("Relations:", result.relations)

# Hiển thị nhanh
pipeline.visualize(result)
```

## Cấu trúc dự án

```
projectI/
├── app.py                  # Mã nguồn giao diện Streamlit
├── pyproject.toml          # Quản lý dependencies
├── README.md               # Tài liệu dự án
├── notebook/               # Jupyter Notebooks huấn luyện mô hình
│   ├── finetune-gliner.ipynb
│   └── finetune-standard-ner.ipynb
└── src/                    # Mã nguồn chính
    ├── __init__.py
    ├── models.py           # Định nghĩa Data Model (Pydantic)
    ├── preprocessing.py    # Module tiền xử lý
    ├── ner.py              # Module Named Entity Recognition
    ├── re_module.py        # Module Relation Extraction (Rule-based)
    ├── visualization.py    # Module vẽ đồ thị
    └── pipeline.py         # Class pipeline chính
```