##  Student Exam Analysis
###  Giới thiệu

Student Exam Analysis là một project Python dùng để phân tích dữ liệu kết quả học tập của sinh viên.
Dự án giúp xử lý dữ liệu thô, trực quan hóa và rút ra insight về các yếu tố ảnh hưởng đến điểm số.

###  Tính năng chính
- Đọc dữ liệu từ file CSV
- Làm sạch dữ liệu (missing values, duplicates, format)
- Phân tích thống kê cơ bản
- Trực quan hóa dữ liệu (biểu đồ)
- Tìm mối quan hệ giữa các yếu tố và kết quả thi

### Cấu trúc project
```
StudentExamAnalysis/
│── data/
│   ├── raw/                     # Dữ liệu gốc
│   └── processed/              # Dữ liệu sau khi xử lý
│
│── scripts/
│   ├── clean_data.py           # Làm sạch dữ liệu
│   ├── analyze_data.py         # Phân tích dữ liệu
│   └── visualize.py            # Vẽ biểu đồ
│
│── outputs/                    # Kết quả (ảnh, báo cáo)
│
│── main.py                     # Chạy project
│── requirements.txt            # Thư viện cần thiết
│── README.md                   # Tài liệu project
```
### Cài đặt
1. Clone repo
```bash
git clone https://github.com/your-username/StudentExamAnalysis.git
cd StudentExamAnalysis
```
2. Tạo virtual environment
```bash
python -m venv .venv
```
3. Kích hoạt môi trường

- Windows:
```
.venv\Scripts\activate
```
- Mac/Linux:
```
source .venv/bin/activate
```
4. Cài thư viện
```bash
pip install -r requirements.txt
```
### Cách sử dụng
1. Làm sạch dữ liệu
python scripts/clean_data.py
2. Phân tích dữ liệu
python scripts/analyze_data.py
3. Trực quan hóa
python scripts/visualize.py

### Dataset
1. File dữ liệu nằm tại:
- data/raw/StudentPerformanceFactors.csv
2. Dữ liệu bao gồm:
- Hours Studied
- Attendance
- Sleep Hours
- Previous Scores
- Exam Score
- ...
### Ví dụ output
- Biểu đồ phân phối điểm số
- Biểu đồ so sánh thời gian học vs điểm
- Heatmap correlation

### Công nghệ sử dụng
- Python 
- Pandas
- NumPy
- Matplotlib / Seaborn

### License

Dự án phục vụ mục đích học tập.