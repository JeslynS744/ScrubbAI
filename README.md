# Scrubb AI

[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Scrubb AI** is an intelligent data profiling, EDA, and automated cleaning platform built with Streamlit and OpenRouter LLMs.

---

## 🌟 Key Features

* **Holistic Data Health & Profiling:** Instantly ingests multi-format datasets (CSV, Excel, Parquet, JSON) to generate comprehensive health scorecards and automated feature diagnostics.
* **Interactive Exploratory Data Analysis (EDA):** Powers dynamic Plotly visualizations, correlation heatmaps, and automated IQR-based outlier detection.
* **Privacy-Preserving AI Analyst:** Uses OpenRouter LLMs strictly on aggregate statistics to deliver secure data diagnostic insights and actionable remediation plans.
* **Interactive Data Transformation Suite:** Offers an intuitive UI for deduplication, missing value imputation, data type casting, and outlier mitigation.
* **Reproducibility & Export:** Exports cleaned datasets instantly while auto-generating standalone, runnable Pandas Python scripts for batch pipelines.
* **Built-in User Authentication:** Secures user accounts via SQLite-backed PBKDF2-HMAC-SHA256 password hashing and isolated session management.

---

## 🚀 Quick Start

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/JeslynS744/ScrubbAI.git
cd ScrubbAI
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 | Mac/Linux: source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key (Optional)
Set your OpenRouter API key via environment variable:
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
ScrubbAI/
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```