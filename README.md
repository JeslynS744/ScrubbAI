# 🧹 Scrubb AI — Intelligent Data Profiling, EDA & Cleaning Assistant

[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1+-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg?style=for-the-badge)](LICENSE)

**Scrubb AI** is an end-to-end, privacy-conscious data profiling, exploratory data analysis (EDA), and automated data cleaning platform built with Streamlit and OpenRouter LLMs. It empowers data scientists, analysts, and engineers to diagnose data quality issues, interactively clean messy datasets, receive AI-driven remediation roadmaps, and generate production-ready reproducible transformation scripts in seconds.

---

## 🌟 Key Features

### 1. 📊 Holistic Data Health & Profiling
- **Multi-Format Ingestion**: Seamlessly load datasets up to 100MB in **CSV**, **Excel (XLSX/XLS)**, **Parquet**, and **JSON**.
- **Instant Health Scorecard**: View high-level metrics including total rows, detected features, missing cell percentage, and exact duplicate row counts.
- **Deep Feature Diagnostics**: Automated column-by-column profiling classifying features into numeric, categorical, datetime, text, or boolean with cardinality, null rates, and distribution summaries.

### 2. 🔍 Interactive Exploratory Data Analysis (EDA)
- **Dynamic Visualizations**: High-performance interactive charts powered by **Plotly** (Histograms, Box plots, Scatter plots, Bar charts, Correlation heatmaps).
- **Outlier Detection**: Automated IQR (Interquartile Range) boundary calculations highlighting potential anomalies.
- **Multivariate Analysis**: Easily correlate numerical attributes and uncover hidden relationships within your dataset.

### 3. 🤖 Privacy-Preserving AI Analyst
- **OpenRouter LLM Integration**: Connects with modern language models (e.g., `openai/gpt-4o-mini`, Claude 3.5 Sonnet, etc.) via OpenRouter.
- **Privacy-First Architecture**: **Zero raw records are sent to external APIs**. The engine extracts aggregate statistics and column schemas, ensuring strict compliance and data security.
- **Structured Action Plans**: Receives actionable, categorized recommendations:
  - 🚨 **Prioritized Actions**: High, Medium, and Low severity issues with direct fix instructions.
  - 🛠️ **Missing Value Strategies**: Tailored imputation strategies (Mean, Median, Mode, Forward Fill) with rationale and confidence levels.
  - 📈 **Visualization Recommendations**: Targeted chart suggestions designed to reveal domain-specific patterns.

### 4. 🧼 Interactive Data Transformation Suite
- **Duplicate Removal**: One-click deduplication.
- **Flexible Missing Value Remediation**: Drop nulls or impute values using Mean, Median, Mode, Forward-fill, or custom placeholders.
- **Data Type Casting**: Convert dates, integers, floats, and strings effortlessly.
- **Outlier Mitigation**: Filter or winsorize extreme values.

### 5. 💾 Reproducibility & Export
- **Cleaned Data Download**: Instant export to CSV or Excel formats.
- **Python Script Generator**: Automatically constructs a standalone, runnable Python script that mirrors all UI transformations using native Pandas code for batch pipelines.

### 6. 🔐 Built-in User Authentication
- **Multi-User Ready**: SQLite database with industry-standard `PBKDF2-HMAC-SHA256` password hashing (310,000 iterations).
- **Session Isolation**: Each user session is isolated in memory with fast reset and logout capabilities.

---

## 🏗️ Architecture & Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend & UI** | Streamlit, Custom Modern CSS (Plus Jakarta Sans, Glassmorphism) |
| **Data Processing** | Python 3.10+, Pandas, NumPy, OpenPyXL |
| **Interactive Charts** | Plotly Express |
| **LLM Reasoning** | OpenRouter API (`openai/gpt-4o-mini`, etc.) |
| **Authentication** | SQLite3, `hashlib` (PBKDF2-HMAC-SHA256), `hmac`, `secrets` |

---

## 🚀 Quick Start Guide

### Prerequisites
- Python **3.10** or higher
- Git
- An OpenRouter API key ([Get one here](https://openrouter.ai/keys)) *(Optional, for AI Analyst features)*

### 1. Clone the Repository
```bash
git clone https://github.com/JeslynS744/ScrubbAI.git
cd ScrubbAI
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
You can configure your OpenRouter API key using either Streamlit Secrets or Environment Variables:

#### Option A: Streamlit Secrets (Recommended for local development)
1. Copy the example secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
2. Open `.streamlit/secrets.toml` and set your key:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_api_key_here"
   ```

#### Option B: Environment Variable
```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Linux / macOS
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📁 Project Structure

```text
ScrubbAI/
├── .streamlit/
│   ├── secrets.toml.example  # Example Streamlit configuration template
├── .env.example              # Example environment variable template
├── .gitignore                # Git ignore configuration (protects keys and DB)
├── app.py                    # Main Streamlit application and backend logic
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation and setup guide
```

---

## 🔒 Security & Privacy

- **Data Privacy**: No row-level user data is ever transmitted over the network to LLMs. Only aggregate statistical summaries (column names, null percentages, min/max/quartiles) are sent for AI inference.
- **Credential Protection**: Database files (`*.db`) and `.streamlit/secrets.toml` are strictly excluded from version control via `.gitignore`.
- **Password Security**: Passwords are saved with dynamic salt generation and high-iteration PBKDF2 hashing.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
