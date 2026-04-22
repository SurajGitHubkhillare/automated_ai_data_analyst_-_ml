# 🤖 Multi-Agent AI Data Analyst System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![PyCaret](https://img.shields.io/badge/PyCaret-Blue?style=for-the-badge)](https://pycaret.org/)
[![PandasAI](https://img.shields.io/badge/PandasAI-Green?style=for-the-badge)](https://github.com/gventuri/pandas-ai)

A professional-grade, end-to-end **AI Data Science Platform** that automates the entire machine learning lifecycle. From secure authentication and interactive data cleaning to advanced AutoML and natural language data querying, this system acts as a virtual data scientist for your datasets.

---

## ✨ Key Features

### 🔐 1. Secure Access & Admin Control
- **Role-Based Authentication:** Secure login/signup system using `BCrypt` and `SQLite`.
- **Admin Dashboard:** Manage users, monitor login history, and export security logs.

### 🧹 2. Interactive Data Preparation
- **Smart Cleaning:** Automated numerical and categorical imputation (Mean, Median, Mode).
- **Outlier Removal:** Z-score based outlier detection and filtering.
- **Normalization:** Robust feature scaling (Z-Score, Min-Max, Robust Scaling).
- **Transformation Report:** Real-time feedback on data loss and statistical changes.

### 📊 3. Automated Insight Generation
- **Auto EDA:** Generates comprehensive, interactive profiling reports via `ydata-profiling`.
- **Advanced Visualizations:** Correlation heatmaps, distribution plots, and missing value analysis.

### 🚀 4. Automated Machine Learning (AutoML)
- **Model Comparison:** Automatically trains and evaluates 15+ algorithms for Classification or Regression.
- **Hyperparameter Optimization:** Bayesian Search-based model tuning.
- **Ensemble Learning:** Native support for Bagging, Boosting, and Model Stacking.
- **Interpretability:** Integrated SHAP (SHapley Additive exPlanations) for "Black Box" model interpretation.

### 💬 5. Chat with Data (GenAI)
- **Natural Language Querying:** Ask questions like *"What is the average age of customers who churned?"* 
- **LLM Integration:** Powered by **Google Gemini** via `PandasAI` to translate English into executable Pandas code.

### 🔮 6. Production Ready
- **Anomaly Detection:** Unsupervised outlier detection using Isolation Forests and KNN.
- **Prediction Pipeline:** Real-time manual input or batch CSV prediction modes.
- **Deployment Exports:** Generate standalone training scripts and **FastAPI** wrappers with one click.

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit (Custom CSS & Glassmorphism UI)
- **Core ML:** PyCaret, Scikit-Learn
- **Generative AI:** Google Gemini, PandasAI
- **Data Engineering:** Pandas, Numpy
- **Security:** SQLite3, BCrypt
- **Reporting:** YData-Profiling, SHAP

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/ai-data-analyst.git
   cd ai-data-analyst
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure
- `app.py`: Main entry point and sidebar routing.
- `auth.py`: SQLite-based authentication & logging logic.
- `components.py`: Core logic for AutoML, EDA, and Chat interfaces.
- `style.css`: Premium custom styling for the Streamlit UI.
- `app_data.db`: Local database for user credentials and logs.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
