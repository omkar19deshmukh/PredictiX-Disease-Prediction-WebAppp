# 🩺 PredictiX - Multi-Disease Prediction System

**Author**: Akash Mandal  
**GitHub**: [Akashmandal08](https://github.com/Akashmandal08)  
**Email**: akashmandal.9490@gmail.com  
**Repository**: [PredictiX-Disease-Prediction-WebApp](https://github.com/Akashmandal08/PredictiX-Disease-Prediction-WebApp.git)

---

PredictiX is an intelligent, user-friendly web application that leverages the power of machine learning to provide preliminary predictions for four major chronic diseases: Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer. This tool is designed to serve as a health awareness platform, empowering users to assess their risk based on relevant medical parameters and encouraging them to seek timely professional medical advice.

The application features a stunning, modern user interface with a "Cyberpunk Neon" theme, glassmorphism effects, and interactive visualizations, all built with Streamlit.

---
### 📖 About The Project
The early detection of chronic diseases can be life-saving. PredictiX was created to make preliminary health assessment more accessible to everyone. By consolidating multiple machine learning models into a single, intuitive platform, this project aims to:

Promote Health Awareness: Provide users with a simple tool to understand potential health risks based on their data.
Leverage AI for Good: Showcase a practical application of machine learning in the healthcare domain.
Encourage Proactive Care: Motivate users who receive a high-risk prediction to consult with healthcare professionals for a formal diagnosis.

The application features a clean, modern interface built with Streamlit, ensuring a seamless user experience across different devices.

---

### ✨ Key Features
Four Prediction Modules: Separate, dedicated pages for Diabetes, Heart Disease, Parkinson's, and Breast Cancer prediction.

- Multi-Model Approach: Each disease prediction is supported by multiple machine learning models (e.g., Logistic Regression, SVM, Random Forest, XGBoost) to provide a more robust assessment.

- Stunning & Modern UI: A sleek, "Cyberpunk Neon" dark-themed interface with glassmorphism cards, animated gradient buttons, and a clean layout.

- Interactive Data Visualization: A dedicated dashboard to explore the underlying datasets using beautifully styled charts created with Seaborn.

- Responsive Design: The application is fully functional on both desktop and mobile browsers.



## 🚀 Features

- 📊 Predict diseases using ML models
- 🔎 Interactive UI with dark theme and navigation menu
- 📈 Visualize feature distributions and correlations
- 💾 Supports local deployment

---

## 🧰 Tech Stack

This project is built with a modern Python technology stack:

- Python: The core programming language.

- Streamlit: The open-source framework for building and deploying the web application.

- Pandas: For data manipulation and analysis.

- Scikit-learn: For building and evaluating machine learning models.

- XGBoost: For implementing the high-performance gradient boosting models.

- Seaborn & Matplotlib: For creating beautiful, static data visualizations.

- Pickle: For serializing and saving the trained models.

---

## 📂 Project Structure

```
PredictiX/
│
├── diabetes_model.pkl
├── heart_model.pkl
├── parkinsons_model.pkl
├── breast_cancer_model.pkl
├── app.py
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── parkinsons.csv
│   └── breast_cancer.csv
└── requirements.txt
```

---

## 🛠️ How to Run the Project Locally

### ✅ Prerequisites

- Python 3.7+
- Git You can download it from git-scm.com.

### 🔄 Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/PredictiX-Disease-Prediction-WebApp.git
   cd PredictiX-Disease-Prediction-WebApp
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**
   ```bash
   streamlit run main_app.py
   ```

5. **Visit in your browser**
   ```
   http://localhost:8501
   ```

---

## 📷 Screenshots



> Diabetes<img width="1920" height="1020" alt="Screenshot 2025-08-03 114333" src="https://github.com/user-attachments/assets/0cf3eb30-9f21-447c-b669-b66cfd5e383c" />

---
> Heart Disease<img width="1920" height="1020" alt="Screenshot 2025-08-03 114350" src="https://github.com/user-attachments/assets/966c4300-b007-4b86-9c1a-ed1651ff6845" />

---
>  Parkinson’s<img width="1920" height="1020" alt="Screenshot 2025-08-03 114402" src="https://github.com/user-attachments/assets/a04606b3-84c3-4daa-a546-e012e8a3d9e2" />

---
>  Breast Cancer Prediction<img width="1920" height="1020" alt="Screenshot 2025-08-03 114413" src="https://github.com/user-attachments/assets/cadbfab9-7675-4cd4-8a82-b504d1aafc8b" />

---
>  Data Visualization dashboards<img width="1920" height="1020" alt="Screenshot 2025-08-03 114425" src="https://github.com/user-attachments/assets/06b6ac66-5229-4111-b35a-bbff489cfe6c" />

---
<img width="1920" height="1020" alt="Screenshot 2025-08-03 114436" src="https://github.com/user-attachments/assets/f6cedf9f-dc2c-45a7-8537-0bc96ece99a1" />

---

## 📌 Datasets Used

- **Diabetes**: PIMA Indians Diabetes Dataset
- **Heart**: UCI Heart Disease Dataset
- **Parkinson’s**: UCI Parkinson's Dataset
- **Breast Cancer**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

---

## 🧠 ML Models

Each disease prediction is powered by a pre-trained machine learning model saved as `.pkl` files using joblib:
- Logistic Regression / Random Forest / SVM (per disease)

---

## ✨ Author

Made with ❤️ by [AKASH MANDAL]  
GitHub: [Akashmandal08](https://github.com/Akashmandal11)

---

## 📃 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

