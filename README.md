# PredictiX-Disease-Prediction-WebAppp


PredictiX is an intelligent, user-friendly web application that leverages the power of machine learning to provide preliminary predictions for four major chronic diseases: Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer. This tool is designed to serve as a health awareness platform, empowering users to assess their risk based on relevant medical parameters and encouraging them to seek timely professional medical advice.

The application features a stunning, modern user interface with a "Cyberpunk Neon" theme, glassmorphism effects, and interactive visualizations, all built with Streamlit.

📖 About The Project
The early detection of chronic diseases can be life-saving. PredictiX was created to make preliminary health assessment more accessible to everyone. By consolidating multiple machine learning models into a single, intuitive platform, this project aims to:

Promote Health Awareness: Provide users with a simple tool to understand potential health risks based on their data. Leverage AI for Good: Showcase a practical application of machine learning in the healthcare domain. Encourage Proactive Care: Motivate users who receive a high-risk prediction to consult with healthcare professionals for a formal diagnosis.

The application features a clean, modern interface built with Streamlit, ensuring a seamless user experience across different devices.

✨ Key Features
Four Prediction Modules: Separate, dedicated pages for Diabetes, Heart Disease, Parkinson's, and Breast Cancer prediction.

Multi-Model Approach: Each disease prediction is supported by multiple machine learning models (e.g., Logistic Regression, SVM, Random Forest, XGBoost) to provide a more robust assessment.

Stunning & Modern UI: A sleek, "Cyberpunk Neon" dark-themed interface with glassmorphism cards, animated gradient buttons, and a clean layout.

Interactive Data Visualization: A dedicated dashboard to explore the underlying datasets using beautifully styled charts created with Seaborn.

Responsive Design: The application is fully functional on both desktop and mobile browsers.

🚀 Features
📊 Predict diseases using ML models
🔎 Interactive UI with dark theme and navigation menu
📈 Visualize feature distributions and correlations
💾 Supports local deployment
🧰 Tech Stack
This project is built with a modern Python technology stack:

Python: The core programming language.

Streamlit: The open-source framework for building and deploying the web application.

Pandas: For data manipulation and analysis.

Scikit-learn: For building and evaluating machine learning models.

XGBoost: For implementing the high-performance gradient boosting models.

Seaborn & Matplotlib: For creating beautiful, static data visualizations.

Pickle: For serializing and saving the trained models.

📂 Project Structure
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
🛠️ How to Run the Project Locally
✅ Prerequisites
Python 3.7+
Git You can download it from git-scm.com.
🔄 Steps
Clone the Repository

git clone https://github.com/your-username/PredictiX-Disease-Prediction-WebApp.git
cd PredictiX-Disease-Prediction-WebApp
Create a Virtual Environment

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
Install Dependencies

pip install -r requirements.txt
Run the Streamlit App

streamlit run main_app.py
Visit in your browser

http://localhost:8501
📷 Screenshots
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/af655fbb-4243-473a-9bd8-71b595e7ce17" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/3ed41fdb-6674-4343-b2e9-73db97191f69" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/7e2f0bf4-5a2d-4927-b8bf-da4d8d2770e2" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/f99e3ddd-f71e-45eb-8db0-bda66819a6dc" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/9545f00c-cece-4e8b-9961-fb861541c46c" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/3ae9a9f3-4d36-4050-a968-a1207d813bff" />

📌 Datasets Used
Diabetes: PIMA Indians Diabetes Dataset
Heart: UCI Heart Disease Dataset
Parkinson’s: UCI Parkinson's Dataset
Breast Cancer: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
🧠 ML Models
Each disease prediction is powered by a pre-trained machine learning model saved as .pkl files using joblib:

Logistic Regression / Random Forest / SVM (per disease)
