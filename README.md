💳 Credit Card Fraud Detection Dashboard
A futuristic, interactive dashboard for detecting credit card fraud using machine learning.
Built with Streamlit, scikit-learn, and a beautiful neon-glow UI.
🚀 Features
Real-time fraud prediction (Logistic Regression, Random Forest, or Linear Regression)
Interactive EDA: histograms, box plots, scatter plots, correlation heatmaps
Modern, sci-fi dashboard design with neon accents
Responsive and user-friendly interface
📁 Data
Sample Data:
A small sample (creditcard_sample.csv, 1000 rows) is included for quick testing and demo purposes.
Full Dataset:
The full dataset is not included due to size.
You can download it from Kaggle - Credit Card Fraud Detection.
After downloading, place creditcard.csv in the project root directory.
🛠️ How to Run
Install requirements:
Apply to card.py
Run
    streamlit run card.py
(Optional) Train a model:
By default, the app uses a pre-trained model (credit_fraud.pkl).
To retrain, run the provided notebook or script.
Start the app:
Apply to card.py
Run
Open in your browser:
Go to http://localhost:8501
📝 Usage
Use the sidebar to navigate between Introduction, EDA, Model & Prediction, and Conclusion.
Try predictions with custom input or the sample data.
For best results, use the full dataset (see above).
⚠️ Notes
Do not upload the full creditcard.csv to GitHub (it’s too large). Use the sample for sharing and demos.
Add creditcard.csv to your .gitignore file.
📊 Credits
Data: Kaggle - Credit Card Fraud Detection
UI: Inspired by modern sci-fi dashboards
