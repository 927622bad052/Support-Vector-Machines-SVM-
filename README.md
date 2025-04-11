# 🔍 SVM Project – Iris Classifier

This repository demonstrates a simple yet powerful application of **Support Vector Machines (SVM)** using Python and `scikit-learn`. The project uses the classic **Iris dataset** to train and evaluate an SVM classifier.

---

## 🧠 Overview

- 📊 Learn how SVMs work on multiclass classification  
- 🧪 Use `scikit-learn` to train, test, and evaluate the model  
- 📈 Visualize decision boundaries and results  
- 📁 Clean modular structure with reusable utility functions

---

## 📁 Project Structure

svm-project/
├── README.md               # Project overview
├── LICENSE                 # License information
├── requirements.txt        # Required libraries
├── .gitignore              # Files to ignore in Git
├── data/                   # (Optional) Datasets
├── notebooks/              # (Optional) Jupyter notebooks
└── src/
├── main.py             # Train & evaluate SVM
└── svm_utils.py        # Utility functions (e.g., plots)

---

## ⚙️ Getting Started

### ✅ Prerequisites

- Python 3.7 or higher  
- `pip` package manager

### 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/svm-project.git
cd svm-project

# 2. Install dependencies
pip install -r requirements.txt
🚀 Run the Project
python src/main.py

🧪 Dataset Used
	•	Iris Dataset from sklearn.datasets
	•	3 classes: Setosa, Versicolor, Virginica
	•	4 features: sepal length, sepal width, petal length, petal width

📦 requirements.txt
scikit-learn
matplotlib
pandas
numpy

📈 Example Output

Training Accuracy: 97.78%
Test Accuracy: 96.67%

Classification Report:
              precision    recall  f1-score   support
     Setosa       1.00      1.00      1.00        10
 Versicolor       1.00      0.90      0.95        10
  Virginica       0.91      1.00      0.95        10

👨‍💻 Author

Made with ❤️ by Shiva
🎓 B.Tech AI & Data Science @ MKCE
🛠️ Python | ML | GitHub Projects | Always building

🌟 Like this project?

Leave a ⭐ on the repo and follow for more ML projects!
---

Let me know if you want the **code files** (`main.py`, `svm_utils.py`) or even a **Jupyter notebook** version of this SVM project next!
