# 🧠 AI Tools Assignment

### Group Members
- **Joseph Muthui**
- **Racheal Wanjiru**
- **Member 3**

---

## 🎯 Objective
This project demonstrates practical knowledge of **AI tools, frameworks, and ethical considerations** through both theoretical and hands-on tasks.  
It integrates machine learning, deep learning, and NLP frameworks — showing how they can be applied to solve real-world problems creatively.

---

## 📁 Project Structure

ai-tools-assignment/
│
├── notebooks/
│ ├── task1_iris_decision_tree.py # Decision Tree Classifier (Scikit-learn)
│ ├── task2_text_processing.ipynb # Text cleaning, tokenization, WordCloud
│ └── task3_spacy_ner.py # NER & sentiment with spaCy
│
├── scripts/
│ └── data_cleaner.py # Reusable data cleaning function
│
├── streamlit_app/
│ └── app.py # Creative web app for sentiment testing
│
├── report/
│ └── AI_Tools_Report.md # Written summary (used to generate PDF)
│
├── presentation/
│ └── slides_outline.txt # Slide plan for 3-min group presentation
│
├── assets/
│ ├── screenshots/ # Output images for report
│ └── sample_reviews.txt # Sample text data for NLP
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🧩 Tasks Overview

### **Part 1 — Theoretical Understanding**
- Explained key differences between **TensorFlow**, **PyTorch**, **Scikit-learn**, and **spaCy**.
- Compared frameworks based on **use case, ease of use**, and **community support**.

---

### **Part 2 — Practical Implementation**

#### 🟢 Task 1: Classical ML (Scikit-learn)
- Dataset: **Iris Species**
- Model: Decision Tree Classifier
- Metrics: Accuracy, Precision, Recall  
- Output: Confusion matrix visualization and prediction summary.

#### 🟡 Task 2: Deep Learning (TensorFlow/Keras)
- Dataset: **MNIST Handwritten Digits**
- Model: CNN (Convolutional Neural Network)
- Goal: >95% test accuracy  
- Output: Accuracy plot and sample digit predictions.

Certainly! Here's the updated note including instructions to unzip the dataset file before running Task 3:

---

### Task 3: NLP Review Analysis with spaCy

**Goal:**  
Perform Named Entity Recognition (NER) to extract product names, brands, and organizations, and conduct rule-based sentiment analysis on sample Amazon product reviews.

**Prerequisites:**  
- Install spaCy and download the English model:
  ```bash
  pip install spacy
  python -m spacy download en_core_web_sm
  ```

**Dataset:**  
Uses internal sample data for quick testing. For full-scale use, download the Amazon Reviews dataset from Kaggle:  
[Amazon Reviews Dataset on Kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews)  

**Note:** After downloading, unzip the file to extract `test.ft.txt`, which is required to run the Task 3 code.

---

Let me know if you want me to add this directly into your README file!

---

### **Bonus Task — Streamlit App**
- Created an interactive **Streamlit web app** for live text sentiment prediction.
- Users can input text and see instant **Positive/Negative** feedback.
- Designed with emojis and color-coded sentiment bars for creativity 🎨

To run locally:
```bash
cd streamlit_app
streamlit run app.py
Part 3 — Ethics & Optimization
Reflected on bias in datasets like MNIST and user reviews.

Discussed how tools like TensorFlow Fairness Indicators and spaCy rule-based filters can reduce discrimination.

Debugged and optimized TensorFlow code with clear documentation.

⚙️ Installation
Clone the repo and install all dependencies:

bash
Copy code
git clone https://github.com/<your-username>/ai-tools-assignment.git
cd ai-tools-assignment
pip install -r requirements.txt
Run Jupyter Notebooks:

bash
Copy code
jupyter notebook
📊 Tools & Frameworks
Category	Tools Used
Classical ML	Scikit-learn
Deep Learning	TensorFlow / Keras
NLP	spaCy, NLTK, TextBlob
Visualization	Matplotlib, WordCloud
Deployment	Streamlit
Documentation	Markdown, PDF

📸 Sample Outputs
Iris Decision Tree Confusion Matrix ✅

MNIST Accuracy Curve 📈

NER Entity Visualization 🧾

Streamlit App Screenshot 💬

(All screenshots are stored in assets/screenshots/.)

🧭 Reflection
This project improved our understanding of:

How AI frameworks differ in scope and performance.

The importance of clean data pipelines and model interpretability.

The ethical side of AI — addressing bias and ensuring fairness.

🏁 Final Deliverables
Item	Location
PDF Report	report/AI_Tools_Report.md (convert to PDF)
GitHub Repository	Entire folder
Presentation Slides	presentation/slides_outline.txt
Streamlit Demo	streamlit_app/app.py

🧰 Requirements

See requirements.txt for all dependencies.

👨‍💻 Author

Joseph Muthui Nzamba
AI Tools & NLP Assignment
Multimedia University of Kenya
