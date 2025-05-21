# Resume-Screening-App
AI-Powered Resume Screening App categorizing resumes into job roles using NLP (TF-IDF) and ML (KNeighbors, OneVsRest) with a Streamlit interface. Trained on a Kaggle dataset.

# AI-Powered Resume Category Prediction 📄🔍

This project is an AI-driven application designed to automatically classify resumes into predefined job categories. It leverages Natural Language Processing (NLP) techniques for text preprocessing and feature extraction, and Machine Learning models for classification. The application is built with Streamlit, providing an interactive web interface for users to upload resumes and instantly get category predictions.

The development process involved:
1.  **Dataset Analysis & Visualization**: Initial exploration and visualization of a resume dataset from Kaggle (link below) to understand its structure and characteristics.
2.  **Data Preprocessing**: Cleaning the resume text, handling special characters, and preparing it for feature extraction.
3.  **Numerical Encoding**: Using `LabelEncoder` to convert categorical job titles into numerical format suitable for machine learning models.
4.  **Vectorization**: Employing `TfidfVectorizer` to transform the textual data into numerical vectors, capturing the importance of words.
5.  **Model Training**: Training `KNeighborsClassifier` and `OneVsRestClassifier` on the vectorized data to predict resume categories.
6.  **Streamlit App Development**: Building an interactive web application for users to upload resumes (PDF, DOCX, TXT) and receive category predictions.

---

## ✨ Features

* **Resume Parsing**: Extracts text from PDF, DOCX, and TXT file formats.
* **Text Cleaning**: Preprocesses resume text by removing URLs, special characters, and irrelevant patterns.
* **Category Prediction**: Predicts the job category of an uploaded resume using a pre-trained machine learning model.
* **Interactive UI**: User-friendly web interface built with Streamlit.
* **Optional Text Display**: Allows users to view the extracted text from the resume.

---

## 📊 Dataset

The models were trained on a resume dataset sourced from Kaggle.
* **Dataset Link**: [Provide the link to your Kaggle dataset here]

---

## 🛠️ Methodology

1.  **Text Extraction**: Functions are implemented to extract raw text from uploaded resume files (`.pdf`, `.docx`, `.txt`).
2.  **Text Cleaning (`cleanResume`)**:
    * Removes URLs, RT/cc patterns, hashtags, mentions.
    * Strips punctuation and non-ASCII characters.
    * Normalizes whitespace.
3.  **Feature Engineering (`TfidfVectorizer`)**:
    * The cleaned resume text is transformed into a matrix of TF-IDF features. The `tfidf.pkl` file contains the fitted TF-IDF vectorizer.
4.  **Label Encoding (`LabelEncoder`)**:
    * Job categories from the training dataset were encoded into numerical labels. The `encoder.pkl` file contains the fitted label encoder for converting predictions back to category names.
5.  **Model Training & Prediction**:
    * The primary classification model used is a `OneVsRestClassifier` which is `clf.pkl`.
    * The trained model (`clf.pkl`) is loaded to predict the category of new resumes.

---

## 🚀 Technologies Used

* **Python 3.x**
* **Streamlit**: For building the interactive web application.
* **Scikit-learn**: For machine learning tasks (TF-IDF, LabelEncoder, model training/prediction).
* **Pandas & NumPy**: (Implied for data handling during training, though not directly in the app code provided)
* **NLTK**: (Often used for text preprocessing in similar projects, though not explicitly in the app script. Mention if used during training/cleaning logic development).
* **python-docx**: For reading text from `.docx` files.
* **PyPDF2**: For reading text from `.pdf` files.
* **Pickle**: For saving and loading trained models and vectorizers.
* **re (Regular Expressions)**: For text cleaning.

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tahangz/Resume-Screening-App.git](https://github.com/your-username/Resume-Screening-App.git)
    cd Resume-Screening-App
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install streamlit scikit-learn python-docx PyPDF2
    ```

4.  **Ensure you have the pre-trained model and vectorizer files**:
    * `clf.pkl` (your trained classifier model)
    * `tfidf.pkl` (your fitted TF-IDF vectorizer)
    * `encoder.pkl` (your fitted LabelEncoder)
    Place these files in the root directory of the project or update the paths in `app.py` if they are located elsewhere.

---

## ▶️ How to Run the App

Once the setup is complete, run the Streamlit application using the following command in your terminal:

```bash
streamlit run app.py
````

## 📂 File Structure (Simplified)
your-repository-name/
│
├── app.py                # Main Streamlit application script
├── clf.pkl               # Trained classification model
├── tfidf.pkl             # Fitted TF-IDF vectorizer
├── encoder.pkl           # Fitted LabelEncoder
├── requirements.txt      # (Recommended) List of dependencies
└── README.md             # This file
