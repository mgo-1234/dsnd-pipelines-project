
# 👗 Fashion Forward Forecasting (StyleSense) — Udacity Project

Predict whether a customer **recommends** a product based on their written review and profile details.

---

## 🪞 Project Overview

**StyleSense** is a rapidly growing online fashion retailer.  
As more customers submit reviews, many forget to mark whether they recommend the product.  
Your task as a data scientist is to **train a machine learning pipeline** that predicts the “recommended” label automatically, using review text, customer demographics, and product metadata.

This model helps StyleSense:
- Gain insights into customer satisfaction  
- Detect trending products earlier  
- Improve personalization and product quality decisions  

---

## 🧩 How to Run the Project

### 1️⃣ Exploratory Data Analysis (EDA)
Generate summary statistics and figures:
```bash
python -m src.eda --csv starter/data/reviews.csv --target "Recommended IND" --text "Review Text"
````

### 2️⃣ Train, Tune & Evaluate the Model

Train the ML pipeline and save the best model + reports:

```bash
python -m src.tune_pipeline --csv starter/data/reviews.csv --target "Recommended IND" --text "Review Text"
```

### 3️⃣ Predict on New Reviews

Use the saved model to generate predictions:

```bash
python -m src.predict --csv sample_infer.csv --out predictions.csv
```

### Example output (`predictions.csv`):

```
Age,Positive Feedback Count,Division Name,Department Name,Class Name,Review Text,predicted_recommend,recommend_proba
29,4,General,Bottoms,Jeans,"Great fit and comfy denim, totally buying another!",1,0.978
52,0,General,Tops,Blouses,"Seams came apart and fabric feels rough.",0,0.217
```

---

## ⚙️ Machine Learning Pipeline

| Data Type       | Processing Steps                                                                                         | Description                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Numeric**     | `SimpleImputer(strategy="median")` → `StandardScaler(with_mean=False)`                                   | Handles missing values and scales numeric inputs.                                            |
| **Categorical** | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=True)` | Encodes product metadata and handles unknown categories.                                     |
| **Text**        | `FunctionTransformer(to_1d_str)` → `TfidfVectorizer`                                                     | Converts review text to TF-IDF features (tuned n-grams and max_features; English stopwords). |
| **Model**       | `LogisticRegression`                                                                                     | Tuned via GridSearchCV (`C`, `solver`, `class_weight`).                                      |

All steps are combined in a **single scikit-learn Pipeline**, ensuring identical preprocessing during training and inference.

---

## 🎯 Model Performance (Test Set)

| Metric        |  Score |
| ------------- | -----: |
| **Accuracy**  | 0.8878 |
| **Precision** | 0.9024 |
| **Recall**    | 0.9671 |
| **F1-Score**  | 0.9336 |
| **ROC-AUC**   | 0.9268 |

Visuals:

* 📊 `models/figures/confusion_matrix.png`
* 📈 `models/figures/roc_curve.png`

---

## 📁 Repository Structure

```
dsnd-pipelines-project/
│
├── src/
│   ├── eda.py               # EDA report and data summaries
│   ├── tune_pipeline.py     # Pipeline, GridSearchCV, evaluation, figures
│   ├── predict.py           # Load saved model and predict on new data
│   ├── transformers.py      # Shared transformer utilities (pickle-safe)
│
├── models/
│   ├── best_pipeline.joblib
│   ├── best_params.json
│   ├── test_report.json
│   ├── cv_results.csv
│   └── figures/
│       ├── confusion_matrix.png
│       └── roc_curve.png
│
├── starter/data/reviews.csv # Dataset (Udacity)
├── sample_infer.csv         # Example for inference
└── README.md
```

---

## 🧪 Techniques Used

* **scikit-learn Pipelines** to unify preprocessing and modeling
* **TF-IDF** (with n-grams) for NLP feature extraction
* **Hyperparameter tuning** via `GridSearchCV` + stratified CV
* **Held-out evaluation** with Accuracy, Precision, Recall, F1, ROC-AUC
* **Joblib serialization** for reproducible deployment

---

## 🚧 Possible Improvements

* Add **spaCy POS features** (e.g., adjective/verb ratios, exclamation count)
* Try **ensembles** (Random Forest, XGBoost) and compare via the same pipeline
* Build a **Streamlit** dashboard for interactive exploration and predictions

---

## 👩‍💻 Author

**Melinda (mgo-1234)**
Udacity Data Scientist Nanodegree — *Machine Learning Pipelines Project*

---

## ✅ Summary

This project meets the rubric by:

* Using a single, modular **Pipeline** that handles numeric, categorical, and text data
* Applying **proper preprocessing** (imputation, scaling, OHE, TF-IDF)
* Performing **hyperparameter tuning** and rigorous **evaluation** on a test set
* Saving artifacts for **reproducible inference** (`best_pipeline.joblib`)

```
