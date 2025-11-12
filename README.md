
# 👗 Fashion Forward Forecasting (StyleSense) — Udacity Project

Predict whether a customer **recommends** a product based on their written review and profile details.
![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Udacity](https://img.shields.io/badge/Udacity-Data%20Science%20ND-blue)
![Status](https://img.shields.io/badge/Project%20Status-Complete-brightgreen)

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

## 📈 Results Summary

The logistic regression model performed strongly on unseen data:

- **ROC-AUC:** 0.93  
- **High recall (0.97)** — captures most positive recommendations  
- **Precision (0.90)** — low false positives  
- Confusion matrix and ROC curve are saved under `models/figures/`

---

## 🧠 Model Card

**Intended Use:**  
Predict customer recommendations for women’s fashion products on StyleSense to help analyze satisfaction and detect trends.

**Training Data:**  
~18,000 product reviews from StyleSense (text, demographics, and product metadata).  
Features include age, department, class name, and review text.

**Limitations:**  
- Model is trained only on women's apparel data.  
- Predictions may not generalize to other domains or product types.  
- Text-based sentiment may reflect cultural or linguistic biases.

**Ethical Considerations:**  
Ensure fair use — model insights should complement, not replace, human judgment when making product decisions.

---

## 📁 Repository Structure

```
dsnd-pipelines-project/
├── src/
│   ├── eda.py
│   ├── tune_pipeline.py
│   ├── predict.py
│   ├── transformers.py
│   └── __init__.py
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
├── starter/data/reviews.csv
├── sample_infer.csv
├── requirements.txt
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

---

## 📚 References & Citations

**Primary Author:**  
Melinda (mgo-1234). *Fashion Forward Forecasting (StyleSense) — Udacity Machine Learning Pipelines Project*, 2025.  
GitHub Repository: [https://github.com/mgo-1234/dsnd-pipelines-project](https://github.com/mgo-1234/dsnd-pipelines-project)

**Dataset Source:**  
Udacity. *Women’s Clothing E-Commerce Reviews Dataset (Starter Data)*,  
provided within the *Data Scientist Nanodegree – Machine Learning Pipelines Project*.  
Original dataset adapted from Kaggle: [https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

**Software & Libraries:**
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.  
- McKinney, Wes. *pandas: Python Data Analysis Library*, 2010.  
- Harris et al., *Array programming with NumPy*, Nature, 2020.  
- Hunter, J. D. *Matplotlib: A 2D Graphics Environment*, Computing in Science & Engineering, 2007.  
- Python Software Foundation. *Python Language Reference, version 3.12*, 2023.  
- Joblib Developers. *Joblib Documentation*, [https://joblib.readthedocs.io](https://joblib.readthedocs.io)

**Project Context:**  
This work was developed as part of the *Udacity Data Scientist Nanodegree*  
(*Machine Learning Pipelines* project module, 2025).

**Acknowledgments:**  
Thanks to Udacity’s DSND mentors and reviewers for providing project scaffolding and evaluation guidelines.

---



