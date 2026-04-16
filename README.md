# Titanic Survival Analysis

This project analyzes the Titanic dataset from Kaggle to predict passenger survival using data science and machine learning techniques, including Bayesian hyperparameter tuning and ensemble methods.

## Project Overview

The analysis covers the full ML pipeline:
- Data preprocessing and missing value handling
- Exploratory data analysis (EDA) with visualizations
- Feature engineering (`FamilySize`, `AgeGroup`, `FareGroup`)
- Model training with **Bayesian optimization** (`BayesSearchCV`)
- Ensemble prediction using **VotingClassifier**
- Kaggle submission generation

## Files

| File | Description |
|---|---|
| `KaggleTitanicAnalysis.ipynb` | Main notebook — full analysis, modeling, and submission generation |
| `Preprocessing_Colab.ipynb` | Supplementary notebook — compares preprocessing techniques (imputation, encoding, scaling) on Google Colab |
| `train.csv` | Kaggle training dataset |
| `test.csv` | Kaggle test dataset |
| `submission.csv` | Generated predictions for Kaggle submission |

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm scikit-optimize
```

For the Colab preprocessing notebook, additionally:
```bash
pip install missingno
```

## Exploratory Data Analysis (EDA)

- Age distribution histogram and survival analysis by gender (stacked bar chart)
- Survival rate by passenger class (`Pclass`)
- Age distribution by survival status (boxplot)
- Embarkation point vs survival (countplot)
- Correlation heatmap of numerical features

## Feature Engineering

| Feature | Description |
|---|---|
| `FamilySize` | `SibSp + Parch + 1` — total family members aboard |
| `AgeGroup` | Binned into: Child (0–12), Teen (12–18), Young Adult (18–30), Adult (30–50), Senior (50+) |
| `FareGroup` | Binned into: Low (0–10), Medium (10–30), High (30+) |

## Preprocessing

- **Drop** `Cabin` (~77% missing), `PassengerId`, `Ticket`, `Name`
- **Impute** `Age` with median, `Fare` with mean
- **One-hot encode** `Sex`, `Embarked`, `AgeGroup`, `FareGroup` (with `drop="first"`)

The **Colab notebook** additionally demonstrates and compares:
- 3 imputation methods: `SimpleImputer`, `KNNImputer`, `IterativeImputer` (MICE)
- 3 encoding methods: `OneHotEncoder`, `pd.get_dummies`, `LabelEncoder`
- 2 scaling methods: `StandardScaler`, `MinMaxScaler`

## Machine Learning Models

All models are tuned using **Bayesian optimization** (`BayesSearchCV`, 10 iterations, 5-fold cross-validation):

| Model | Tuned Hyperparameters |
|---|---|
| **Logistic Regression** | `C` (regularization, log-uniform 0.01–100) |
| **Random Forest** | `n_estimators` (50–200), `max_depth` (3–20) |
| **Gradient Boosting** | `n_estimators` (50–200), `learning_rate` (0.01–0.3), `max_depth` (2–10) |
| **XGBoost** | `n_estimators` (50–200), `learning_rate` (0.01–0.3), `max_depth` (2–10) |
| **LightGBM** | `n_estimators` (50–200), `learning_rate` (0.01–0.3), `num_leaves` (10–50) |

Final prediction uses a **VotingClassifier** (hard voting) combining all 5 tuned models.

## Model Evaluation

- Each model reports **best parameters** and **mean 5-fold CV accuracy**
- Best and worst models are identified by CV score
- Ensemble CV score is reported for the combined VotingClassifier

## Running the Notebook

1. Install dependencies and open the notebook:

   ```bash
   pip install -r requirements.txt  # or install manually (see Requirements)
   jupyter notebook KaggleTitanicAnalysis.ipynb
   ```

2. Run cells sequentially to reproduce the full pipeline from EDA to submission.

## Results and Findings

- **Gender** and **passenger class** are the strongest predictors of survival
- Feature engineering (`FamilySize`, `AgeGroup`, `FareGroup`) improves model performance
- The **ensemble model** (VotingClassifier) combines strengths of individual classifiers for the final Kaggle submission

## License

This project is for **educational purposes** and follows an **open-source** license.

---

Happy Coding!
