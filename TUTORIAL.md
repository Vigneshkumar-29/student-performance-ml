# Student Performance Prediction System: The Complete Step-by-Step Tutorial

This tutorial teaches you Machine Learning (ML) by building one project from raw data to a working web app.
Every step follows the same pattern: **Concept → Why → Code → Code explained → Expected output → Checklist**.

> **How to use this:** the complete, tested code is already in this folder. Read a step, open the file it mentions, run the command, compare your output with "Expected output", tick the checklist, then move on. Small differences in decimals are normal.

---

## Step 1: Project Overview

### What the project does
It takes information about a student (study hours, attendance, past scores, and so on) and predicts whether the student will **Pass** or **Fail**, along with a probability such as "87% chance of passing".

### Why Machine Learning?
You could write rules by hand: *"if attendance < 75 then fail"*. But real results depend on many factors interacting. Is a student with 90% attendance but two past failures safer than one with 70% attendance and perfect scores? Hand-written rules get messy fast. In ML we show the computer **many past examples** (inputs and the known outcome) and let it **discover the pattern itself**.

### What type of ML problem is this?
- **Supervised learning**: our past examples come with the correct answer (Pass/Fail), like a student studying from a worked answer sheet.
- **Classification**: the answer is a *category* (Pass or Fail), not a number. (Predicting an exact mark like 73.5 would be *regression*.)
- More precisely, **binary classification**, because there are exactly two categories.

### Input and output
| | |
|---|---|
| **Input (features)** | study hours, attendance %, previous score, assignment score, internal score, sleep hours, class participation, previous failed subjects |
| **Output (target)** | Pass (1) or Fail (0), plus the probability of passing |

### What you will learn
Reading and cleaning messy data · exploring data with charts · splitting data fairly · training three algorithms · judging a model honestly (accuracy, precision, recall, F1, confusion matrix) · avoiding data leakage · saving a model · building a web app around it.

**✅ Checklist:** you can explain in one sentence why this is "supervised binary classification".

---

## Step 2: Project Folder Structure

### Concept and why
A clean structure keeps the project understandable. A rule of thumb: *data, code, models and the app each live in their own place.*

```
student-performance-ml/
│
├── data/                    # CSV files
│   ├── students_raw.csv         messy data, exactly as "collected"
│   └── students_clean.csv       the same data after cleaning
├── notebooks/               # Jupyter notebook for learning and experiments
│   └── student_performance_walkthrough.ipynb
├── models/                  # Saved trained model + comparison table
├── reports/figures/         # Charts saved by the code
├── src/                     # All reusable Python code ("source")
│   ├── __init__.py              makes src a package so we can write `from src.config import ...`
│   ├── config.py                paths, column names, valid ranges (one place to change them)
│   ├── generate_data.py         creates the practice dataset
│   ├── data_cleaning.py         fixes problems in the data
│   ├── features.py              feature engineering + the full preprocessing Pipeline
│   ├── eda.py                   exploration + charts
│   └── train.py                 split, train, evaluate, save
├── app.py                   # Streamlit web app
├── requirements.txt         # Libraries to install
├── TUTORIAL.md              # This file
└── README.md                # Project front page
```

**Why `src/`?** The training script and the web app both need the same cleaning and preprocessing code. Putting it in one place means we never copy-paste it (copies drift apart and cause bugs).

**✅ Checklist:** you can say which folder holds data, which holds code, and which holds the trained model.

---

## Step 3: The Dataset

### Concept
A **dataset** is a table. Each **row** is one example (one student). Each **column** is a piece of information about it.

- **Features**: the columns the model uses as input (also called *independent variables* or *X*).
- **Target**: the column we want to predict (*dependent variable* or *y*). Here it is `result`.

### Public dataset or our own?
A well-known public option is the **UCI "Student Performance"** dataset (Portuguese secondary schools). It does not contain our exact columns (for example, "internal assessment score" or "class participation"), so using it would force us to change the project. Instead, **we generate a realistic dataset** with `src/generate_data.py`. This also lets us plant realistic problems on purpose so you can practise fixing them. Once you finish this project, trying the UCI dataset is a great next challenge.

### How our data is generated (so you know it is not magic)
1. Each student gets a hidden "ability" level. That is why a student with high attendance also tends to have higher scores.
2. All columns are created from ability plus random noise, and clipped to realistic ranges.
3. A formula turns the features into a *probability* of passing, and then a random draw decides the actual result. Randomness matters: real life has surprises, so no model can reach 100%.
4. `make_messy()` then damages the data on purpose.

### Run it
```bash
python -m src.generate_data
```
**Expected output:** `Saved 1525 rows to .../data/students_raw.csv`

### Column guide
| Column | Type | Meaning |
|---|---|---|
| `student_id` | identifier | just a label, **not** a feature |
| `study_hours` | numerical | hours per day |
| `attendance` | numerical | percent, 0 to 100 |
| `previous_score` | numerical | 0 to 100 |
| `assignment_score` | numerical | 0 to 100 |
| `internal_score` | numerical | 0 to 100 |
| `sleep_hours` | numerical | hours per night |
| `previous_failed_subjects` | numerical (count) | 0, 1, 2, … |
| `class_participation` | **categorical** | Low / Medium / High |
| `result` | **target** (categorical) | Pass / Fail |

### Problems planted in the data
| Problem | Count | What it looks like |
|---|---|---|
| **Missing values** | about 50 in each feature column | empty cells (NaN = "Not a Number") |
| **Duplicate rows** | 25 | the same student entered twice |
| **Incorrect values** | 20 | attendance of 120%, study hours of -3, sleep of 24 h, internal score of 150 |
| **Inconsistent text** | 40 | `high`, `  Medium `, `LOW` instead of `High`, `Medium`, `Low` |

**Why `student_id` is not a feature:** it is just a counter. If a model used it, it would be learning noise (or worse, accidental patterns) instead of real signals.

**✅ Checklist:** `data/students_raw.csv` exists. You can name the target column and one numerical and one categorical column.

---

## Step 4: Data Exploration

### Concept
Before building anything, **look at your data**. Exploration answers: how big is it, what types are the columns, what is missing, what looks strange?

### Code
Run the ready-made script (it prints everything below and saves two charts):
```bash
python -m src.eda
```
The key pandas commands (also in `notebooks/` so you can run them one by one):
```python
import pandas as pd
df = pd.read_csv("data/students_raw.csv")

df.head()           # first 5 rows
df.shape            # (rows, columns)
df.dtypes           # data type of each column
df.isnull().sum()   # missing values per column
df.duplicated().sum()   # number of exact duplicate rows
df.describe()       # count, mean, std, min, quartiles, max
df["result"].value_counts()   # how many Pass vs Fail
```

### Code explained
- `isnull().sum()`: `isnull()` gives True/False for every cell; `sum()` counts the Trues per column.
- `describe()`: **always check `min` and `max`.** That is how you spot impossible values.
- `value_counts()`: counts each category, so we can see if the target is balanced.

### Expected output (key parts)
```
Shape (rows, columns): (1525, 10)
Duplicate rows: 25
Missing values: about 50-56 in each feature column, 0 in student_id and result
study_hours  min = -3.0      <- impossible!
sleep_hours  max = 24.0      <- impossible!
Pass    866
Fail    659
```
Target split is about **57% Pass / 43% Fail**: reasonably balanced. If it were 95/5, plain accuracy would be misleading (a model that always says "Pass" would score 95%).

### Visualizations (saved in `reports/figures/`)
1. **`01_missing_values.png`** (bar chart of missing values per column). *Tells us:* the gaps are spread evenly across columns (about 3.5% each), so no single column is badly broken. Evenly spread, random-looking gaps are the easy case.
2. **`02_feature_histograms.png`** (a histogram per numeric column). *A histogram shows how many students fall into each value range.* It reveals the shape of each feature (most are bell-shaped) and shows the odd values: bars far away from the crowd (for example at 24 hours of sleep) are errors.

**✅ Checklist:** you know the dataset has 1,525 rows, 25 duplicates, missing values, and impossible values in 4 columns.

---

## Step 5: Data Cleaning

### Concept
**"Garbage in, garbage out."** A model learns whatever is in the data, including mistakes. Cleaning fixes the data before learning.

### Why each step is needed
| Step | Why it is required |
|---|---|
| **Remove duplicates** | A duplicated student counts twice, which biases the model. Worse, a duplicate can land in *both* train and test sets, so the test would be partly "memorised" and give a falsely good score. |
| **Convert data types** | Models need numbers. `pd.to_numeric(..., errors="coerce")` turns anything unreadable (like the text "N/A") into a proper missing value. The target `Pass/Fail` becomes `1/0`. |
| **Fix inconsistent text** | To a computer, `"high"` and `"High"` are different categories. `str.strip().str.title()` makes them match. |
| **Incorrect values** | An attendance of 120% would mislead the model (and distort scaling). We mark such values as missing instead of deleting the whole row, so the student's other columns still help. |
| **Missing values** | Most models crash on NaN. We fill them in **later, inside the Pipeline** (see below). |

### Important: why we do NOT fill missing values here
Filling gaps with the **median** needs a calculation *over the data*. If we computed it over *all* rows, then information from the future test set would sneak into training. This mistake is called **data leakage**, and it makes results look better than they will be in real life. So:

- **Rules that learn nothing** (drop duplicates, "score > 100 is wrong"): safe to do before the split → `data_cleaning.py`.
- **Steps that learn from data** (median, scaling): only after the split, fitted on training data → inside the Pipeline in `features.py`.

### Code
Open `src/data_cleaning.py`. The heart of it:
```python
df = df.drop_duplicates()
df = df.drop_duplicates(subset="student_id")

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[col] = df[col].astype("string").str.strip().str.title()      # "  Medium " -> "Medium"
df.loc[~df[col].isin(PARTICIPATION_ORDER), col] = np.nan        # unknown label -> missing

for col, (low, high) in VALID_RANGES.items():
    wrong = (df[col] < low) | (df[col] > high)
    df.loc[wrong, col] = np.nan                                  # impossible -> missing

df[TARGET] = df[TARGET].map({"Pass": 1, "Fail": 0})
```
`VALID_RANGES` lives in `config.py`, so changing a rule means editing one line.

Run it:
```bash
python -m src.data_cleaning
```

### Expected output
```
Duplicates removed: 25
Incorrect values in 'study_hours' set to missing: 5
Incorrect values in 'attendance' set to missing: 6
Incorrect values in 'internal_score' set to missing: 5
Incorrect values in 'sleep_hours' set to missing: 4
Saved 1500 clean rows to .../data/students_clean.csv
```
It also lists the missing values still remaining (about 50-58 per column). These are intentionally left for the Pipeline.

**✅ Checklist:** 1,500 rows remain; `students_clean.csv` exists; you can explain why median-filling is *not* done here.

---

## Step 6: Exploratory Data Analysis (EDA)

### Concept
EDA means asking **"which features seem related to passing?"** using charts and summary numbers. It guides which features to keep and gives you sanity checks on the model later.

Run: `python -m src.eda`. Charts are saved in `reports/figures/`. (Plots use the *clean* data.)

### Pass / Fail distribution: `03_pass_fail_distribution.png`
A bar chart of how many students passed vs failed. **Insight:** 57% pass, so the classes are fairly balanced; accuracy is a fair metric here, though we will still look at precision/recall.

### Study hours vs result: `04_study_hours_vs_result.png`
A **box plot**: the box covers the middle 50% of students, the line inside is the median, the "whiskers" show the typical range, and dots are unusual values.
**Insight:** the median study time of passing students is about **4.6 h/day** vs **3.1 h/day** for failing students. More study goes with passing, but the boxes overlap, so study time alone does not decide the result.

### Attendance vs result: `05_attendance_vs_result.png`
**Insight:** the median attendance is about **82%** for passers vs **72%** for failers. Failing students cluster near and below the common 75% requirement, which is why we later build a `low_attendance` feature.

### Previous score vs result: `06_previous_score_vs_result.png`
**Insight:** the median is about **70** for passers vs **52** for failers. Past performance is among the strongest signals, which is academically sensible: results are persistent.

### Participation: `07_participation_vs_pass_rate.png`
**Insight:** pass rate is about **30%** (Low), **61%** (Medium), **83%** (High). A large, clean effect.

### Failed subjects: `08_failed_subjects_vs_pass_rate.png`
**Insight:** pass rate falls from **77%** (no failures) to **58%** (1), **31%** (2), **17%** (3), and about 12% (4). *Academic insight:* a student with 2+ past failures is the clearest "needs help now" group.

### Correlation heatmap: `09_correlation_heatmap.png`
**Correlation** is a number from -1 to +1: +1 means "rise together", -1 means "one rises as the other falls", 0 means no straight-line relationship. The last row shows each feature against `result`:

| Feature | Correlation with passing |
|---|---|
| previous_score | +0.56 |
| internal_score | +0.56 |
| assignment_score | +0.52 |
| attendance | +0.48 |
| study_hours | +0.40 |
| sleep_hours | +0.04 |
| previous_failed_subjects | -0.43 |

**Insights:**
1. Academic scores and attendance are the strongest signals; failed subjects push the other way.
2. **`sleep_hours` is almost unrelated** to passing in this dataset. We keep it (the project brief asks for it, and a model can ignore weak features), but a real project might drop it. *Do not assume every feature matters. Check.*
3. The three scores correlate with each other at about 0.6 (they carry overlapping information). This matters when reading Logistic Regression coefficients (Step 11).
4. Correlation measures **straight-line association, not cause and effect.**

**✅ Checklist:** you can name the 3 strongest predictors and the one weak one.

---

## Step 7: Feature Engineering

### Concept
**Feature engineering** means creating new columns from existing ones to give the model information in a more useful form. Think of it as summarising: instead of three separate scores, you might also give the model "the average of the three".

### Which columns are used?
- **Target:** `result` (1 = Pass, 0 = Fail)
- **Original features (8):** `study_hours`, `attendance`, `previous_score`, `assignment_score`, `internal_score`, `sleep_hours`, `previous_failed_subjects`, `class_participation`
- **Not used:** `student_id` (a label, carries no real information)
- **Engineered features (3), created in `add_features()` in `src/features.py`:**

| New feature | How | Why it might help |
|---|---|---|
| `avg_academic_score` | mean of previous, assignment and internal scores | one summary of overall academic ability; averaging also reduces noise from any single score |
| `has_failed_before` | 1 if `previous_failed_subjects > 0` else 0 | "has ever failed" may matter more than the exact count |
| `low_attendance` | 1 if attendance < 75 else 0 | many schools have a 75% rule, so crossing it is a meaningful event |

### Why the original features were selected
Each one is something a teacher could actually know before the exam, and each has a plausible academic link to results (practice, presence, past performance, wellbeing, engagement). We exclude anything that would be unknown at prediction time. Including such a column would be another kind of data leakage.

### Code
```python
def add_features(df):
    df = df.copy()
    df["avg_academic_score"] = df[["previous_score", "assignment_score", "internal_score"]].mean(axis=1)
    df["has_failed_before"] = (df["previous_failed_subjects"] > 0).astype(float)
    df.loc[df["previous_failed_subjects"].isna(), "has_failed_before"] = np.nan   # unknown stays unknown
    df["low_attendance"] = (df["attendance"] < 75).astype(float)
    df.loc[df["attendance"].isna(), "low_attendance"] = np.nan
    return df
```
- `df.copy()`: never modify the caller's table by accident.
- `mean(axis=1)`: average **across columns**, i.e. one result per student.
- The `.isna()` lines: if attendance is missing, "low attendance" is *unknown*, not "no". Without these lines a missing value would silently become 0, which is a wrong guess.
- This function works **one row at a time**, so it cannot leak test information.

### An honest result
I measured it: Logistic Regression with 5-fold cross-validation scored **F1 = 0.847 without** the engineered features and **0.845 with** them. **They did not help.** That is normal and worth learning: the model could already combine the three scores itself, so our "average" added no new information. In real projects, *engineered features must prove their worth by experiment.* We keep them in the project so you can see how the mechanism works (and set `ENGINEERED_FEATURES = []` in `config.py` to try the difference yourself).

### The Pipeline (very important idea)
`build_pipeline()` chains **everything** into one object:
```
add_features → fill missing (median / most frequent) → scale numbers / encode categories → model
```
- **`SimpleImputer(strategy="median")`**: fills a missing number with the median (the middle value) of the *training* data. Median is better than mean when outliers exist.
- **`StandardScaler`**: rescales each number to mean 0, spread 1. Without it, a column like attendance (0-100) would dominate one like study hours (0-12) in Logistic Regression.
- **`OrdinalEncoder`**: turns `Low/Medium/High` into `0/1/2`. Models need numbers, and these categories have a natural order, so keeping the order is sensible. (For unordered categories like city names you would use `OneHotEncoder`.)
- **`ColumnTransformer`**: applies different steps to different columns.

**Why a Pipeline?** (1) During training, each step is fitted on **training data only**, so no leakage. (2) The same pipeline object is saved and used in the app, so **training and prediction preprocessing can never get out of sync.**

**✅ Checklist:** you can list the target, the 8 original features and the 3 new ones, and explain why a Pipeline prevents leakage.

---

## Step 8: Train-Test Split

### Concept
Imagine a student who memorises the answers to last year's paper and scores 100% on that same paper. Did they learn the subject? Unknown. The fair test is a **new paper**.

ML is the same. We split our data in two:
- **Training data (80%)**: the model learns from this.
- **Testing data (20%)**: kept hidden until the end, then used as the "final exam". The score on it estimates how the model will do on **new, real students**.

### Code (in `src/train.py`)
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```
### Code explained
- `X` = the feature columns, `y` = the target column.
- **`test_size=0.2`**: 20% of rows go to testing. 1,500 × 0.2 = 300 test students, 1,200 training students. 80/20 is a common default; with small data you may want more training rows.
- **`random_state=42`**: the split is random, but a fixed seed makes it **the same every run**, so your results are repeatable and comparable. (42 has no special meaning; any number works.)
- **`stratify=y`**: keeps the Pass/Fail ratio the same in both parts. Otherwise, by bad luck, the test set might contain far more Passes than the training set.

### Expected output
`Training rows: 1200 | Testing rows: 300`

**✅ Checklist:** you can explain why we never evaluate on the training data.

---

## Step 9: Model Training

### What does "training" mean?
The model looks at the training rows and adjusts its internal numbers until its predictions match the known results as well as possible. In scikit-learn, this is one line: `pipeline.fit(X_train, y_train)`. Afterwards `pipeline.predict(new_students)` gives predictions.

### Model 1: Logistic Regression
*Despite the name, it is a classification method.* It gives each feature a **weight** (positive pushes toward Pass, negative toward Fail), adds them up (weight × value), and squeezes the total into a probability between 0 and 1 with an S-shaped curve. Think of it as a *weighted checklist*.
**Strengths:** fast, simple, easy to interpret, gives well-behaved probabilities. **Weakness:** can only draw a "straight" boundary between Pass and Fail.

### Model 2: Decision Tree
A flowchart of yes/no questions learned from data: *"Attendance > 74%? → Previous score > 61%? → …"* until it reaches Pass or Fail.
**Strengths:** very easy to explain and visualise. **Weakness:** if allowed to grow deep, it **memorises** the training data and does worse on new data (this is **overfitting**). We limit it with `max_depth=5`.

### Model 3: Random Forest
Many (here 200) decision trees, each trained on a slightly different random sample, that **vote**. Like asking 200 slightly different teachers instead of one.
**Strengths:** usually more accurate and stable than one tree, handles curved relationships. **Weakness:** slower, harder to explain.

### Code
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}
for name, model in models.items():
    pipeline = build_pipeline(model)      # preprocessing + this model
    pipeline.fit(X_train, y_train)        # learn
    predictions = pipeline.predict(X_test)
```
Notes: `max_iter=1000` gives Logistic Regression enough attempts to settle on its weights. `max_depth` limits complexity to reduce overfitting. We use **default-style settings on purpose**. Tuning is a later topic.

**✅ Checklist:** you can describe each of the three models in one sentence and explain *overfitting*.

---

## Step 10: Model Evaluation

### The four outcomes
For each test student, the model's prediction is either right or wrong, and "Pass" is our **positive** class:

| | Predicted **Fail** | Predicted **Pass** |
|---|---|---|
| **Actually Fail** | ✅ True Negative (TN) | ❌ False Positive (FP): said Pass, actually failed |
| **Actually Pass** | ❌ False Negative (FN): said Fail, actually passed | ✅ True Positive (TP) |

This table is the **confusion matrix**: it shows *where* the model gets confused, not just how often.

### The metrics, in plain language
- **Accuracy** = (TP + TN) / all. *"Out of all students, how many predictions were right?"* Easy, but misleading if one class is rare.
- **Precision** = TP / (TP + FP). *"When the model says Pass, how often is it right?"* Low precision = it tells too many failing students they are safe.
- **Recall** = TP / (TP + FN). *"Of all students who really pass, how many did the model find?"* Low recall = it wrongly worries about students who would have passed.
- **F1-score** = the balance (harmonic mean) of precision and recall. One number for "good at both".

### Results on the 300-student test set
| Model | Accuracy | Precision | Recall | F1-score | CV F1 (train) |
|---|---|---|---|---|---|
| Logistic Regression | 0.830 | 0.870 | 0.825 | 0.847 | **0.843** |
| Decision Tree | 0.810 | 0.896 | 0.754 | 0.819 | 0.817 |
| Random Forest | 0.827 | 0.852 | 0.842 | 0.847 | 0.836 |

### Choosing the best model, and why
We pick by **cross-validated F1 on the training data**. Cross-validation splits the training data into 5 parts, trains on 4, checks on the 5th, rotates, and averages. This gives a fair comparison *without touching the test set*. If we chose the winner by looking at the test set, the test set would no longer be a truly untouched final exam.

**Winner: Logistic Regression** (CV F1 0.843). Reasons to be comfortable with that:
1. The three models are **very close** (differences of 1-2 points are within normal noise for 300 test students). When models tie, **prefer the simplest**: it is faster, easier to explain, and its probabilities are smooth.
2. Our data was generated from a formula resembling logistic regression, so it is natural that the simple model fits well. *On real, messier data, Random Forest often wins.* Do not generalise "Logistic Regression is best" from this one project.
3. F1 balances both kinds of mistakes. If your school cared more about *never missing a failing student*, you would switch the positive class to "Fail" or favour recall for Fail, a decision about the real-world cost of mistakes, not just maths.

### Confusion matrix for the chosen model
```
[[108  21]      row 1 = actually Fail:  108 caught, 21 wrongly told "Pass"
 [ 30 141]]     row 2 = actually Pass:   30 wrongly told "Fail", 141 correct
```
Plain English: of the 129 students who really failed, the model caught 108 (**84%**). Of the 171 who really passed, it found 141 (**82%**). Charts for all three models are saved as `reports/figures/confusion_matrices.png`.

### Run it
```bash
python -m src.train
```
Expected output ends with `Best model: Logistic Regression` and `Saved model to .../models/student_performance_model.joblib`.

**✅ Checklist:** you can read the confusion matrix, define precision and recall in your own words, and explain why we chose the model with cross-validation instead of the test set.

---

## Step 11: Save the Model

### Concept
Training takes time and needs the data. **Serialization** means converting a Python object (our trained model) into a file, so it can be loaded again later in milliseconds. Joblib is a library designed for this (it handles NumPy arrays well).

### Why not train inside the Streamlit app?
- Streamlit re-runs the script on **every click**. Retraining each time would make the app slow.
- The app should not need the training data at all.
- Everyone who uses the app should get **the same, consistent model**, not one that changes with each run.

*Training = done once, offline. Prediction = done many times, instantly.*

### Code (end of `src/train.py`)
```python
import joblib
joblib.dump({
    "pipeline": best_pipeline,          # preprocessing + model, together
    "model_name": best_name,
    "metrics": results_table.loc[best_name].to_dict(),
    "feature_columns": raw_feature_columns,
}, MODEL_PATH)                          # models/student_performance_model.joblib
```
and in the app: `bundle = joblib.load(MODEL_PATH)`.

Notice we save the **whole Pipeline**, not just the model. That is the key to "same preprocessing in training and prediction".

### Things to know
- The saved pipeline contains a reference to `add_features` from `src/features.py`, so `app.py` imports it before loading (otherwise Python cannot rebuild the object).
- **Only load joblib/pickle files you trust.** They can execute code when loaded.
- A model saved with one scikit-learn version may not load on a very different version. Keep `requirements.txt` consistent.

**✅ Checklist:** `models/student_performance_model.joblib` and `models/model_comparison.csv` exist.

---

## Step 12 and 13: The Streamlit Frontend and User Experience

### Concept
**Streamlit** turns a Python script into a web app, with no HTML/JavaScript. The script runs top to bottom each time the user interacts. Widgets (`st.slider`, `st.button`, …) return their current values as normal Python variables.

### Layout (see `app.py`)
| Part | Code | Purpose |
|---|---|---|
| Page title | `st.set_page_config`, `st.title` | clear identity |
| Sidebar inputs | `st.sidebar.slider(...)`, `selectbox`, `number_input` | all 8 student details in one place |
| Predict button | `st.button("Predict")` | prediction runs only on click |
| Result | `st.success` (green PASS) / `st.error` (red FAIL) | instant, colour-coded visual indicator |
| Probability | `st.metric`, `st.progress` | numbers *and* a bar |
| Explanation | text + "Positive/Risk factors" columns | the result in plain English |
| About the model | `st.expander` | metrics for the curious, hidden by default |

### Input validation (two layers)
1. **Built into the widgets:** sliders have `min`/`max`, so a user cannot enter attendance of 150% or negative hours.
2. **Logic checks:** `find_input_problems()` blocks combinations that make no sense (study + sleep > 22 hours a day) and shows `st.error` instead of predicting. Also, if the model file is missing, the app shows how to create it (`st.stop()`).

### The heart of the app
```python
student = pd.DataFrame([{ "study_hours": study_hours, "attendance": attendance, ... }])

prediction = pipeline.predict(student)[0]                       # 0 or 1
pass_column = list(pipeline.classes_).index(1)                  # which column is "Pass"?
probability_pass = pipeline.predict_proba(student)[0][pass_column]
```
- The column names **must match training exactly**, because the pipeline picks columns by name.
- `predict` gives the class; `predict_proba` gives the probability of each class. We look up the column for class 1 (Pass) instead of assuming its position.
- `@st.cache_resource` loads the model once and keeps it in memory.

### What the "explanation" is, and is not
The "Positive factors / Risk factors" text comes from **simple rules** (e.g. attendance below 75%), written in `explain()`. It is a helpful summary, **not** a mathematical breakdown of the model's decision. A real feature-attribution method (like SHAP) is a good future improvement.

### Expected behaviour (verified)
| Student | Result |
|---|---|
| Default sliders | PASS, about 69% pass probability |
| Strong (95% attendance, scores mid-80s, no failures, High participation) | PASS, ~100% |
| Weak (55% attendance, scores in the 30s, 4 failures, Low participation) | FAIL, ~0% pass probability |
| Study 12 h + sleep 11 h | red error message, no prediction |

The 100%/0% extremes look dramatic. Logistic Regression becomes very confident for unusual, extreme inputs. In real use, treat probabilities as a guide rather than certainty. The app's caption says so.

**✅ Checklist:** the app runs (Step 16), a prediction appears when you click **Predict**, and the validation error appears for 12 + 11 hours.

---

## Step 14: Project Architecture

```
User Input                (sidebar sliders / dropdown in the browser)
    ↓
Streamlit                 (app.py collects values, builds a 1-row table)
    ↓
Data Preprocessing        (inside the saved Pipeline: add_features → fill gaps → scale/encode)
    ↓
Trained ML Model          (Logistic Regression, loaded with joblib)
    ↓
Prediction                (Pass or Fail)
    ↓
Prediction Probability    (e.g. 69% pass)
    ↓
Result displayed in Streamlit   (green/red message, metrics, progress bar, explanation)
```
And the offline part that creates the model (run once):
```
generate_data.py → students_raw.csv → data_cleaning.py → train/test split
→ Pipeline + 3 models → evaluation → best Pipeline → joblib file → used by app.py
```

---

## Step 15: Requirements

`requirements.txt` lists the libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, joblib, and jupyter for the notebook).

```bash
pip install -r requirements.txt
```
`-r` means "read the list from this file". pip installs every library on it, so anyone can recreate your environment with one command.

---

## Step 16: Running the Project

Run all commands from inside the `student-performance-ml/` folder.

**1. Create a virtual environment** (an isolated box of libraries for this project, so versions do not clash with other projects):
```bash
python -m venv .venv
```
**2. Activate it:**
```bash
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```
You should see `(.venv)` at the start of your terminal line.

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```
**4. Create the data, explore it, train the model:**
```bash
python -m src.generate_data        # creates data/students_raw.csv
python -m src.eda                  # prints exploration, saves charts (optional)
python -m src.train                # cleans, trains, evaluates, saves the model
```
(`python -m src.train` rather than `python src/train.py`. The `-m` form lets Python find our `src` package.)

**5. Start the app:**
```bash
streamlit run app.py
```
Your browser opens at `http://localhost:8501`. Press `Ctrl + C` in the terminal to stop it.

### Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You are not in the project root folder, or you ran `python src/train.py`. Use `python -m src.train` from the root. |
| App says "Model file not found" | Run `python -m src.train` first. |
| `FileNotFoundError: students_raw.csv` | Run `python -m src.generate_data` first. |
| `streamlit: command not found` | Activate the virtual environment, or use `python -m streamlit run app.py`. |
| Slightly different metrics | Different library versions can change decimals a little. That is fine. |

---

## Step 17: README
See `README.md`. A good README lets a stranger understand, install and run your project in five minutes. Add your own Streamlit screenshots to the placeholder section after you run the app (take them from your browser and save them in a `docs/` folder).

---

## Step 18: What You Learned

| Concept | Plain-English meaning | Where in this project |
|---|---|---|
| **Dataset** | A table of examples (rows) and their properties (columns) | `students_raw.csv` |
| **Features** | The input columns the model learns from | study hours, attendance, … |
| **Target** | The column we want to predict | `result` (Pass/Fail) |
| **Classification** | Predicting a *category* | Pass vs Fail |
| **Training data** | The examples the model learns from (80%) | `X_train`, `y_train` |
| **Testing data** | Hidden examples for an honest final score (20%) | `X_test`, `y_test` |
| **Data preprocessing** | Cleaning and converting data so models can use it | `data_cleaning.py` + the Pipeline |
| **Data leakage** | Test information accidentally influencing training, which gives falsely good scores | we split first; imputing/scaling is inside the Pipeline |
| **Feature engineering** | Creating new, more informative columns | `add_features()` (and the lesson that they must prove their worth) |
| **Model training** | The model adjusting itself to fit the training examples | `pipeline.fit(X_train, y_train)` |
| **Overfitting** | Memorising training data, failing on new data | why we limit tree depth and keep a test set |
| **Prediction** | Using the trained model on new data | `pipeline.predict(student)` |
| **Accuracy** | Share of all predictions that are right | 83% |
| **Precision** | When it says Pass, how often it is right | 87% |
| **Recall** | Of the real passers, how many it found | 82.5% |
| **F1-score** | Balance of precision and recall | 0.847 |
| **Confusion matrix** | Table of right and wrong predictions by type | `[[108 21] [30 141]]` |
| **Cross-validation** | Rotating mini train/validation splits for a fair comparison | model selection |
| **Model serialization** | Saving a trained model to a file | `joblib.dump` / `joblib.load` |
| **Streamlit** | Python library that turns scripts into web apps | `app.py` |

---

## Step 19: Code Quality Rules: How They Were Applied
- **Beginner-friendly, meaningful names, comments:** `previous_failed_subjects`, `probability_pass`, and a comment wherever the *why* is not obvious.
- **Separation of concerns:** cleaning (`data_cleaning.py`), features/pipeline (`features.py`), training (`train.py`), the app (`app.py`).
- **No leakage:** rule-based cleaning before the split; every learned step (median, scaling) is inside the Pipeline and fitted on training data only; the winner is chosen with cross-validation on training data, not the test set.
- **Same preprocessing in training and prediction:** one saved Pipeline object.
- **Simple ML only:** three classic algorithms, no tuning, no stacking.

---

## Step 20: Where to Go Next (Honest Limits and Exercises)

**Know the limits of this project**
- The data is **synthetic**. Real students are messier and the numbers (and which model wins) will differ.
- The predictions are statistical estimates, not facts about an individual. Use them to **offer support early**, not to label students. Be careful about fairness and privacy with any real student data.

**Exercises, in order of difficulty**
1. In `config.py`, set `ENGINEERED_FEATURES = []`, re-run `python -m src.train`, and compare the scores.
2. Change `test_size` to 0.3 and `max_depth` of the tree to 2 and 10. What happens to the Decision Tree's score? (Hint: underfitting vs overfitting.)
3. Change the positive class so that "Fail" is what you try to catch. Which model has the best recall for failing students?
4. Replace the synthetic data with the UCI Student Performance dataset and adapt the column names.
5. Tune the Random Forest with `GridSearchCV`.
6. Deploy the app free on Streamlit Community Cloud.
