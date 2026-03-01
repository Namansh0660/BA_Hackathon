# 🎓 OULAD — Student Burnout & Dropout Risk Monitor

> **Predicting student disengagement and personalising interventions using the Open University Learning Analytics Dataset**

[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20OULAD-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-93.4%25-10B981?style=flat-square)](/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python)](/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](/)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Behavioural Features Engineered](#behavioural-features-engineered)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Selection](#model-selection)
- [Results](#results)
- [Dashboard](#dashboard)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Key Insights](#key-insights)

---

## Overview

This project builds an end-to-end machine learning system to:

1. **Detect early burnout risk** in Open University distance-learning students using behavioural signals from the Virtual Learning Environment (VLE)
2. **Segment students** into engagement tiers via K-Means clustering
3. **Predict learning style preferences** using Gradient Boosting (93.4% accuracy, 5 classes)
4. **Deliver personalised course recommendations** across 15 distinct intervention pathways
5. **Visualise everything** in a real-time Streamlit dashboard with risk gauges, Z-score deviation charts, and actionable intervention plans

**The core insight:** ~42% of students face negative outcomes (withdrawal or failure). Traditional approaches wait until withdrawal occurs. This system detects disengagement weeks earlier using clickstream behavioural patterns.

---

## Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | Open University Learning Analytics Dataset (OULAD) |
| **Type** | Public — Real-world behavioural data |
| **Source** | [Kaggle — anlgrbz/student-demographics-online-education-dataoulad](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) |
| **Institution** | The Open University, UK (largest distance-learning university in Europe) |
| **Coverage** | 7 course modules, academic years 2013–2014 |
| **Raw Records** | 10,655,280 VLE interaction records |
| **Students** | 23,343 unique students (after preprocessing) |

### Why This Dataset Fits Behavioural Analytics

OULAD is one of the richest publicly available educational behavioural datasets. It captures:

- **Clickstream behaviour**: Daily interactions with 19+ activity types (forums, quizzes, content, wikis, etc.)
- **Temporal patterns**: Submission timing, engagement trajectories over the full module (up to 269 days)
- **Socioeconomic context**: IMD (Index of Multiple Deprivation) band — a UK government measure of area-level deprivation
- **Multi-dimensional outcomes**: Pass / Fail / Distinction / Withdrawn — allowing nuanced rather than binary modelling
- **Diverse demographics**: Age bands, education levels, disability status, 13 geographic regions across UK and Ireland

### Source Files

```
courses.csv              22 rows    — Module codes and presentation lengths
assessments.csv         206 rows    — Assessment types, weights, due dates
studentAssessment.csv   173,912     — Student scores, submission dates
studentInfo.csv          32,593     — Demographics, region, education, final result
studentRegistration.csv  32,593     — Registration/unregistration dates
studentVle.csv        10,655,280    — Daily VLE click counts per student per resource
vle.csv                   6,364     — VLE resource metadata and activity types
```

---

## Behavioural Features Engineered

**15 features** were engineered from raw interaction logs. They fall into three groups:

### Engagement Intensity Features

| Feature | Formula | Behavioural Signal |
|---------|---------|-------------------|
| `assessment_engagement_score` | `sum_click × session_count` | Combines volume AND frequency — distinguishes binge vs. steady engagement |
| `module_engagement_rate` | `total_clicks ÷ module_length_days` | Daily VLE intensity normalised by module duration |
| `weighted_engagement` | `engagement_score × assessment_weight` | Up-weights engagement around high-stakes assessments |
| `activity_diversity` | `count(unique activity types)` | Breadth of resource usage — strongest cluster differentiator |
| `engagement_consistency` | `std(sum_click)` per student | Low variance = steady habits; high variance = boom-bust patterns |
| `engagement_dropoff` | `(max − min clicks) ÷ count` | Rate of engagement decline — early burnout trajectory signal |

### Academic Performance Features

| Feature | Formula | Behavioural Signal |
|---------|---------|-------------------|
| `submission_timeliness` | `date_submitted − due_date` | Persistent lateness signals time management issues / external stressors |
| `improvement_rate` | `(last_score − first_score) ÷ (n−1)` | Negative values = academic trajectory declining |
| `score_per_weight` | `score ÷ (weight + 1)` | Performance normalised by assessment stakes |
| `cumulative_score` | running sum of scores | Total academic achievement arc |
| `learning_pace` | `diff(date_submitted)` | Long gaps between submissions = potential disengagement |

### Composite / Context Features

| Feature | Formula | Behavioural Signal |
|---------|---------|-------------------|
| `repeat_student` | `1 if prev_attempts > 0` | Returning students have different risk profiles |
| `time_since_registration` | `assessment_date − registration_date` | Student readiness at time of assessment |
| `banked_assessment_ratio` | `mean(is_banked)` | Proportion of transferred credits |
| `performance_by_registration` | `score ÷ (registration_date + 1)` | Early/late registrants performance adjusted |

### Rule-Based Label: `study_method_preference`

The classification target was constructed using a threshold-based rule function on activity type counts (threshold = 5 interactions):

| Class | Triggering Activities | Students | % |
|-------|----------------------|----------|---|
| Collaborative | ouelluminate, ouwiki, oucollaborate, oucontent | 9,942 | 42.6% |
| Offline Content | No VLE activity exceeds threshold | 6,932 | 29.7% |
| Interactive | quiz, externalquiz, repeatactivity, questionnaire | 3,435 | 14.7% |
| Informational | Mixed usage, no dominant pattern | 2,571 | 11.0% |
| Resource-Based | resource, homepage, folder, url, page, glossary | 463 | 2.0% |

---

## Pipeline Architecture

```
Raw CSVs (7 files)
       │
       ▼
1. Ingest & Merge          → 1,579,985 rows × 26 columns
       │
       ▼
2. Null Handling            → score→0, withdrawal_status derived,
                              imd_band imputed by region mode
       │
       ▼
3. EDA & Category Cleanup   → 'No Formal quals'→'Lower Than A Level'
                              age bands merged to 2 groups
       │
       ▼
4. Feature Engineering      → +15 engineered features
                              +study_method_preference label
       │
       ▼
5. Student-Level Aggregation → 23,343 rows × 27 columns (mean/mode)
       │
       ▼
6. Scale & Encode            → MinMaxScaler (numerical)
                              OneHotEncoder (categorical)
                              → 23,343 × 68 matrix
       │
       ▼
7. K-Means Clustering        → k=3 on 7 engagement features
   (Silhouette = 0.397)       → engagement_classification label
       │
       ▼
8. Classification Training   → 8 models evaluated
   Gradient Boosting selected → 93.4% test accuracy, F1=0.932
       │
       ▼
9. Recommendation Engine     → 5 styles × 3 engagement levels
                              → 4 course recommendations per student
       │
       ▼
10. Streamlit Dashboard       → Real-time risk scoring + interventions
```

---

## Model Selection

### Clustering

K-Means was applied to 7 engagement features. Key finding: clustering on engagement features only achieves **4.7× higher silhouette score** (0.461) than clustering on all 68 features (0.097), confirming that engagement signals are the most discriminative.

| k | Silhouette (Engagement Features) | Decision |
|---|----------------------------------|---------|
| 2 | 0.461 (best score) | Too few tiers for actionable intervention |
| **3** | **0.397** | **SELECTED — Low / Moderate / High Risk tiers** |
| 4 | 0.415 | Marginal gain, interpretability cost |

### Classification

All 8 models were evaluated with stratified 70/30 split:

| Model | Train Acc | Test Acc | Test F1 | Note |
|-------|-----------|----------|---------|------|
| **Gradient Boosting** | 95.8% | **93.4%** | **0.932** | **SELECTED** |
| SVM (C=10, linear) | 93.0% | 93.3% | 0.927 | Close second |
| Random Forest | 100.0% | 92.2% | 0.919 | Overfit |
| Decision Tree | 100.0% | 90.4% | 0.904 | Overfit |
| Logistic Regression | 88.2% | 87.7% | 0.866 | |
| AdaBoost | 84.3% | 83.8% | 0.831 | |
| KNN (k=7) | 79.3% | 74.0% | 0.721 | |
| GaussianNB | 41.8% | 42.6% | 0.492 | Poor |

**Gradient Boosting was selected** because it achieves the best test F1 with only a 2.4% train-test gap — the smallest of all competitive models, demonstrating genuine generalisation rather than memorisation.

---

## Results

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Collaborative | 0.93 | 0.95 | 0.94 | 2,983 |
| Offline Content | 0.96 | 0.99 | 0.98 | 2,080 |
| Interactive | 0.92 | 0.90 | 0.91 | 1,030 |
| Informational | 0.91 | 0.82 | 0.86 | 771 |
| Resource-Based | 0.78 | 0.58 | 0.67 | 139 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** | **7,003** |

> Note: Resource-Based recall (0.58) is lower due to severe class imbalance (only 2% of students). SMOTE oversampling is planned for the next iteration.

### Engagement Cluster Profiles

| Cluster | Students | Avg Engagement | Profile |
|---------|----------|----------------|---------|
| Cluster 1 | 4,157 (17.8%) | High | Activity diversity 0.77 — active, self-directed |
| Cluster 0 | 11,843 (50.7%) | Moderate | Typical engagement pattern |
| Cluster 2 | 7,343 (31.5%) | Low | Activity diversity 0.36 — at-risk / offline learners |

---

## Dashboard

The Streamlit dashboard (`app.py`) provides two modes:

### Dashboard Mode (Existing Students)
- **Student selector** — search any of 23,343 students by ID
- **Risk gauge** — 0-100 burnout score with colour-coded threshold zones
- **Probability breakdown** — model confidence per engagement class
- **Z-score deviation chart** — student vs. cohort average across 9 behavioural metrics
- **Cohort distribution histograms** — where the student sits in the overall distribution
- **Engagement radar chart** — 7-dimension profile overlay vs. cohort mean
- **Intervention plan** — colour-coded action list with escalating urgency
- **Weakest metrics panel** — lists which specific features triggered the alert

### Simulation Mode (New Student)
- Sliders for: avg score, submission delay, VLE clicks, sessions, engagement rate, activity diversity, improvement rate
- Demographic overrides: gender, education, age, IMD band, disability
- All outputs recalculate in real time as sliders move

### Sidebar
- Model info panel with live performance metrics
- Adjustable Low / Medium / High risk threshold sliders

---

## Repository Structure

```
oulad-burnout-risk/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── notebook/
│   ├── app.py                     # Streamlit dashboard (main entry point)
│   ├── notebook.ipynb
│   ├── GB.pkl                      # Trained Gradient Boosting model
│   ├── SVC.pkl                     # Trained SVM model (alternative)
│   ├── data.csv                    # Preprocessed student-level features
│   └── merged.csv                  # Intermediate merged dataset
│
└── data/
    └── raw/                        # Place raw dataset CSVs here
```

> **Note:** The raw OULAD CSV files (`studentVle.csv`, `studentInfo.csv`, etc.) are not included in this repository due to file size constraints. Download them from Kaggle using the link above and place them in the `data/raw/` directory before running the notebook.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/oulad-burnout-risk.git
cd oulad-burnout-risk
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle OULAD](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) and place the CSV files in `data/raw/`.

### 4. Run the notebook

```bash
jupyter notebook notebook/notebook.ipynb
```

This will generate `data/data.csv`, `models/GB.pkl`, and `models/SVC.pkl`.

### 5. Launch the dashboard

```bash
# Navigate to the notebook directory and run the application
cd notebook
streamlit run app1.py
```

---

## Key Insights

### 1. 📉 Socioeconomic Status is the Dominant Predictor
IMD band (Index of Multiple Deprivation) explains more variance in outcomes than any other demographic feature. Students from the most deprived areas (0-20% band) consistently underperform relative to students from affluent areas (80-100%), independent of age, gender, or disability status.

### 2. 🖱️ Engagement Quality > Engagement Quantity  
Activity *diversity* (the breadth of VLE resource types used) is a stronger predictor of cluster membership than raw click volume. A student who uses forums, quizzes, and content is measurably different from one who only accesses the homepage, even if click counts are similar.

### 3. ♿ Disability Predicts Withdrawal, Not Failure
Disabled students do not academically underperform their peers but withdraw at disproportionately higher rates. This suggests accessibility and support barriers — not capability gaps — are the primary issue for this population.

### 4. 📈 Zero-Click Students Outperform Low-Click Students
Students with no VLE activity score better on average than students with 1–20 clicks. This counter-intuitive finding identifies a cohort of effective offline learners who should not be flagged as at-risk solely on VLE inactivity.

### 5. ⏰ Submission Timing is a Composite Signal
Submission timeliness alone has a correlation of only −0.1 with scores. It becomes meaningful when combined with declining engagement scores and low activity diversity — a pattern that identifies students under external stress.

---

## Assumptions

All assumptions are explicitly documented:

1. **Missing scores (173 cases)** = non-submissions, treated as 0
2. **date_unregistration nulls (92%)** = still-enrolled students, replaced with derived categorical features
3. **IMD band nulls** = imputed using modal band for student's region
4. **Student-level aggregation** uses mean/mode, assuming approximately unimodal within-student distributions
5. **Zero VLE records** = inferred offline learners, not absent students
6. **study_method threshold (5 interactions)** = minimum for intentional vs. casual resource use

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | pandas, numpy |
| ML Models | scikit-learn (KMeans, GradientBoostingClassifier, etc.) |
| Feature Scaling | MinMaxScaler, OneHotEncoder |
| Model Persistence | joblib |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Notebook | Jupyter |

---

## License

This project uses the OULAD public dataset under its Kaggle terms of service. The code is released under the MIT License.