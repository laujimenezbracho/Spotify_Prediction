# Hit or Miss: Predicting Song Popularity with Spotify Audio Features

# Project Overview
Can we predict whether a song will be a hit or a miss using only its audio characteristics? This project uses Spotify’s audio features dataset to analyze how musical attributes such as loudness, energy, danceability, and acousticness influence a song’s popularity. We frame the problem as a binary classification task, comparing interpretable statistical models with more powerful machine-learning approaches under real-world data imbalance.

# Objectives
- Understand which audio features correlate with song popularity
- Transform a skewed popularity score into a meaningful hit / miss classification
- Compare Logistic Regression and XGBoost
- Handle severely imbalanced data
- Evaluate trade-offs between interpretability and predictive performance

# Dataset
- Source: Spotify Audio Features dataset
- Target Variable: popularity (Spotify score from 0–100)

# Feature Engineering
Target Definition
- Hit = 1 if popularity > 30
- Miss = 0 otherwise
- (Threshold chosen because Spotify begins algorithmic promotion around this score)

# Selected Features
- Energy-related
- Energy
- Loudness
- Tempo
- Mood-related
- Valence
- Danceability
- Audio characteristics
- Acousticness
- Instrumentalness
- Speechiness
- Liveness

# Excluded Features
- Key
- Mode
- Time signature
- Track ID, artist name, track name

# Preprocessing
- Standardization using StandardScaler
- Tested interaction terms → no meaningful AUC improvement
- Class imbalance handled explicitly in modeling stage

# Models Implemented
Logistic Regression
- Baseline interpretable model
- class_weight = "balanced"
- Tested default threshold and custom threshold = 0.65
  
XGBoost Classifier
- Captures non-linear relationships
- Handles imbalance using scale_pos_weight
- Hyperparameter tuning with GridSearchCV (5-fold CV)

