# Did You Forget? Recommender System

[![Recall@5 Score](https://img.shields.io/badge/Recall%405-0.2522-green.svg)](https://img.shields.io/badge/Recall%405-0.2522-green.svg) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)

An unsupervised ML recommender system that predicts forgotten items in partial checkout carts using a **Hybrid RF + Item-Item Collaborative Filtering** approach. Achieved **Recall@5 = 0.2522** by blending customer habit patterns with functional item correlations.

## Problem Statement

**Challenge**: Predict the top 5 items a customer forgot in their partial cart using **unsupervised ML only**.  
**Metric**: Recall@5 (proportion of actual forgotten items appearing in top-5 predictions).  
**Data**: Large-scale retail transaction logs with timestamped baskets.

## Solution Architecture
Final Score = RF_Score + λ × ItemItem_Similarity_Boost
```bash
RF_Score = Frequency^α × Recency^(1-α)
α = 0.53 (60% Frequency bias)
λ = 0.35 (Similarity boost strength)
```


### Core Components

| Component | Key Parameters | Role |
|-----------|---------------|------|
| **RF Score** | `α=0.53`, `Freq^2.0` | Captures **habit/staple** buying patterns |
| **Item-Item CF** | `λ=0.35`, Cosine Similarity | Adds **functional correlation** boost |
| **Stability** | Pre-computed matrices | No RAM crashes (vs Apriori) |

## Performance Evolution

| Iteration | Technique | Max Recall@5 | Key Insight |
|-----------|-----------|--------------|-------------|
| **Initial** | RF + Apriori | **0.2223** | RAM unstable at low support |
| **Stable** | Pure RF | **0.2206** | Reliable but capped |
| **Winner** | RF + Item-Item CF | **0.2522** | Stable + precise tuning |

##  Quick Start
```bash
# Clone & install
git clone https://github.com/yourusername/did-you-forget-reco.git
cd did-you-forget-reco
pip install -r requirements.txt

# Run evaluation
python src/evaluate_model.py --data_path data/transactions.parquet

# Hyperparameter sweep
python src/tune_hyperparams.py --alpha_range 0.4 0.6 --lambda_range 0.2 0.5
```
## Project Structure
Basket-Recommender-Did-You-Forget/

- data/                         # Raw datasets (add your parquet files)
- 📁 notebooks/                    # Additional analysis (optional)
- 📁 submissions/                  # Training scripts (future)
- 📄 README.md                     # This file!
- 📄 Explaination-Writeup.docx     # Detailed write-up & results

## Key Insights & Learnings
- Frequency^2.0 was the biggest lift - staple items dominate "forgotten" predictions
- Item-Item CF > Apriori for stability (no MemoryErrors on large datasets)
- α=0.53 sweet spot - micro-tuning matters (0.02 change = 2-3% Recall lift)

## Tech Stack
Core: pandas, numpy, scikit-learn
Matrix Ops: scipy.sparse (memory efficient)
Viz: matplotlib, seaborn
Notebooks: Jupyter for reproducible experiments


