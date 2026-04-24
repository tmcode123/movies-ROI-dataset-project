# 🎬 Movie ROI Analysis

**Which genre gives the best return on investment — and can we predict whether a film will be profitable?**

Analysis of 3,800+ films (1980–2016) using IMDB and TMDB data.

## Key findings

- **Horror dominates low-budget ROI** — median 3.6× return under $10M budget, with a 78% profitability rate
- **The edge disappears at scale** — above $40M, genre stops mattering; blockbuster economics compress ROI across the board
- **Audience engagement beats budget** — `num_voted_users` is the strongest predictor of profitability, more than money spent
- **The model works** — Gradient Boosting predicts profitability with 79% accuracy and AUC = 0.80

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit altair

python 01_data_cleaning.py        # clean raw data → data/movies_clean.csv
jupyter notebook ROI_analysis.ipynb   # full analysis: EDA + model + findings
streamlit run ROI_app.py          # interactive dashboard with profitability predictor
```

## Project structure

```
├── 01_data_cleaning.py       # ETL pipeline with audit trail
├── ROI_analysis.ipynb        # Full analysis notebook
├── ROI_app.py                # Streamlit dashboard
└── data/
    ├── movie_metadata.csv    # Raw — IMDB sourced
    └── tmdb_5000_movies.csv  # Raw — TMDB API
```

## Data sources
[IMDB Movie Metadata](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset) · [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
