"""
ROI_app.py — Movie ROI Explorer
================================
Rebuilt dashboard focused on one question:
Which genre gives the best return on investment, and can we predict profitability?

Run: streamlit run ROI_app.py
Requires: data/movies_clean.csv  (produced by 01_data_cleaning.py)
"""

import warnings
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie ROI Explorer",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie ROI Explorer")
st.caption(
    "Which genre gives the best return on investment? "
    "Based on 3,800+ films (1980–2016) from IMDB & TMDB."
)

# ── Load & prepare data ───────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("data/movies_clean.csv")
    df["budget_tier"] = pd.cut(
        df["budget"],
        bins=[0, 10e6, 40e6, 100e6, np.inf],
        labels=["Low (<$10M)", "Mid ($10–40M)", "High ($40–100M)", "Blockbuster (>$100M)"],
    )
    top_genres = [
        "Horror", "Comedy", "Action", "Drama", "Adventure",
        "Animation", "Thriller", "Crime", "Biography", "Fantasy",
    ]
    df["genre_bucket"] = df["primary_genre"].where(
        df["primary_genre"].isin(top_genres), "Other"
    )
    df["roi_capped"] = df["roi"].clip(upper=20)  # cap outliers for display
    return df


@st.cache_resource
def train_model(df):
    genre_d  = pd.get_dummies(df["genre_bucket"], prefix="genre")
    rating_d = pd.get_dummies(df["content_rating"].fillna("Unknown"), prefix="rating")
    tier_d   = pd.get_dummies(df["budget_tier"].astype(str), prefix="tier")

    X = pd.concat([
        df[["log_budget", "imdb_score", "duration", "title_year",
            "num_voted_users", "num_critic_for_reviews"]].fillna(0),
        genre_d, rating_d, tier_d,
    ], axis=1)

    y = df["profitable"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    clf.fit(X_train, y_train)
    return clf, X.columns.tolist()


df = load_data()
clf, feature_cols = train_model(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("🎯 Profitability Predictor")
st.sidebar.caption("Enter a film's details to estimate its chance of turning a profit.")

budget_input   = st.sidebar.number_input("Budget ($M)", min_value=0.1, max_value=500.0, value=15.0, step=0.5)
genre_input    = st.sidebar.selectbox("Genre", sorted(df["genre_bucket"].unique()))
rating_input   = st.sidebar.selectbox("Content Rating", ["PG", "PG-13", "R", "G", "Not Rated"])
duration_input = st.sidebar.slider("Runtime (mins)", 70, 180, 100)
imdb_input     = st.sidebar.slider("Expected IMDB Score", 1.0, 10.0, 6.5, 0.1)

# ── Prediction ────────────────────────────────────────────────────────────────

def build_input_row(budget_m, genre, rating, duration, imdb, df_ref, all_cols):
    row = {col: 0 for col in all_cols}
    row["log_budget"]            = np.log1p(budget_m * 1e6)
    row["imdb_score"]            = imdb
    row["duration"]              = duration
    row["title_year"]            = 2015
    row["num_voted_users"]       = df_ref["num_voted_users"].median()
    row["num_critic_for_reviews"] = df_ref["num_critic_for_reviews"].median()

    genre_col  = f"genre_{genre}"
    rating_col = f"rating_{rating}"

    budget_usd = budget_m * 1e6
    if   budget_usd < 10e6:  tier = "Low (<$10M)"
    elif budget_usd < 40e6:  tier = "Mid ($10–40M)"
    elif budget_usd < 100e6: tier = "High ($40–100M)"
    else:                    tier = "Blockbuster (>$100M)"
    tier_col = f"tier_{tier}"

    for col in [genre_col, rating_col, tier_col]:
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])


input_row = build_input_row(
    budget_input, genre_input, rating_input, duration_input, imdb_input, df, feature_cols
)
profit_prob = clf.predict_proba(input_row)[0][1]

# Sidebar result
st.sidebar.markdown("---")
color = "green" if profit_prob >= 0.6 else "orange" if profit_prob >= 0.45 else "red"
verdict = "✅ Likely Profitable" if profit_prob >= 0.6 else "⚠️ Uncertain" if profit_prob >= 0.45 else "❌ Likely Unprofitable"
st.sidebar.markdown(f"### {verdict}")
st.sidebar.metric("Probability of Profit", f"{profit_prob*100:.0f}%")

# Historical comparison
comp = df[
    (df["genre_bucket"] == genre_input) &
    (df["budget"].between(budget_input * 0.5e6, budget_input * 2e6))
]
if len(comp) >= 5:
    hist_pct = comp["profitable"].mean() * 100
    hist_roi = comp["roi"].median()
    st.sidebar.metric("Historical profit rate (similar films)", f"{hist_pct:.0f}%")
    st.sidebar.metric("Historical median ROI", f"{hist_roi:.1f}×")

# ── Main content ──────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Genre ROI", "📈 Trends", "🔍 Film Explorer"])

# ── Tab 1: Genre ROI ──────────────────────────────────────────────────────────

with tab1:
    st.subheader("Which genre gives the best return on investment?")

    tier_filter = st.radio(
        "Budget tier",
        ["All", "Low (<$10M)", "Mid ($10–40M)", "High ($40–100M)", "Blockbuster (>$100M)"],
        horizontal=True,
    )

    df_t = df if tier_filter == "All" else df[df["budget_tier"] == tier_filter]

    genre_summary = (
        df_t.groupby("genre_bucket")
        .agg(
            n=("roi", "count"),
            median_roi=("roi_capped", "median"),
            pct_profit=("profitable", "mean"),
            p25=("roi_capped", lambda x: x.quantile(0.25)),
            p75=("roi_capped", lambda x: x.quantile(0.75)),
        )
        .query("n >= 15")
        .reset_index()
        .sort_values("median_roi", ascending=False)
    )
    genre_summary["pct_profit_label"] = (genre_summary["pct_profit"] * 100).round(0).astype(int).astype(str) + "%"

    col1, col2 = st.columns(2)

    with col1:
        roi_chart = (
            alt.Chart(genre_summary)
            .mark_bar()
            .encode(
                x=alt.X("median_roi:Q", title="Median ROI (Gross / Budget)"),
                y=alt.Y("genre_bucket:N", sort="-x", title=""),
                color=alt.Color(
                    "median_roi:Q",
                    scale=alt.Scale(scheme="redyellowgreen", domain=[0.5, 3]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("genre_bucket:N", title="Genre"),
                    alt.Tooltip("median_roi:Q", title="Median ROI", format=".2f"),
                    alt.Tooltip("n:Q", title="Films"),
                ],
            )
            .properties(title="Median ROI by Genre", height=320)
        )
        breakeven = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(
            color="red", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(x="x:Q")
        st.altair_chart(roi_chart + breakeven, width='stretch')

    with col2:
        profit_chart = (
            alt.Chart(genre_summary)
            .mark_bar()
            .encode(
                x=alt.X("pct_profit:Q", title="% of Films Profitable", axis=alt.Axis(format="%")),
                y=alt.Y("genre_bucket:N", sort="-x", title=""),
                color=alt.Color(
                    "pct_profit:Q",
                    scale=alt.Scale(scheme="redyellowgreen", domain=[0.3, 0.9]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("genre_bucket:N", title="Genre"),
                    alt.Tooltip("pct_profit:Q", title="% Profitable", format=".0%"),
                    alt.Tooltip("n:Q", title="Films"),
                ],
            )
            .properties(title="% of Films That Turned a Profit", height=320)
        )
        half = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
            color="red", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(x="x:Q")
        st.altair_chart(profit_chart + half, width='stretch')

    # Summary cards for top 3
    st.markdown("#### 🏆 Top 3 genres by median ROI")
    top3 = genre_summary.head(3)
    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1, c2, c3], top3.iterrows()):
        col.metric(
            label=row["genre_bucket"],
            value=f"{row['median_roi']:.1f}× ROI",
            delta=f"{row['pct_profit']*100:.0f}% profitable ({int(row['n'])} films)",
        )

# ── Tab 2: Trends ─────────────────────────────────────────────────────────────

with tab2:
    st.subheader("How has genre ROI changed over time?")

    trend_genres = st.multiselect(
        "Select genres to compare",
        sorted(df["genre_bucket"].unique()),
        default=["Horror", "Comedy", "Action", "Drama"],
    )

    if trend_genres:
        trend_df = (
            df[df["genre_bucket"].isin(trend_genres)]
            .groupby(["decade", "genre_bucket"])
            .agg(median_roi=("roi_capped", "median"), n=("roi", "count"))
            .query("n >= 10")
            .reset_index()
        )

        trend_chart = (
            alt.Chart(trend_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("decade:O", title="Decade"),
                y=alt.Y("median_roi:Q", title="Median ROI"),
                color=alt.Color("genre_bucket:N", title="Genre"),
                tooltip=[
                    alt.Tooltip("decade:O", title="Decade"),
                    alt.Tooltip("genre_bucket:N", title="Genre"),
                    alt.Tooltip("median_roi:Q", title="Median ROI", format=".2f"),
                    alt.Tooltip("n:Q", title="Films"),
                ],
            )
            .properties(height=360)
        )
        breakeven_line = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
            color="red", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(y="y:Q")
        st.altair_chart(trend_chart + breakeven_line, width='stretch')
        st.caption("Red dashed line = break-even (ROI = 1.0). Points with < 10 films are hidden.")
    else:
        st.info("Select at least one genre above.")

# ── Tab 3: Film explorer ──────────────────────────────────────────────────────

with tab3:
    st.subheader("Explore individual films")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        genres_sel = st.multiselect("Genre", sorted(df["genre_bucket"].unique()), default=["Horror"])
    with col_b:
        min_y, max_y = int(df["title_year"].min()), int(df["title_year"].max())
        year_range = st.slider("Year range", min_y, max_y, (2000, max_y))
    with col_c:
        show_profitable = st.radio("Show", ["All", "Profitable only", "Unprofitable only"])

    df_explore = df[
        df["genre_bucket"].isin(genres_sel) &
        df["title_year"].between(year_range[0], year_range[1])
    ].copy()

    if show_profitable == "Profitable only":
        df_explore = df_explore[df_explore["profitable"] == 1]
    elif show_profitable == "Unprofitable only":
        df_explore = df_explore[df_explore["profitable"] == 0]

    scatter = (
        alt.Chart(df_explore)
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("budget:Q", title="Budget ($)", scale=alt.Scale(type="log")),
            y=alt.Y("roi_capped:Q", title="ROI (capped at 20×)"),
            color=alt.Color("genre_bucket:N", title="Genre"),
            size=alt.Size("gross:Q", title="Gross", legend=None),
            tooltip=[
                alt.Tooltip("movie_title:N", title="Film"),
                alt.Tooltip("title_year:O", title="Year"),
                alt.Tooltip("genre_bucket:N", title="Genre"),
                alt.Tooltip("budget:Q", title="Budget ($)", format="$,.0f"),
                alt.Tooltip("gross:Q", title="Gross ($)", format="$,.0f"),
                alt.Tooltip("roi:Q", title="ROI", format=".2f"),
                alt.Tooltip("imdb_score:Q", title="IMDB"),
            ],
        )
        .properties(height=400)
    )
    breakeven_h = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
        color="red", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(y="y:Q")
    st.altair_chart(scatter + breakeven_h, width='stretch')
    st.caption(f"Showing {len(df_explore):,} films. Bubble size = gross earnings. Log scale on x-axis.")

    # Table
    display_cols = ["movie_title", "title_year", "genre_bucket", "budget", "gross", "roi", "imdb_score"]
    st.dataframe(
        df_explore[display_cols]
        .rename(columns={"movie_title": "Title", "title_year": "Year",
                         "genre_bucket": "Genre", "imdb_score": "IMDB"})
        .sort_values("roi", ascending=False)
        .head(200),
        width='stretch',
        column_config={
            "budget": st.column_config.NumberColumn("Budget", format="$%.0f"),
            "gross":  st.column_config.NumberColumn("Gross",  format="$%.0f"),
            "roi":    st.column_config.NumberColumn("ROI", format="%.2f×"),
        },
    )
