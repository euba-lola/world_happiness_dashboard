import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="World Happiness Dashboard (2015–2019)", layout="wide")


# ----------------------------
# 1) DATA LOADING / CLEANING
# ----------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("df_combined.csv")
    df["year"] = df["year"].astype(int)

    # Make sure these columns exist (prevents crashes if a column is missing)
    expected_cols = [
        "country", "region", "year", "happiness_score", "gdp_per_capita",
        "social_support", "healthy_life_expectancy", "freedom",
        "generosity", "perceptions_of_corruption"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df_combined.csv: {missing}")

    return df


# ----------------------------
# 2) PLOT FUNCTIONS (RETURN fig)
# ----------------------------
def fig_region_counts(df):
    region_counts = (
        df.dropna(subset=["region"])
          .groupby("region")["country"]
          .nunique()
          .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()
    region_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Region")
    ax.set_ylabel("Number of Countries")
    ax.set_title("Number of Countries per Region (2015–2019)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def fig_region_year_stacked(df):
    region_year_counts = (
        df.dropna(subset=["region"])
          .groupby(["year", "region"])["country"]
          .nunique()
          .unstack()
          .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    region_year_counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Countries")
    ax.set_title("Regional Distribution of Countries by Year (2015–2019)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def fig_top5_trends(df):
    top5_per_year = (
        df.sort_values(["year", "happiness_score"], ascending=[True, False])
          .groupby("year")
          .head(5)
    )
    top_countries = top5_per_year["country"].unique()
    trend_data = df[df["country"].isin(top_countries)][["country", "year", "happiness_score"]].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    for country in top_countries:
        temp = trend_data[trend_data["country"] == country]
        ax.plot(temp["year"], temp["happiness_score"], marker="o", label=country)

    ax.set_xlabel("Year")
    ax.set_ylabel("Happiness Score")
    ax.set_title("Happiness Score Trends (Top 5 Countries Each Year, 2015–2019)")
    ax.set_xticks([2015, 2016, 2017, 2018, 2019])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def fig_corr_heatmap(df):
    cols = [
        "happiness_score",
        "gdp_per_capita",
        "social_support",
        "healthy_life_expectancy",
        "freedom",
        "generosity",
        "perceptions_of_corruption",
    ]
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values)

    # Labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    # Annotate cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("Correlation Matrix of Happiness Score and Key Factors")
    fig.tight_layout()
    return fig


def fig_region_happiness_trends(df):
    region_pivot = (
        df.dropna(subset=["region"])
          .groupby(["year", "region"])["happiness_score"]
          .mean()
          .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for region in region_pivot.columns:
        ax.plot(region_pivot.index, region_pivot[region], marker="o", label=region)

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Happiness Score")
    ax.set_title("Average Happiness Score by Region (2015–2019)")
    ax.set_xticks([2015, 2016, 2017, 2018, 2019])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def fig_gdp_vs_happy_outliers_annotated(df):
    gdp_median = df["gdp_per_capita"].median()
    happy_median = df["happiness_score"].median()

    high_gdp_low_happy = df[(df["gdp_per_capita"] >= gdp_median) & (df["happiness_score"] < happy_median)]
    low_gdp_high_happy = df[(df["gdp_per_capita"] < gdp_median) & (df["happiness_score"] >= happy_median)]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(df["gdp_per_capita"], df["happiness_score"], alpha=0.12, s=20, label="All Countries", zorder=1)
    ax.scatter(high_gdp_low_happy["gdp_per_capita"], high_gdp_low_happy["happiness_score"],
               s=70, edgecolor="black", label="High GDP, Low Happiness", zorder=3)
    ax.scatter(low_gdp_high_happy["gdp_per_capita"], low_gdp_high_happy["happiness_score"],
               s=70, edgecolor="black", label="Low GDP, High Happiness", zorder=3)

    ax.axvline(gdp_median, linestyle="--", alpha=0.6)
    ax.axhline(happy_median, linestyle="--", alpha=0.6)

    annotate_high = high_gdp_low_happy.sort_values("gdp_per_capita", ascending=False).head(5)
    annotate_low = low_gdp_high_happy.sort_values("happiness_score", ascending=False).head(5)

    for _, row in annotate_high.iterrows():
        ax.annotate(row["country"], (row["gdp_per_capita"], row["happiness_score"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    for _, row in annotate_low.iterrows():
        ax.annotate(row["country"], (row["gdp_per_capita"], row["happiness_score"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Happiness Score")
    ax.set_title("GDP vs Happiness (Outliers Annotated)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_finland_indicators(finland_data):
    indicators = [
        "happiness_score",
        "gdp_per_capita",
        "social_support",
        "healthy_life_expectancy",
        "freedom",
        "generosity",
        "perceptions_of_corruption"
    ]

    titles = [
        "Happiness Score",
        "GDP per Capita",
        "Social Support",
        "Healthy Life Expectancy",
        "Freedom",
        "Generosity",
        "Perceptions of Corruption"
    ]
    # Distinct colours for each indicator
    colors = [
        "#1f77b4",  # blue – happiness
        "#2ca02c",  # green – GDP
        "#9467bd",  # purple – social support
        "#ff7f0e",  # orange – life expectancy
        "#17becf",  # teal – freedom
        "#8c564b",  # brown – generosity
        "#d62728"   # red – corruption
    ]
    
    fig = plt.figure(figsize=(12, 10))

    for i, (col, title) in enumerate(zip(indicators, titles), start=1):
        plt.subplot(4, 2, i)
        plt.plot(finland_data["year"], finland_data[col], marker="o")
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.xticks([2015, 2016, 2017, 2018, 2019])
        plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 8)
    plt.axis("off")

    plt.tight_layout()
    return fig


# ✅ Missing function you referenced in your UI
def fig_finland_happiness_gdp(finland_data):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(finland_data["year"], finland_data["happiness_score"], marker="o", label="Happiness Score")
    ax.plot(finland_data["year"], finland_data["gdp_per_capita"], marker="o", label="GDP per Capita")
    ax.set_title("Finland: Happiness vs GDP (2015–2019)")
    ax.set_xlabel("Year")
    ax.set_xticks([2015, 2016, 2017, 2018, 2019])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------
# 3) STREAMLIT UI
# ----------------------------
df = load_and_prepare_data()

st.title("World Happiness Dashboard (2015–2019)")

with st.sidebar:
    st.header("Filters")
    years = sorted(df["year"].unique().tolist())
    year_choice = st.multiselect("Year(s)", years, default=years)

    regions = sorted(df["region"].dropna().unique().tolist())
    region_choice = st.multiselect("Region(s)", regions, default=regions)

# Filter safely
df_f = df[df["year"].isin(year_choice)]
df_f = df_f[df_f["region"].isin(region_choice)].copy()

finland_data = df_f[df_f["country"] == "Finland"].sort_values("year")

st.caption(f"Showing {len(df_f):,} rows")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Relationships", "Data"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_region_counts(df_f))
    with c2:
        st.pyplot(fig_region_year_stacked(df_f))

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_top5_trends(df_f))
    with c2:
        st.pyplot(fig_region_happiness_trends(df_f))

    st.divider()
    st.subheader("Finland: Socio-Economic Profile (2015–2019)")

    if finland_data.empty:
        st.warning("Finland data is not available for the selected filters (try including Western Europe and all years).")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(fig_finland_indicators(finland_data))
        with right:
            st.pyplot(fig_finland_happiness_gdp(finland_data))

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_corr_heatmap(df_f))
    with c2:
        st.pyplot(fig_gdp_vs_happy_outliers_annotated(df_f))

with tab4:
    st.subheader("Data preview (filtered)")
    st.dataframe(df_f, use_container_width=True)

    st.download_button(
        "Download filtered data as CSV",
        df_f.to_csv(index=False).encode("utf-8"),
        file_name="world_happiness_filtered.csv",
        mime="text/csv"
    )

