# World Happiness Report Analysis (2015–2019) 📊🌍

This notebook presents a complete exploratory and analytical workflow using **World Happiness Report–style data from 2015 to 2019**. The analysis focuses on understanding **global and regional happiness patterns**, identifying **key determinants of happiness**, and generating **policy-relevant insights**.

---

## Project Objectives

The analysis aims to:

- Examine how happiness scores vary across **countries and regions**
- Identify which socio-economic and institutional factors are **most strongly associated with happiness**
- Explore **global and regional trends** in happiness over time (2015–2019)
- Highlight **outlier countries** (e.g., high GDP but relatively low happiness)
- Produce a **clean, merged dataset** suitable for visualization and dashboard deployment

---

## Data Overview

The notebook uses **five annual datasets**:

- `2015.csv`
- `2016.csv`
- `2017.csv`
- `2018.csv`
- `2019.csv`

Each dataset includes:
- `happiness_score` and `happiness_rank`
- Predictors such as:
  - `gdp_per_capita`
  - `social_support`
  - `healthy_life_expectancy`
  - `freedom`
  - `generosity`
  - `perceptions_of_corruption`

Because variable names and structures differ across years (especially in 2018–2019), extensive **data standardisation** is performed.

---

## Notebook Workflow

### 1. Data Loading & Inspection
- Import yearly datasets
- Inspect structure, missing values, and column inconsistencies

### 2. Data Cleaning & Standardisation
- Rename columns into consistent `snake_case`
- Add missing variables to ensure uniform structure
- Harmonise variable definitions across all years
- Map `region` for 2018–2019 using a lookup table
- Append all datasets into a single combined dataframe

📦 **Final output:** `df_combined` (2015–2019)

---

### 3. Exploratory Data Analysis (EDA)

The notebook includes visual and statistical exploration of:

- Distribution of countries across regions
- Global happiness score patterns
- Regional differences in happiness
- Top and bottom-ranking countries
- Year-to-year trends in happiness scores
- Outlier identification (e.g., mismatches between GDP and happiness)

---

### 4. Correlation Analysis

- Correlation matrix between happiness score and predictors
- Interpretation of relative importance of:
  - GDP per capita
  - Healthy life expectancy
  - Social support
  - Freedom
  - Perceptions of corruption
  - Generosity

---

### 5. Trend Analysis (2015–2019)

- Country-level and regional trends over time
- Stability vs. change among top-ranked countries
- Identification of consistent performers and emerging leaders

---

### 6. Policy-Focused Insights

The notebook concludes with **policy-relevant interpretations**, highlighting that:

- Happiness is **multidimensional** and not driven by income alone
- Social support, health, freedom, and governance consistently matter
- Economic growth without social and institutional investment yields limited well-being gains
- Long-term improvements require **integrated policy approaches**

---

## Outputs

This notebook produces:

- ✅ A fully cleaned and merged dataset (`df_combined.csv`)
- ✅ Exploratory and explanatory visualisations
- ✅ A structured analytical narrative
- ✅ Insights used to build the Streamlit dashboard

---

## Related Dashboard

The results from this notebook are deployed in an interactive Streamlit app:

🔗 **Live Dashboard:**  
https://worldhappinessdashboard-xvfefazshfsfhg6y9z3tzj.streamlit.app/

---

## Author

**Morenikeji Euba**  
Data Science | Public Health | Well-Being Analytics

---

## Notes

- This notebook is designed to be **reproducible**
- All transformations are explicitly documented in code
- The analysis is suitable for **academic assessment, policy communication, and dashboard development**

---
