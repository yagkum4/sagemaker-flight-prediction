import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from wordcloud import WordCloud, STOPWORDS
from IPython.display import display, HTML


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def display_html(size=3, content="content"):
    display(HTML(f"<h{size}>{content}</h{size}>"))


def rotate_xlabels(ax, angle=35):
    ax.set_xticklabels(ax.get_xticklabels(), rotation=angle, ha="right")


def rotate_ylabels(ax, angle=0):
    ax.set_yticklabels(ax.get_yticklabels(), rotation=angle)


# -------------------------------------------------------
# Pair Plots
# -------------------------------------------------------

def pair_plots(data, height=3, aspect=1.5, hue=None, legend=False):
    display_html(2, "Pair Plots")
    g = sns.PairGrid(data=data, height=height, aspect=aspect, hue=hue, corner=True)
    g.map_lower(sns.scatterplot)
    if legend:
        g.add_legend()


# -------------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------------

def correlation_heatmap(data, figsize=(12, 6), method="spearman", cmap="RdBu"):
    cm = data.corr(method=method, numeric_only=True)
    mask = np.triu(np.ones_like(cm, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, vmin=-1, vmax=1, cmap=cmap, annot=True, fmt=".2f",
                linewidths=1.5, square=True, mask=mask, ax=ax)
    rotate_xlabels(ax)
    rotate_ylabels(ax)
    ax.set_title(f"{method.title()} Correlation Matrix Heatmap")


# -------------------------------------------------------
# Numeric Summary
# -------------------------------------------------------

def num_summary(data, var):
    col = data[var]
    display_html(2, var)

    # Quick glance
    display_html(3, "Quick Glance")
    display(col)

    # Meta data
    display_html(3, "Meta-data")
    print(f"Data Type       : {col.dtype}")
    print(f"Missing Data    : {col.isna().sum()} rows ({col.isna().mean()*100:.2f}%)")
    print(f"Available Data  : {col.count()} / {len(col)} rows")

    # Percentiles
    display_html(3, "Percentiles")
    display(col.quantile([0, .05, .1, .25, .5, .75, .9, .95, .99, 1]).to_frame("value"))

    # Central tendency
    display_html(3, "Central Tendency")
    display(pd.Series({
        "mean": col.mean(),
        "trimmed mean (5%)": stats.trim_mean(col.dropna(), 0.05),
        "median": col.median()
    }).to_frame("value"))

    # Spread
    display_html(3, "Spread")
    display(pd.Series({
        "var": col.var(),
        "std": col.std(),
        "IQR": col.quantile(.75) - col.quantile(.25),
        "mad": stats.median_abs_deviation(col.dropna()),
        "coef_variance": col.std() / col.mean()
    }).to_frame("value"))

    # Normality tests
    display_html(3, "Normality Tests")

    shapiro = stats.shapiro(col.dropna())
    print("\nShapiro-Wilk Test")
    print(f"Statistic: {shapiro.statistic}, p-value: {shapiro.pvalue}")

    ad = stats.anderson(col.dropna(), dist="norm")
    print("\nAnderson-Darling Test")
    print(f"Statistic: {ad.statistic}, Critical (5%): {ad.critical_values[2]}")


# -------------------------------------------------------
# Numeric–Numeric Hypothesis Testing
# -------------------------------------------------------

def num_num_hyp_testing(data, var1, var2):
    display_html(2, f"Hypothesis Test: {var1} vs {var2}")

    df = data[[var1, var2]].dropna()

    # Pearson
    pr = stats.pearsonr(df[var1], df[var2])
    print("\nPearson Test")
    print(f"Statistic: {pr.statistic}, p = {pr.pvalue}")

    # Spearman
    sr = stats.spearmanr(df[var1], df[var2])
    print("\nSpearman Test")
    print(f"Statistic: {sr.statistic}, p = {sr.pvalue}")


# -------------------------------------------------------
# Numeric Univariate Plots
# -------------------------------------------------------

def num_univar_plots(data, var, bins=10, figsize=(15, 7)):
    display_html(2, f"Univariate Analysis of {var}")
    col = data[var]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    sns.histplot(data, x=var, bins=bins, kde=True, ax=axes[0])
    axes[0].set_title("Histogram")

    sns.ecdfplot(data=data, x=var, ax=axes[1], color="red")
    axes[1].set_title("CDF")

    # Power transform
    pt = PowerTransformer().fit_transform(col.dropna().values.reshape(-1, 1))
    sns.kdeplot(x=pt.ravel(), fill=True, ax=axes[2], color="#f2b02c")
    axes[2].set_title("Power Transformed")

    sns.boxplot(data, x=var, ax=axes[3], color="lightgreen")
    axes[3].set_title("Boxplot")

    sns.violinplot(data, x=var, ax=axes[4], color="purple")
    axes[4].set_title("Violin Plot")

    stats.probplot(col.dropna(), dist="norm", plot=axes[5])
    axes[5].set_title("QQ Plot")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# Cramer's V
# -------------------------------------------------------

def cramers_v(data, var1, var2):
    ct = pd.crosstab(data[var1], data[var2])
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, c = ct.shape
    return np.sqrt(chi2 / (n * (min(r - 1, c - 1))))


def cramersV_heatmap(data, figsize=(12, 6)):
    cats = data.select_dtypes(include="object").columns
    matrix = pd.DataFrame(index=cats, columns=cats, dtype=float)

    for c1 in cats:
        for c2 in cats:
            matrix.loc[c1, c2] = cramers_v(data, c1, c2) if c1 != c2 else 1

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, cmap="Blues", mask=mask, vmin=0, vmax=1)
    plt.title("Cramer's V Heatmap")
    plt.show()


# -------------------------------------------------------
# FIXED — Numeric Bivariate Plots (NO edgecolor bug)
# -------------------------------------------------------

def num_bivar_plots(data, var_x, var_y, figsize=(12, 4.5),
                    scatter_kwargs=None, hexbin_kwargs=None):

    scatter_kwargs = scatter_kwargs or {}
    hexbin_kwargs = hexbin_kwargs or {}

    # remove problematic params if present
    scatter_kwargs.pop("edgecolor", None)
    scatter_kwargs.pop("edgecolors", None)

    display_html(2, f"Bivariate Numeric Analysis: {var_x} vs {var_y}")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter
    sns.scatterplot(data=data, x=var_x, y=var_y, ax=axes[0], **scatter_kwargs)
    axes[0].set_title("Scatter Plot")

    # Hexbin
    hb = axes[1].hexbin(data[var_x], data[var_y], gridsize=40,
                        cmap="viridis", **hexbin_kwargs)
    axes[1].set_title("Hexbin Plot")
    plt.colorbar(hb, ax=axes[1])

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# Categorical Summary
# -------------------------------------------------------

def cat_summary(data, var):
    col = data[var]
    display_html(2, var)

    display_html(3, "Quick Glance")
    display(col)

    display_html(3, "Meta-data")
    print(f"Data Type: {col.dtype}")
    print(f"Cardinality: {col.nunique(dropna=True)}")
    print(f"Missing: {col.isna().sum()} ({col.isna().mean()*100:.2f}%)")

    display_html(3, "Summary")
    display(col.describe())


# -------------------------------------------------------
# Numeric vs Categorical Plots
# -------------------------------------------------------

def num_cat_bivar_plots(data, num_var, cat_var, figsize=(15, 4), estimator="mean"):

    display_html(2, f"Bivariate: {cat_var} vs {num_var}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sns.barplot(data=data, x=cat_var, y=num_var, ax=axes[0], estimator=np.mean)
    axes[0].set_title("Bar Plot")
    rotate_xlabels(axes[0])

    sns.boxplot(data=data, x=cat_var, y=num_var, ax=axes[1])
    axes[1].set_title("Box Plot")
    rotate_xlabels(axes[1])

    sns.violinplot(data=data, x=cat_var, y=num_var, ax=axes[2])
    axes[2].set_title("Violin Plot")
    rotate_xlabels(axes[2])

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# Missing Values
# -------------------------------------------------------

def missing_info(data):
    miss = data.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return pd.DataFrame({"count": miss, "percentage": (miss / len(data)) * 100})


def plot_missing_info(data, figsize=(10, 4)):
    df = missing_info(data)
    df.plot(kind="bar", figsize=figsize, color="green", edgecolor="black")
    plt.title("Missing Values per Column")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# DatePlots
# -------------------------------------------------------

def dt_univar_plots(data, var, target=None):
    display_html(3, f"Date Variable: {var}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    sns.histplot(data, x=var, ax=axes[0])
    axes[0].set_title("Distribution")

    if target:
        sns.lineplot(data=data, x=var, y=target, ax=axes[1])
        axes[1].set_title("Trend vs Target")
    else:
        axes[1].remove()

    plt.tight_layout()
    plt.show()
