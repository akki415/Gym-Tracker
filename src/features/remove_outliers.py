import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
"""
for ploting outliers i am using box-plot and whith [IQR]
method. Reason i am using it is it basically give five point summary
and make it easy to spot outliers

"""
plt.style.use("seaborn-v0_8-ticks")

df[["acc_x", "label"]].boxplot(
    by="label",
    figsize=(20, 10),
    patch_artist=True,
    boxprops={"facecolor": "skyblue", "color": "black"},
)

"""say i want to look all acc_ outliers @ single plot for better observation"""

outliers_columns = list(df.columns[:6])

df[outliers_columns[:3] + ["label"]].boxplot(
    by="label",
    figsize=(25, 15),
    medianprops={"color": "red", "linewidth": 2, "linestyle": "--"},
    layout=(1, 3),
    patch_artist=True,
    boxprops={"facecolor": "skyblue", "color": "black"},
)

df[outliers_columns[3:6] + ["label"]].boxplot(
    by="label",
    figsize=(25, 15),
    medianprops={"color": "red", "linewidth": 2, "linestyle": "--"},
    layout=(1, 3),
    patch_artist=True,
    boxprops={"facecolor": "lightpink", "color": "black"},
)


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# IQR function
data_frame = df.copy()


def iqr_outlier_analysis(data_frame, columns, multiplier=1.5):
    """
    Calculate IQR and detect outliers for selected columns.

    Returns:
        results: IQR statistics for each column
        outlier_data: DataFrame containing only outlier values
        outlier_mask: Boolean mask indicating outliers
    """

    results = []
    outlier_mask = pd.DataFrame(index=data_frame.index)

    for col in columns:
        data = data_frame[col].dropna()  # dropping nan values

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Outlier mask
        mask = (data_frame[col] < lower_bound) | (data_frame[col] > upper_bound)

        outlier_mask[f"{col}_outlier"] = mask

        results.append(
            {
                "column": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
                "Outlier Count": mask.sum(),
                "Outlier %": mask.mean() * 100,
            }
        )

    # Summary table
    results = pd.DataFrame(results)

    # ADD THE OUTLIER COLUMNS TO ORIGINAL DATAFRAME
    data_frame[outlier_mask.columns] = outlier_mask
    data_frame["outlier"] = outlier_mask.any(axis=1)

    return data_frame, results, outlier_mask


# Loop over all columns
data_frame_iqr, results, outlier_mask = iqr_outlier_analysis(
    data_frame, outliers_columns
)
# plotting for IOR masking

iqr_outlier_columns = list(data_frame.columns[-3:])
raw_data_columns = list(data_frame.columns[3:6])


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, raw_col, outlier_col in zip(axes, raw_data_columns, iqr_outlier_columns):
    # Plot all observations
    for i, label in enumerate(data_frame["label"].unique()):
        label_data = data_frame[data_frame["label"] == label]

        # All observations
        ax.scatter([i] * len(label_data), label_data[raw_col], alpha=0.4)

        # Already detected IQR outliers
        outlier_values = label_data.loc[label_data[outlier_col], raw_col].dropna()

        ax.scatter(
            [i] * len(outlier_values),
            outlier_values,
            color="red",
            zorder=3,
            label="IQR flagged" if i == 0 else None,
        )

    ax.set_title(raw_col)
    ax.set_xlabel("Label")
    ax.set_ylabel("Acceleration")

    ax.set_xticks(range(len(data_frame["label"].unique())))
    ax.set_xticklabels(data_frame["label"].unique())

    ax.legend()

plt.suptitle("IQR-Flagged Accelerometer Observations by Label")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------
# Plotting binary outliers
# --------------------------------------------------------------


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots(figsize=(20, 5))

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.grid(True)
    plt.show()


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution


# Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    chauvenet_dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0
            - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    chauvenet_dataset[col + "_outlier"] = mask
    return chauvenet_dataset


# Loop over all columns
for col in outliers_columns:
    dataset_chauvenet = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset_chauvenet,
        col=col,
        outlier_col=col + "_outlier",
        reset_index=True,
    )


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------


# LOF function
def local_outlier_factor(dataset, columns, n=20):

    lof_dataset = dataset.copy()
    lof = LocalOutlierFactor(n_neighbors=n)
    outliers = lof.fit_predict(lof_dataset[columns])
    X_scores = lof.negative_outlier_factor_

    lof_dataset["outlier_lof"] = outliers == -1

    return lof_dataset, outliers, X_scores


# Looping and plotting over all columns
lof_dataset, outliers, x_scores = local_outlier_factor(df, outliers_columns)

for col in outliers_columns:
    plot_binary_outliers(
        dataset=lof_dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

label = "bench"
dataset = data_frame_iqr[data_frame_iqr["label"] == label]
dataset["label"].unique()

for col in outliers_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier", reset_index=True
    )


dataset_lof = df[df["label"] == label]
lof_dataset, outliers, x_scores = local_outlier_factor(dataset_lof, outliers_columns)
for col in outliers_columns:
    plot_binary_outliers(
        dataset=lof_dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )


for col in outliers_columns:
    dataset_chauvenet = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(
        dataset=dataset_chauvenet,
        col=col,
        outlier_col=col + "_outlier",
        reset_index=True,
    )


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------
""" Have decided to use chauvenet function """
# Test on single column

s_column = "gyr_z"
dataset_chauvenet = mark_outliers_chauvenet(df, s_column)
dataset_chauvenet[dataset_chauvenet["gyr_z_outlier"]]


outlier_removed_df = df.copy()

for label in df["label"].unique():
    # Get row positions belonging to this label
    row_positions = np.where(df["label"].values == label)[0]

    for column in outliers_columns:
        # Data for this label
        label_data = df.iloc[row_positions]

        # Detect outliers
        dataset_chauvenet = mark_outliers_chauvenet(label_data, column)

        # Get the outlier mask
        outlier_mask = dataset_chauvenet[column + "_outlier"].to_numpy()

        # Replace outliers with NaN
        outlier_removed_df.iloc[
            row_positions[outlier_mask], outlier_removed_df.columns.get_loc(column)
        ] = np.nan


# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

# check if there is ectually nan values now
outlier_removed_df.info()

outlier_removed_df.to_pickle("../../data/interim/02_outlier_removed.pkl")
