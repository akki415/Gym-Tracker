import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_transformation import LowPassFilter, PrincipalComponentAnalysis
from frequency_abstraction import FourierTransformation
from sklearn.cluster import KMeans
from temporal_abstraction import NumericalAbstraction

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outlier_removed.pkl")
predictor_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()
""" First i would like to interpulate the current dataset"""
for col in predictor_columns:
    df[col] = df[col].interpolate()
df.info()


""" defining some plotting parameters"""
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


""" Seeing the ploting for particular set"""
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]


# looping over all sets
for set in df["set"].unique():
    start = df[df["set"] == set].index[0]
    stop = df[df["set"] == set].index[-1]

    set_duration = stop - start
    df.loc[(df["set"] == set), "duration"] = set_duration.seconds

df.groupby(["category"])["duration"].mean()


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
"""
using to smoothning the lines or curves adjusting frequency
using butterflow filter
"""
butter_df = df.copy()
butter_filter = LowPassFilter()

sample_freq = 1000 / 200  # using ms as during make_dataset used 200ms
cutoff_freq = 1.5  # used to smooth curves

butter_df = butter_filter.low_pass_filter(
    data_table=butter_df,
    col="acc_y",
    sampling_frequency=sample_freq,
    cutoff_frequency=cutoff_freq,
    order=5,
)


butter_subset_df = butter_df[butter_df["set"] == 45]

fig, axes = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)
axes[0].plot(butter_subset_df["acc_y"].reset_index(drop=True), label="raw_acc_x")
axes[1].plot(
    butter_subset_df["acc_y_lowpass"].reset_index(drop=True), label="butter_acc_x"
)
axes[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True
)
axes[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True
)

# applying on all columns
for col in predictor_columns:
    butter_df = butter_filter.low_pass_filter(
        data_table=butter_df,
        col=col,
        sampling_frequency=sample_freq,
        cutoff_frequency=cutoff_freq,
        order=5,
    )
    # replacing raw data with butterflow lowpass filter cal one
    butter_df[col] = butter_df[col + "_lowpass"]
    del butter_df[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

pca_df = butter_df.copy()
PCA = PrincipalComponentAnalysis()

pca_values = PCA.determine_pc_explained_variance(
    data_table=pca_df, cols=predictor_columns
)
# lets plot these value in respect to no.of columns
# will use elbow mthd

plt.figure(figsize=(10, 20))
plt.plot(range(1, len(predictor_columns) + 1), pca_values)
plt.ylabel("PCA VALUES")
plt.xlabel("pca number")
plt.show()
""" 
as per plot er get to know that after 3 columns 
there is no significant decrease in curve givinhg a elbow
"""
optimal_num_columns = 3

pca_df = PCA.apply_pca(
    data_table=pca_df, cols=predictor_columns, number_comp=optimal_num_columns
)
pca_df_subset = pca_df[pca_df["set"] == 35]
pca_df_subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

squared_df = pca_df.copy()

acc_r = squared_df["acc_x"] ** 2 + squared_df["acc_y"] ** 2 + squared_df["acc_z"] ** 2
gyr_r = squared_df["gyr_x"] ** 2 + squared_df["gyr_y"] ** 2 + squared_df["gyr_z"] ** 2

squared_df["acc_r"] = np.sqrt(acc_r)
squared_df["gyr_r"] = np.sqrt(gyr_r)

subset_squared_df = squared_df[squared_df["set"] == 14]

subset_squared_df[["acc_r", "gyr_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
""" 
So idea behing using this is as it gives a moving window 
over the data and calculate mean using data of past 
steps say 5 , according to window_step

you can imagine it as a moving average like in share markt
"""
temporal_df = squared_df.copy()
num_abs = NumericalAbstraction()

all_predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

w_size = int(1000 / 200)

temporal_df_list = []
for s in temporal_df["set"].unique():
    subset = temporal_df[temporal_df["set"] == s].copy()

    for col in all_predictor_columns:
        subset = num_abs.abstract_numerical(
            data_table=subset,
            cols=[col],
            window_size=w_size,
            aggregation_function="mean",
        )
        subset = num_abs.abstract_numerical(
            data_table=subset,
            cols=[col],
            window_size=w_size,
            aggregation_function="std",
        )
    temporal_df_list.append(subset)
temporal_df = pd.concat(temporal_df_list)

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
""" Will be using Discrete Fourier Transformation (DFT)
the idea is that any sequence of measurements we perform can be 
represented by a combination of sinusoid functions 
with different frequencies

to put simply it will help in disecting frequency using
different frequencies 
"""

freq_df = temporal_df.copy().reset_index()
freq_abs = FourierTransformation()

sample_rate = int(1000 / 200)
win_size = int(2800 / 200)

freq_df_list = []
for s in freq_df["set"].unique():
    print("applying function on set ", s)
    subset = freq_df[freq_df["set"] == s].reset_index(drop=True).copy()
    subset = freq_abs.abstract_frequency(
        subset,
        cols=all_predictor_columns,
        window_size=win_size,
        sampling_rate=sample_rate,
    )
    freq_df_list.append(subset)

freq_df = pd.concat(freq_df_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
freq_df = freq_df.dropna()
freq_df = freq_df.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

cluster_df = freq_df.copy()
k_values = range(2, 10)
cluster_columns = ["acc_x", "acc_y", "acc_z"]

inertias_list = []

for k in k_values:
    subset = cluster_df[cluster_columns]
    kmean_clustering = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmean_clustering.fit_predict(subset)
    inertias_list.append(kmean_clustering.inertia_)

plt.figure(figsize=(10, 20))
plt.plot(k_values, inertias_list)
plt.xlabel("k")
plt.ylabel("inertia")
plt.show()
# as per above plot using elbow method k = 5 is optimal

kmean_clustering = KMeans(n_clusters=5, n_init=20, random_state=0)
subset_k = cluster_df[cluster_columns]
cluster_df["cluster"] = kmean_clustering.fit_predict(subset)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in cluster_df["cluster"].unique():
    subset = cluster_df[cluster_df["cluster"] == c]

    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=c,
    )

ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.set_title("K-Means Clustering (3D)")

plt.legend()
plt.tight_layout()
plt.show()


# clustering as per exercises/label

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in cluster_df["label"].unique():
    subset = cluster_df[cluster_df["label"] == l]

    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=c,
    )

ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.set_title("K-Means Clustering (3D)")

plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

cluster_df.to_pickle("../../data/interim/03_data_build_features.pkl")
