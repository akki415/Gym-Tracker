from glob import glob

import pandas as pd

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_1 = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion/"

f = files[0]

participant = f.split("-")[0].strip(data_path).strip("\\")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").removesuffix("_MetaWear_2019")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

accelerometer_df = pd.DataFrame()
gyroscope_df = pd.DataFrame()


for f in files:
    participant = f.split("-")[0].strip(data_path).strip("\\")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").removesuffix("_MetaWear_2019")

    # print(category)

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        accelerometer_df = pd.concat([accelerometer_df, df])

    if "Gyroscope" in f:
        gyroscope_df = pd.concat([gyroscope_df, df])


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

accelerometer_df.info()
accelerometer_df["epoch (ms)"]

accelerometer_df.index = pd.to_datetime(accelerometer_df["epoch (ms)"], unit="ms")
gyroscope_df.index = pd.to_datetime(gyroscope_df["epoch (ms)"], unit="ms")


accelerometer_df.drop(["epoch (ms)", "time (01:00)"], axis=1, inplace=True)
gyroscope_df.drop(["epoch (ms)", "time (01:00)"], axis=1, inplace=True)


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    accelerometer_df = pd.DataFrame()
    gyroscope_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].strip(data_path).strip("\\")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").removesuffix("_MetaWear_2019")

        # print(category)

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            accelerometer_df = pd.concat([accelerometer_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyroscope_df = pd.concat([gyroscope_df, df])

    accelerometer_df.index = pd.to_datetime(accelerometer_df["epoch (ms)"], unit="ms")
    gyroscope_df.index = pd.to_datetime(gyroscope_df["epoch (ms)"], unit="ms")

    accelerometer_df.drop(
        ["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True
    )
    gyroscope_df.drop(
        ["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True
    )

    return accelerometer_df, gyroscope_df


acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

acc_df.iloc[:, :3]
merged_data = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

merged_data.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

merged_data[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in merged_data.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)
data_resampled.info()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
