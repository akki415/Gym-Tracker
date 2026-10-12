import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_a_df = df[df["set"] == 1]
set_a_df.reset_index()

# plt.plot(set_a_df["acc_y"])
plt.plot(set_a_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

unique_exercises = df["label"].unique()

for exercises in unique_exercises:
    subset_label_df = df[df["label"] == exercises]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(subset_label_df[:100]["acc_y"].reset_index(drop=True), label=exercises)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
"""
Here the objective to compare weather there is a difference
in medium or heavy sets performed by an individual of a particular 
exercise 

"""
participants = df["participant"].unique()

for participant in participants:
    category_df = (
        df.query("label == 'bench'")
        .query("participant == @participant")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(20, 5))
    category_df.groupby(["category"])["acc_y"].plot()
    plt.title(f"Bench {participant}".title())
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participants = df["participant"].unique()


for exercise in unique_exercises:
    for participant in participants:
        category_df = (
            df.query("label == @exercise")
            .query("participant == @participant")
            .reset_index(drop=True)
        )

        if len(category_df) > 0:
            fig, ax = plt.subplots(figsize=(20, 5))
            # category_df.groupby(["category"])["acc_y"].plot()
            category_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)

            plt.title(
                f"{exercise} {participant}".title()
            )  # .title() again is used just to capitalise every first letter
            plt.legend()
            plt.show()


for exercise in unique_exercises:
    for participant in participants:
        category_df = (
            df.query("label == @exercise")
            .query("participant == @participant")
            .reset_index(drop=True)
        )

        if len(category_df) > 0:
            fig, ax = plt.subplots(figsize=(20, 5))
            # category_df.groupby(["category"])["acc_y"].plot()
            category_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)

            plt.title(
                f"{exercise} {participant}".title()
            )  # .title() again is used just to capitalise every first letter
            plt.legend()
            plt.show()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"

combine_plot_df = (
    df.query("label == @label").query("participant == @participant").reset_index()
)


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
