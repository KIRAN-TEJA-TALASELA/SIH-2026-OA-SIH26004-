import pandas as pd
import numpy as np
import os
import glob


INPUT_FOLDER = "Sensor_Data"
OUTPUT_FILE = "MASTER_SENSOR_DATASET.csv"


columns = [
    "Time_ms",
    "Raw_ADC",
    "Baseline",
    "Change",
    "Flexion_Index",
    "Flexion_Level"
]



def extract_features(file_path):


    df = pd.read_csv(
        file_path,
        header=None,
        names=columns
    )


    numeric_columns = [
        "Time_ms",
        "Raw_ADC",
        "Baseline",
        "Change",
        "Flexion_Index"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = df.dropna(
        subset=[
            "Time_ms",
            "Flexion_Index"
        ]
    ).reset_index(drop=True)


    df = df.sort_values(
        "Time_ms"
    ).reset_index(drop=True)

    # If there is not enough data, skip the file
    if len(df) < 2:
        return None


    time_seconds = (
        df["Time_ms"].to_numpy() / 1000.0
    )

    flexion = (
        df["Flexion_Index"].to_numpy()
    )


    mean_flexion = np.mean(flexion)

    median_flexion = np.median(flexion)

    min_flexion = np.min(flexion)

    max_flexion = np.max(flexion)

    std_flexion = np.std(
        flexion,
        ddof=1
    )

    range_flexion = (
        max_flexion -
        min_flexion
    )


    q25 = np.percentile(
        flexion,
        25
    )

    q75 = np.percentile(
        flexion,
        75
    )

    # Median Absolute Deviation

    mad = np.median(
        np.abs(
            flexion -
            np.median(flexion)
        )
    )


    duration_seconds = (
        time_seconds[-1] -
        time_seconds[0]
    )


    dt = np.diff(
        time_seconds
    )

    dflexion = np.diff(
        flexion
    )

    valid = dt > 0

    dt_valid = dt[valid]

    dflexion_valid = dflexion[valid]


    velocity = (
        dflexion_valid /
        dt_valid
    )

    absolute_velocity = np.abs(
        velocity
    )

    mean_absolute_velocity = np.mean(
        absolute_velocity
    )

    max_absolute_velocity = np.max(
        absolute_velocity
    )

    max_flexion_rate = np.max(
        velocity
    )

    max_extension_rate = np.min(
        velocity
    )


    interval_flexion = flexion[:-1]

    stable_time = np.sum(
        dt[
            interval_flexion < 2
        ]
    )

    low_time = np.sum(
        dt[
            (interval_flexion >= 2) &
            (interval_flexion < 5)
        ]
    )

    moderate_time = np.sum(
        dt[
            (interval_flexion >= 5) &
            (interval_flexion < 10)
        ]
    )

    high_time = np.sum(
        dt[
            interval_flexion >= 10
        ]
    )


    active = flexion >= 2

    movement_episodes = 0

    if active[0]:
        movement_episodes = 1

    movement_episodes += np.sum(
        active[1:] &
        ~active[:-1]
    )



    auc_flexion = np.trapezoid(
        flexion,
        time_seconds
    )


    features = {

        "Samples":
            len(df),

        "Duration_s":
            duration_seconds,

        "Mean_Flexion_Index":
            mean_flexion,

        "Median_Flexion_Index":
            median_flexion,

        "Min_Flexion_Index":
            min_flexion,

        "Max_Flexion_Index":
            max_flexion,

        "Flexion_Index_STD":
            std_flexion,

        "Flexion_Index_Range":
            range_flexion,

        "Flexion_Index_Q25":
            q25,

        "Flexion_Index_Q75":
            q75,

        "Flexion_Index_MAD":
            mad,

        "Time_STABLE_s":
            stable_time,

        "Time_LOW_s":
            low_time,

        "Time_MODERATE_s":
            moderate_time,

        "Time_HIGH_s":
            high_time,

        "Movement_Episodes":
            int(movement_episodes),

        "Mean_Absolute_Velocity":
            mean_absolute_velocity,

        "Max_Absolute_Velocity":
            max_absolute_velocity,

        "Max_Flexion_Rate":
            max_flexion_rate,

        "Max_Extension_Rate":
            max_extension_rate,

        "AUC_Flexion_Index":
            auc_flexion
    }

    return features



def process_folder(folder_path):

    # Find every CSV file
    csv_files = glob.glob(
        os.path.join(
            folder_path,
            "*.csv"
        )
    )

    if len(csv_files) == 0:

        print(
            "No CSV files found in:",
            folder_path
        )

        return

    print(
        f"\nFound {len(csv_files)} CSV files."
    )

    all_features = []


    for file_path in csv_files:


        filename = os.path.basename(
            file_path
        )

        patient_id = os.path.splitext(
            filename
        )[0]

        print(
            f"\nProcessing: {filename}"
        )


        try:

            features = extract_features(
                file_path
            )

            if features is None:

                print(
                    "Skipped - insufficient data."
                )

                continue


            features = {
                "Patient_ID": patient_id,
                **features
            }

            all_features.append(
                features
            )

            print(
                "Feature extraction successful."
            )

        except Exception as e:

            print(
                f"ERROR processing {filename}:"
            )

            print(e)



    if len(all_features) == 0:

        print(
            "\nNo valid files were processed."
        )

        return

    master_df = pd.DataFrame(
        all_features
    )



    master_df = master_df.sort_values(
        "Patient_ID"
    ).reset_index(drop=True)



    master_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\n")
    print("=" * 60)
    print("MASTER SENSOR DATASET CREATED")
    print("=" * 60)

    print(
        "\nNumber of patients processed:",
        len(master_df)
    )

    print(
        "\nOutput file:"
    )

    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    print(
        "\nMaster dataset shape:",
        master_df.shape
    )

    print(
        "\nMaster dataset:"
    )

    print(
        master_df
    )


if __name__ == "__main__":

    process_folder(
        INPUT_FOLDER
    )