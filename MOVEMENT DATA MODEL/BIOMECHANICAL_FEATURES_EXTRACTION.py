import pandas as pd
import numpy as np
from pathlib import Path



FLEX_FILE = r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\MOVEMENT DATA MODEL\flex_data_processed.csv"

MPU_FILE = r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\MOVEMENT DATA MODEL\mpu_data_single_person.csv"

OUTPUT_DIR = Path(
    r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\MOVEMENT DATA MODEL"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




print("\n================================================")
print("BIOMECHANICAL FEATURE EXTRACTION")
print("================================================")

print("\nLoading Flex data...")

flex = pd.read_csv(FLEX_FILE)

print("Flex shape:", flex.shape)

print("\nLoading MPU data...")

mpu = pd.read_csv(MPU_FILE)

print("MPU shape:", mpu.shape)



if len(flex) != 1:
    raise ValueError(
        f"Expected one Flex row, but found {len(flex)} rows."
    )

if len(mpu) != 1:
    raise ValueError(
        f"Expected one MPU row, but found {len(mpu)} rows."
    )



flex_row = flex.iloc[0]

mpu_row = mpu.iloc[0]



required_flex = [
    "Samples",
    "Duration_s",
    "Mean_Flexion_Index",
    "Median_Flexion_Index",
    "Min_Flexion_Index",
    "Max_Flexion_Index",
    "Flexion_Index_STD",
    "Flexion_Index_Range",
    "Flexion_Index_Q25",
    "Flexion_Index_Q75",
    "Flexion_Index_MAD",
    "Time_STABLE_s",
    "Time_LOW_s",
    "Time_MODERATE_s",
    "Time_HIGH_s",
    "Movement_Episodes",
    "Mean_Absolute_Velocity",
    "Max_Absolute_Velocity",
    "Max_Flexion_Rate",
    "Max_Extension_Rate",
    "AUC_Flexion_Index"
]

missing_flex = [
    col for col in required_flex
    if col not in flex.columns
]

if missing_flex:
    raise ValueError(
        f"Missing Flex columns: {missing_flex}"
    )




required_mpu = [
    "Number_of_Samples",
    "Recording_Duration_s",
    "Estimated_Sampling_Rate_Hz",

    "Accel_Mean_g",
    "Accel_STD_g",
    "Accel_Min_g",
    "Accel_Max_g",
    "Accel_Range_g",
    "Accel_RMS_g",

    "Accel_X_STD_g",
    "Accel_Y_STD_g",
    "Accel_Z_STD_g",

    "Gyro_Mean_dps",
    "Gyro_STD_dps",
    "Gyro_Min_dps",
    "Gyro_Max_dps",
    "Gyro_Range_dps",
    "Gyro_RMS_dps",

    "Gyro_X_STD_dps",
    "Gyro_Y_STD_dps",
    "Gyro_Z_STD_dps",

    "Pitch_Mean_deg",
    "Pitch_STD_deg",
    "Pitch_Min_deg",
    "Pitch_Max_deg",
    "Pitch_Range_deg",

    "Roll_Mean_deg",
    "Roll_STD_deg",
    "Roll_Min_deg",
    "Roll_Max_deg",
    "Roll_Range_deg",

    "Movement_Samples",
    "Movement_Percentage",

    "Max_Acceleration_g",
    "Max_Angular_Velocity_dps"
]

missing_mpu = [
    col for col in required_mpu
    if col not in mpu.columns
]

if missing_mpu:
    raise ValueError(
        f"Missing MPU columns: {missing_mpu}"
    )




flex_duration = float(
    flex_row["Duration_s"]
)

mpu_duration = float(
    mpu_row["Recording_Duration_s"]
)

duration_difference = abs(
    flex_duration - mpu_duration
)

duration_ratio = (
    mpu_duration / flex_duration
    if flex_duration > 0
    else np.nan
)




flex_mean = float(
    flex_row["Mean_Flexion_Index"]
)

flex_median = float(
    flex_row["Median_Flexion_Index"]
)

flex_min = float(
    flex_row["Min_Flexion_Index"]
)

flex_max = float(
    flex_row["Max_Flexion_Index"]
)

flex_range = float(
    flex_row["Flexion_Index_Range"]
)

flex_std = float(
    flex_row["Flexion_Index_STD"]
)

flex_q25 = float(
    flex_row["Flexion_Index_Q25"]
)

flex_q75 = float(
    flex_row["Flexion_Index_Q75"]
)

flex_mad = float(
    flex_row["Flexion_Index_MAD"]
)




if abs(flex_mean) > 1e-9:

    flex_cv = (
        flex_std / abs(flex_mean)
    )

else:

    flex_cv = np.nan




flex_iqr = (
    flex_q75 - flex_q25
)




mean_abs_velocity = float(
    flex_row["Mean_Absolute_Velocity"]
)

max_abs_velocity = float(
    flex_row["Max_Absolute_Velocity"]
)

max_flexion_rate = float(
    flex_row["Max_Flexion_Rate"]
)

max_extension_rate = float(
    flex_row["Max_Extension_Rate"]
)





max_extension_rate_abs = abs(
    max_extension_rate
)




if max_flexion_rate > 0:

    velocity_ratio = (
        max_extension_rate_abs /
        max_flexion_rate
    )

else:

    velocity_ratio = np.nan




movement_episodes = int(
    flex_row["Movement_Episodes"]
)

flex_samples = int(
    flex_row["Samples"]
)

flex_duration_safe = max(
    flex_duration,
    1e-9
)




movement_episode_frequency = (
    movement_episodes /
    flex_duration_safe
)


# Episodes per minute

movement_episodes_per_minute = (
    movement_episode_frequency * 60
)




if movement_episodes > 0:

    mean_episode_interval = (
        flex_duration /
        movement_episodes
    )

else:

    mean_episode_interval = np.nan




time_stable = float(
    flex_row["Time_STABLE_s"]
)

time_low = float(
    flex_row["Time_LOW_s"]
)

time_moderate = float(
    flex_row["Time_MODERATE_s"]
)

time_high = float(
    flex_row["Time_HIGH_s"]
)


# Total movement-associated time

active_time = (
    time_low +
    time_moderate +
    time_high
)


# Percentage of recording spent active

active_time_percentage = (
    active_time /
    flex_duration_safe
) * 100


# High-movement percentage

high_movement_percentage = (
    time_high /
    flex_duration_safe
) * 100



accel_mean = float(
    mpu_row["Accel_Mean_g"]
)

accel_std = float(
    mpu_row["Accel_STD_g"]
)

accel_range = float(
    mpu_row["Accel_Range_g"]
)

accel_rms = float(
    mpu_row["Accel_RMS_g"]
)

max_acceleration = float(
    mpu_row["Max_Acceleration_g"]
)




if abs(accel_mean) > 1e-9:

    accel_cv = (
        accel_std /
        abs(accel_mean)
    )

else:

    accel_cv = np.nan




gyro_mean = float(
    mpu_row["Gyro_Mean_dps"]
)

gyro_std = float(
    mpu_row["Gyro_STD_dps"]
)

gyro_range = float(
    mpu_row["Gyro_Range_dps"]
)

gyro_rms = float(
    mpu_row["Gyro_RMS_dps"]
)

max_angular_velocity = float(
    mpu_row["Max_Angular_Velocity_dps"]
)




if abs(gyro_mean) > 1e-9:

    gyro_cv = (
        gyro_std /
        abs(gyro_mean)
    )

else:

    gyro_cv = np.nan




pitch_range = float(
    mpu_row["Pitch_Range_deg"]
)

roll_range = float(
    mpu_row["Roll_Range_deg"]
)

pitch_std = float(
    mpu_row["Pitch_STD_deg"]
)

roll_std = float(
    mpu_row["Roll_STD_deg"]
)


# Combined orientation variability

orientation_variability = (
    np.sqrt(
        pitch_std ** 2 +
        roll_std ** 2
    )
)


# Combined orientation range

orientation_range = (
    np.sqrt(
        pitch_range ** 2 +
        roll_range ** 2
    )
)




mpu_movement_samples = int(
    mpu_row["Movement_Samples"]
)

mpu_movement_percentage = float(
    mpu_row["Movement_Percentage"]
)



# Ratio of peak angular velocity to mean angular velocity

if gyro_mean > 0:

    angular_velocity_peak_ratio = (
        max_angular_velocity /
        gyro_mean
    )

else:

    angular_velocity_peak_ratio = np.nan


# Ratio of peak acceleration to RMS acceleration

if accel_rms > 0:

    acceleration_peak_ratio = (
        max_acceleration /
        accel_rms
    )

else:

    acceleration_peak_ratio = np.nan




# These combine Flex and MPU summaries.
#
# IMPORTANT:
# They should be interpreted cautiously because the
# recordings have different durations.



if mpu_duration > 0:

    flex_episodes_per_mpu_minute = (
        movement_episodes /
        mpu_duration
    ) * 60

else:

    flex_episodes_per_mpu_minute = np.nan




if max_angular_velocity > 0:

    flex_to_gyro_peak_ratio = (
        max_abs_velocity /
        max_angular_velocity
    )

else:

    flex_to_gyro_peak_ratio = np.nan




biomechanical_features = {



    "Flex_Recording_Duration_s":
        flex_duration,

    "MPU_Recording_Duration_s":
        mpu_duration,

    "Recording_Duration_Difference_s":
        duration_difference,

    "MPU_to_Flex_Duration_Ratio":
        duration_ratio,




    "Flexion_Index_Mean":
        flex_mean,

    "Flexion_Index_Median":
        flex_median,

    "Flexion_Index_Min":
        flex_min,

    "Flexion_Index_Max":
        flex_max,

    "Flexion_Index_Range":
        flex_range,

    "Flexion_Index_STD":
        flex_std,

    "Flexion_Index_IQR":
        flex_iqr,

    "Flexion_Index_MAD":
        flex_mad,

    "Flexion_Index_CV":
        flex_cv,




    "Mean_Flexion_Absolute_Velocity":
        mean_abs_velocity,

    "Peak_Flexion_Absolute_Velocity":
        max_abs_velocity,

    "Peak_Flexion_Rate":
        max_flexion_rate,

    "Peak_Extension_Rate":
        max_extension_rate,

    "Peak_Extension_Rate_Absolute":
        max_extension_rate_abs,

    "Extension_to_Flexion_Velocity_Ratio":
        velocity_ratio,




    "Movement_Episodes":
        movement_episodes,

    "Movement_Episodes_per_Minute":
        movement_episodes_per_minute,

    "Mean_Time_Between_Movement_Episodes_s":
        mean_episode_interval,




    "Time_Stable_s":
        time_stable,

    "Time_Low_s":
        time_low,

    "Time_Moderate_s":
        time_moderate,

    "Time_High_s":
        time_high,

    "Active_Movement_Time_s":
        active_time,

    "Active_Movement_Time_Percentage":
        active_time_percentage,

    "High_Movement_Time_Percentage":
        high_movement_percentage,




    "Acceleration_Mean_g":
        accel_mean,

    "Acceleration_STD_g":
        accel_std,

    "Acceleration_Range_g":
        accel_range,

    "Acceleration_RMS_g":
        accel_rms,

    "Acceleration_CV":
        accel_cv,

    "Peak_Acceleration_g":
        max_acceleration,

    "Acceleration_Peak_to_RMS_Ratio":
        acceleration_peak_ratio,




    "Angular_Velocity_Mean_dps":
        gyro_mean,

    "Angular_Velocity_STD_dps":
        gyro_std,

    "Angular_Velocity_Range_dps":
        gyro_range,

    "Angular_Velocity_RMS_dps":
        gyro_rms,

    "Angular_Velocity_CV":
        gyro_cv,

    "Peak_Angular_Velocity_dps":
        max_angular_velocity,

    "Angular_Velocity_Peak_to_Mean_Ratio":
        angular_velocity_peak_ratio,




    "Pitch_ROM_deg":
        pitch_range,

    "Roll_ROM_deg":
        roll_range,

    "Pitch_STD_deg":
        pitch_std,

    "Roll_STD_deg":
        roll_std,

    "Combined_Orientation_Variability":
        orientation_variability,

    "Combined_Orientation_Range_deg":
        orientation_range,




    "MPU_Movement_Samples":
        mpu_movement_samples,

    "MPU_Movement_Percentage":
        mpu_movement_percentage,




    "Flex_Episodes_per_MPU_Minute":
        flex_episodes_per_mpu_minute,

    "Flex_to_Gyro_Peak_Ratio":
        flex_to_gyro_peak_ratio
}




biomechanical_df = pd.DataFrame(
    [biomechanical_features]
)




biomechanical_df.insert(
    0,
    "Person_ID",
    "Person_001"
)



# Current files have different recording durations.
# Therefore they should NOT be treated as synchronized.

if duration_difference > 5:

    synchronization_status = (
        "NOT_SYNCHRONIZED_DURATION_MISMATCH"
    )

else:

    synchronization_status = (
        "DURATION_MATCH_ACCEPTABLE"
    )


biomechanical_df[
    "Sensor_Synchronization_Status"
] = synchronization_status




output_file = (
    OUTPUT_DIR /
    "biomechanical_features.csv"
)

biomechanical_df.to_csv(
    output_file,
    index=False
)


print("\n================================================")
print("BIOMECHANICAL FEATURE EXTRACTION COMPLETE")
print("================================================")

print("\nOutput file:")
print(output_file)

print("\nNumber of biomechanical features:")
print(
    len(biomechanical_df.columns) - 2
)

print("\nSensor recording durations:")

print(
    f"Flex: {flex_duration:.3f} seconds"
)

print(
    f"MPU : {mpu_duration:.3f} seconds"
)

print(
    f"Difference: {duration_difference:.3f} seconds"
)

print(
    "\nSynchronization status:",
    synchronization_status
)

print("\n================ EXTRACTED FEATURES ================")

print(
    biomechanical_df.T.to_string()
)

print("\n================================================")
print("DONE")
print("================================================")