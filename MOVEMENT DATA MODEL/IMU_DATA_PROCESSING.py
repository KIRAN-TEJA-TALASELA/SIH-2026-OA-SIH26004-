import pandas as pd
import numpy as np
from pathlib import Path


MPU_INPUT = r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\MOVEMENT DATA MODEL\mpu data.csv"

OUTPUT_DIR = Path(
    r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\MOVEMENT DATA MODEL"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




mpu = pd.read_csv(
    MPU_INPUT,
    header=None,
    names=[
        "Time_ms",
        "Accel_X",
        "Accel_Y",
        "Accel_Z",
        "Gyro_X",
        "Gyro_Y",
        "Gyro_Z"
    ]
)

print("\n================ RAW MPU DATA ================")
print("Shape:", mpu.shape)

print("\nColumns:")
print(mpu.columns.tolist())

print("\nFirst 5 rows:")
print(mpu.head())




numeric_columns = [
    "Time_ms",
    "Accel_X",
    "Accel_Y",
    "Accel_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z"
]

for col in numeric_columns:
    mpu[col] = pd.to_numeric(
        mpu[col],
        errors="coerce"
    )


# Remove invalid rows
mpu = mpu.dropna().reset_index(drop=True)




# Convert milliseconds to seconds

mpu["Time_s"] = mpu["Time_ms"] / 1000.0


# Sort according to time

mpu = mpu.sort_values(
    "Time_s"
).reset_index(drop=True)




# MPU6050 default sensitivity:
# ±2g = 16384 LSB/g

mpu["Accel_X_g"] = (
    mpu["Accel_X"] / 16384.0
)

mpu["Accel_Y_g"] = (
    mpu["Accel_Y"] / 16384.0
)

mpu["Accel_Z_g"] = (
    mpu["Accel_Z"] / 16384.0
)




# MPU6050 default sensitivity:
# ±250 degrees/sec = 131 LSB/(degree/sec)

mpu["Gyro_X_dps"] = (
    mpu["Gyro_X"] / 131.0
)

mpu["Gyro_Y_dps"] = (
    mpu["Gyro_Y"] / 131.0
)

mpu["Gyro_Z_dps"] = (
    mpu["Gyro_Z"] / 131.0
)





mpu["Accel_Magnitude_g"] = np.sqrt(
    mpu["Accel_X_g"] ** 2 +
    mpu["Accel_Y_g"] ** 2 +
    mpu["Accel_Z_g"] ** 2
)




mpu["Gyro_Magnitude_dps"] = np.sqrt(
    mpu["Gyro_X_dps"] ** 2 +
    mpu["Gyro_Y_dps"] ** 2 +
    mpu["Gyro_Z_dps"] ** 2
)




mpu["Accel_Pitch_deg"] = np.degrees(
    np.arctan2(
        mpu["Accel_X_g"],
        np.sqrt(
            mpu["Accel_Y_g"] ** 2 +
            mpu["Accel_Z_g"] ** 2
        )
    )
)




mpu["Accel_Roll_deg"] = np.degrees(
    np.arctan2(
        mpu["Accel_Y_g"],
        np.sqrt(
            mpu["Accel_X_g"] ** 2 +
            mpu["Accel_Z_g"] ** 2
        )
    )
)




WINDOW = 5

mpu["Accel_Magnitude_Smooth"] = (
    mpu["Accel_Magnitude_g"]
    .rolling(
        window=WINDOW,
        center=True,
        min_periods=1
    )
    .mean()
)

mpu["Gyro_Magnitude_Smooth"] = (
    mpu["Gyro_Magnitude_dps"]
    .rolling(
        window=WINDOW,
        center=True,
        min_periods=1
    )
    .mean()
)




required_columns = [
    "Time_s",
    "Accel_X_g",
    "Accel_Y_g",
    "Accel_Z_g",
    "Gyro_X_dps",
    "Gyro_Y_dps",
    "Gyro_Z_dps",
    "Accel_Magnitude_g",
    "Gyro_Magnitude_dps",
    "Accel_Pitch_deg",
    "Accel_Roll_deg",
    "Accel_Magnitude_Smooth",
    "Gyro_Magnitude_Smooth"
]

missing = [
    col
    for col in required_columns
    if col not in mpu.columns
]

if missing:
    raise ValueError(
        f"Missing processed columns: {missing}"
    )

print("\nAll processed columns are present.")




n_samples = len(mpu)

duration_s = (
    mpu["Time_s"].max()
    - mpu["Time_s"].min()
)

if duration_s > 0:
    sampling_rate = (
        n_samples / duration_s
    )
else:
    sampling_rate = np.nan




accel_mean = (
    mpu["Accel_Magnitude_g"].mean()
)

accel_std = (
    mpu["Accel_Magnitude_g"].std()
)

accel_min = (
    mpu["Accel_Magnitude_g"].min()
)

accel_max = (
    mpu["Accel_Magnitude_g"].max()
)

accel_range = (
    accel_max - accel_min
)

accel_rms = np.sqrt(
    np.mean(
        mpu["Accel_Magnitude_g"] ** 2
    )
)

accel_smooth_mean = (
    mpu["Accel_Magnitude_Smooth"].mean()
)

accel_smooth_std = (
    mpu["Accel_Magnitude_Smooth"].std()
)



accel_x_mean = mpu["Accel_X_g"].mean()
accel_y_mean = mpu["Accel_Y_g"].mean()
accel_z_mean = mpu["Accel_Z_g"].mean()

accel_x_std = mpu["Accel_X_g"].std()
accel_y_std = mpu["Accel_Y_g"].std()
accel_z_std = mpu["Accel_Z_g"].std()




gyro_mean = (
    mpu["Gyro_Magnitude_dps"].mean()
)

gyro_std = (
    mpu["Gyro_Magnitude_dps"].std()
)

gyro_min = (
    mpu["Gyro_Magnitude_dps"].min()
)

gyro_max = (
    mpu["Gyro_Magnitude_dps"].max()
)

gyro_range = (
    gyro_max - gyro_min
)

gyro_rms = np.sqrt(
    np.mean(
        mpu["Gyro_Magnitude_dps"] ** 2
    )
)

gyro_smooth_mean = (
    mpu["Gyro_Magnitude_Smooth"].mean()
)

gyro_smooth_std = (
    mpu["Gyro_Magnitude_Smooth"].std()
)




gyro_x_mean = mpu["Gyro_X_dps"].mean()
gyro_y_mean = mpu["Gyro_Y_dps"].mean()
gyro_z_mean = mpu["Gyro_Z_dps"].mean()

gyro_x_std = mpu["Gyro_X_dps"].std()
gyro_y_std = mpu["Gyro_Y_dps"].std()
gyro_z_std = mpu["Gyro_Z_dps"].std()




pitch_mean = (
    mpu["Accel_Pitch_deg"].mean()
)

pitch_std = (
    mpu["Accel_Pitch_deg"].std()
)

pitch_min = (
    mpu["Accel_Pitch_deg"].min()
)

pitch_max = (
    mpu["Accel_Pitch_deg"].max()
)

pitch_range = (
    pitch_max - pitch_min
)




roll_mean = (
    mpu["Accel_Roll_deg"].mean()
)

roll_std = (
    mpu["Accel_Roll_deg"].std()
)

roll_min = (
    mpu["Accel_Roll_deg"].min()
)

roll_max = (
    mpu["Accel_Roll_deg"].max()
)

roll_range = (
    roll_max - roll_min
)



gyro_threshold = 5.0

movement_samples = (
    mpu["Gyro_Magnitude_dps"]
    > gyro_threshold
).sum()

movement_percentage = (
    movement_samples / n_samples
) * 100



max_acceleration = (
    mpu["Accel_Magnitude_g"].max()
)

max_angular_velocity = (
    mpu["Gyro_Magnitude_dps"].max()
)




features = {

    "Number_of_Samples": n_samples,

    "Recording_Duration_s": duration_s,

    "Estimated_Sampling_Rate_Hz": sampling_rate,

    # -------------------------
    # Acceleration
    # -------------------------

    "Accel_Mean_g": accel_mean,

    "Accel_STD_g": accel_std,

    "Accel_Min_g": accel_min,

    "Accel_Max_g": accel_max,

    "Accel_Range_g": accel_range,

    "Accel_RMS_g": accel_rms,

    "Accel_Smooth_Mean_g": accel_smooth_mean,

    "Accel_Smooth_STD_g": accel_smooth_std,

    # -------------------------
    # Acceleration axes
    # -------------------------

    "Accel_X_Mean_g": accel_x_mean,

    "Accel_Y_Mean_g": accel_y_mean,

    "Accel_Z_Mean_g": accel_z_mean,

    "Accel_X_STD_g": accel_x_std,

    "Accel_Y_STD_g": accel_y_std,

    "Accel_Z_STD_g": accel_z_std,

    # -------------------------
    # Gyroscope
    # -------------------------

    "Gyro_Mean_dps": gyro_mean,

    "Gyro_STD_dps": gyro_std,

    "Gyro_Min_dps": gyro_min,

    "Gyro_Max_dps": gyro_max,

    "Gyro_Range_dps": gyro_range,

    "Gyro_RMS_dps": gyro_rms,

    "Gyro_Smooth_Mean_dps": gyro_smooth_mean,

    "Gyro_Smooth_STD_dps": gyro_smooth_std,

    # -------------------------
    # Gyroscope axes
    # -------------------------

    "Gyro_X_Mean_dps": gyro_x_mean,

    "Gyro_Y_Mean_dps": gyro_y_mean,

    "Gyro_Z_Mean_dps": gyro_z_mean,

    "Gyro_X_STD_dps": gyro_x_std,

    "Gyro_Y_STD_dps": gyro_y_std,

    "Gyro_Z_STD_dps": gyro_z_std,

    # -------------------------
    # Pitch
    # -------------------------

    "Pitch_Mean_deg": pitch_mean,

    "Pitch_STD_deg": pitch_std,

    "Pitch_Min_deg": pitch_min,

    "Pitch_Max_deg": pitch_max,

    "Pitch_Range_deg": pitch_range,

    # -------------------------
    # Roll
    # -------------------------

    "Roll_Mean_deg": roll_mean,

    "Roll_STD_deg": roll_std,

    "Roll_Min_deg": roll_min,

    "Roll_Max_deg": roll_max,

    "Roll_Range_deg": roll_range,

    # -------------------------
    # Movement
    # -------------------------

    "Movement_Samples": movement_samples,

    "Movement_Percentage": movement_percentage,

    "Max_Acceleration_g": max_acceleration,

    "Max_Angular_Velocity_dps": max_angular_velocity
}




features_df = pd.DataFrame([features])




output_file = (
    OUTPUT_DIR /
    "mpu_data_single_person.csv"
)

features_df.to_csv(
    output_file,
    index=False
)




print("\n================================================")
print("MPU SINGLE-PERSON FEATURE EXTRACTION COMPLETE")
print("================================================")

print("\nNumber of samples:", n_samples)

print(
    "Recording duration:",
    round(duration_s, 3),
    "seconds"
)

print(
    "Estimated sampling rate:",
    round(sampling_rate, 2),
    "Hz"
)

print(
    "\nOutput file created at:"
)

print(output_file)

print("\n================ FEATURES ================")

print(
    features_df.T.to_string()
)

print("\n================================================")
print("DONE")
print("================================================")