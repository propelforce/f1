# %%
import pandas as pd
import numpy as np
import glob
import os

# %%
main_directory = os.getcwd()
raw_files = r'data\raw\*.csv'
    
csv_files_path = os.path.join(main_directory, raw_files)
print(csv_files_path)
csv_files = glob.glob(csv_files_path)


default = "df_"
for csv_file in csv_files:
    filename = os.path.basename(csv_file).split('.')[0]
    globals()[default + filename] = pd.read_csv(csv_file)
    print(f'{default + filename:<30}: {csv_file}')

# %%
# Merge Driver & Standings on Driver ID
df_temp1 = pd.merge(
    df_drivers[['driverId', 'dob']],
    df_driver_standings[['driverId', 'raceId', 'points', 'position', 'wins']],
    on='driverId',
    how='inner'
).rename(columns={'position': 'driver_season_position', 'points': 'driver_race_points'})

# Merge temp1 and race on race id
df_temp2 = pd.merge(
    df_temp1,
    df_races[['raceId', 'circuitId', 'date', 'year']],
    on='raceId',
    how='inner'
)

# Merge temp2 & qualifying results on Race ID and Driver ID
df_temp3 = pd.merge(
    df_temp2,
    df_qualifying[['qualifyId', 'raceId', 'driverId', 'constructorId', 'position']],
    on=['driverId', 'raceId'],
    how='inner'
).rename(columns={'position': 'qualifying_position'})


# Merge temp3 & final results on Race ID and Driver ID
df_temp4 = pd.merge(
    df_temp3,
    df_results[['resultId', 'raceId', 'driverId', 'positionOrder','statusId']],
    on=['driverId', 'raceId'],
    how='inner'
).rename(columns={'positionOrder': 'final_position','statusId':'final_status'})

# Merge temp4 & circuit on CircuitId
df_temp5 = pd.merge(
    df_temp4,
    df_circuits[['circuitId', 'alt']],
    on='circuitId',
    how='inner'
)

df_temp5['alt'] = pd.to_numeric(df_temp5['alt'], errors='coerce')

# Merge temp5 & constructor_results on RaceId & constructorId
df_temp6 = pd.merge(
    df_temp5,
    df_constructor_results[['constructorId', 'raceId', 'points']],
    on=['raceId', 'constructorId'],
    how='inner'
).rename(columns={'points': 'constructor_race_points'})

# Merge temp6 & constructor_standings on RaceId & constructorId
df_temp7 = pd.merge(
    df_temp6,
    df_constructor_standings[['constructorId', 'raceId', 'points']],
    on=['raceId', 'constructorId'],
    how='inner'
).rename(columns={'points': 'constructor_season_points'})


# Pit-stop aggregates
df_pit_stops_agg = df_pit_stops.groupby(['raceId', 'driverId']).agg({'stop': 'sum', 'milliseconds': 'sum'}).reset_index()
df_pit_stops_agg['avg_stop_time'] = df_pit_stops_agg.milliseconds / df_pit_stops_agg.stop

# Merge temp7 and pit-stop aggregates on raceId & driverId
df_temp8 = pd.merge(
    df_temp7,
    df_pit_stops_agg[['raceId', 'driverId', 'stop', 'avg_stop_time']],
    on=['raceId', 'driverId'],
    how='left'
)

# Lap-time aggregates
df_lap_times_agg = df_lap_times.groupby(['raceId', 'driverId']).agg({'position': 'median', 'milliseconds': 'mean'}).reset_index()
df_lap_times_agg.rename(columns={'position': 'driver_median_race_lap_position', 'milliseconds': 'avg_lap_time'}, inplace=True)

# Merge temp8 and lap-time aggregates on raceId & driverId
df_temp9 = pd.merge(
    df_temp8,
    df_lap_times_agg[['raceId', 'driverId', 'driver_median_race_lap_position', 'avg_lap_time']],
    on=['raceId', 'driverId'],
    how='left'
)

# Extract year, month, day from dates
df_temp9['dob_year'] = pd.to_datetime(df_temp9['dob']).dt.year
df_temp9['race_month'] = pd.to_datetime(df_temp9['date']).dt.month
df_temp9['race_day'] = pd.to_datetime(df_temp9['date']).dt.day
df_temp9.rename(columns={'date': 'race_date'}, inplace=True)

# Driver's age
df_temp9['driver_age'] = df_temp9['year'] - df_temp9['dob_year']

#Target variable
df_temp9['final_position'] = pd.to_numeric(df_temp9['final_position'], errors='coerce')
df_temp9['final_position'] = np.where(df_temp9['final_position']> 0,df_temp9['final_position'],0)

df_base_dataset = df_temp9[['race_date','driverId','driver_age','circuitId','alt','raceId','year','race_month','race_day','qualifyId','qualifying_position','wins','driver_race_points','constructorId','constructor_race_points','driver_season_position','constructor_season_points','final_position','final_status']].sort_values(by=['race_date','driverId']).reset_index(drop=True)


# %%
df_base_dataset_features = df_base_dataset.copy()

df_base_dataset_features = df_base_dataset_features.reset_index(drop=True)

df_base_dataset_features.head()

# %%
lag_column_name = 'qualify_pos_minus1'
df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['qualifying_position'].shift(1)

# %%
df_base_dataset_features = df_base_dataset.copy()

for i in range(1, 25):
    
    lag_column_name = 'qualify_pos_minus' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['qualifying_position'].shift(i)

    lag_column_name = 'final_pos_minus' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['final_position'].shift(i)

    lag_column_name = 'driver_race_points' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['driver_race_points'].shift(i)
    
    lag_column_name = 'wins' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['wins'].shift(i)
    
    lag_column_name = 'final_status' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['final_status'].shift(i)
    
    lag_column_name = 'alt' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['alt'].shift(i)
    
    lag_column_name = 'constructorId' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['constructorId'].shift(i)

    lag_column_name = 'constructor_race_points' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['constructor_race_points'].shift(i)

    lag_column_name = 'constructor_season_points' + str(i)
    df_base_dataset_features[lag_column_name] = df_base_dataset_features.groupby('driverId')['constructor_season_points'].shift(i)

df_base_dataset_features = df_base_dataset_features.sort_values(by=['race_date','driverId']).reset_index(drop=True)


# %%
df_base_dataset.to_csv(r'C:\Users\eswar\f1\data\interim\base_dataset.csv',header=True,index=False)
df_base_dataset_features.to_csv(r'C:\Users\eswar\f1\data\interim\base_dataset_features.csv',header=True,index=False)


