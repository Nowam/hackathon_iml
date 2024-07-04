import pickle

import numpy as np
import pandas as pd
from hackathon_code.preprocess.utils import haversine


def add_density(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates to get unique station entries
    data = df.drop_duplicates('station_id').copy()

    # Extract latitude and longitude columns
    latitudes = data['latitude']
    longitudes = data['longitude']

    # Perform KDE
    values = np.vstack([longitudes, latitudes])
    # Open a file and use dump()
    with open('hackathon_code/data/stations_density.pkl', 'rb') as file:
        # A new file will be created
        kde = pickle.load(file)

        # with open('stations_density.pkl', 'rb') as file:
    #     kde = pickle.load(file)
    kde_values = kde(values).T

    # Use .loc to set the density values explicitly
    data.loc[:, 'density'] = kde_values

    # Merge the KDE values back to the original DataFrame based on 'station_id'
    df = df.merge(data[['station_id', 'density']], on='station_id', how='left')

    return df


def count_lines_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, count how many different bus lines stop at that station.
    :param df: DataFrame containing the bus schedule with columns ['station_id',
    'line_id']
    :return: DataFrame with an additional column 'line_count' indicating the count of
    unique lines per station
    """
    # Group by 'station_id' and count unique 'line_id' values
    line_counts = df.groupby('station_id')['line_id'].nunique().reset_index()

    # Rename the column for clarity
    line_counts.columns = ['station_id', 'line_count']

    # Merge the counts back to the original DataFrame
    df = df.merge(line_counts, on='station_id', how='left')

    return df


def load_data(file_path='train_bus_schedule.csv'):
    df = pd.read_csv(file_path, encoding="ISO-8859-8")
    return df


# Calculate the time difference for previous station
def calculate_time_diff(arrival_time, previous_arrival_time):
    if pd.isnull(previous_arrival_time):
        return np.nan
    diff = arrival_time - previous_arrival_time
    if diff.total_seconds() >= 0:
        return diff.total_seconds()
    else:
        return (pd.Timedelta('1 days') + diff).total_seconds()


def add_time_from_next_prev(df: pd.DataFrame) -> pd.DataFrame:
    # Get the arrival time of the previous station
    df['previous_arrival_time'] = df.groupby('trip_id_unique')['arrival_time'].shift(1)
    df['next_arrival_time'] = df.groupby('trip_id_unique')['arrival_time'].shift(-1)

    # Calculate the time difference in seconds for the previous and next stations
    df['time_from_prev_station'] = df.apply(
        lambda row: calculate_time_diff(row['arrival_time'],
                                        row['previous_arrival_time']), axis=1
    )
    df['time_to_next_station'] = df.apply(
        lambda row: calculate_time_diff(row['next_arrival_time'], row['arrival_time']),
        axis=1
    )
    return df


def add_distance_from_prev_next(df: pd.DataFrame) -> pd.DataFrame:
    # Sort the dataframe by trip_id_unique and station_index
    df = df.sort_values(by=['trip_id_unique', 'station_index'])

    # Calculate distances between consecutive stations within the same trip
    df['prev_latitude'] = df.groupby('trip_id_unique')['latitude'].shift()
    df['prev_longitude'] = df.groupby('trip_id_unique')['longitude'].shift()
    df['next_latitude'] = df.groupby('trip_id_unique')['latitude'].shift(-1)
    df['next_longitude'] = df.groupby('trip_id_unique')['longitude'].shift(-1)

    df['distance_from_prev'] = df.apply(
        lambda row: haversine(row['prev_latitude'], row['prev_longitude'],
                              row['latitude'], row['longitude'])
        if pd.notnull(row['prev_latitude']) else 0, axis=1) * 1000

    df['distance_from_next'] = df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], row['next_latitude'],
                              row['next_longitude'])
        if pd.notnull(row['prev_latitude']) else 0, axis=1) * 1000
    return df


def map_time_period(hour):
    if 7 <= hour < 11:
        return 'Morning'
    elif 11 <= hour < 15:
        return 'Noon'
    elif 15 <= hour < 19:
        return 'Afternoon'
    elif 19 <= hour < 23:
        return 'Evening'
    else:
        return 'Night'


def add_time_category(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = (df['arrival_time'].dt.total_seconds() / 60 / 60).astype(int)
    df['start_hour'] = \
    df.groupby('trip_id_unique')['arrival_time'].transform('min').dt.components['hours']
    df['time_period'] = df['start_hour'].apply(map_time_period)
    return df


def add_total_trip_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['trip_id_unique', 'station_index'])
    trip_distances = df.groupby('trip_id_unique')[
        'distance_from_prev'].sum().reset_index()
    trip_distances.columns = ['trip_id_unique', 'total_distance']
    # Merge the total distances back into the original dataframe
    df = df.merge(trip_distances, on='trip_id_unique')
    return df


def add_lines_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=['line_id', 'direction'])


def add_max_line_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every line and direction, find the maximum line count and add it to the DataFrame.
    :param df: DataFrame containing the bus schedule with columns ['line_id',
    'direction', 'station_id']
    :return: DataFrame with an additional column 'max_line_count' indicating the
    maximum line count for each line and direction
    """

    # Calculate the maximum line count for each line_id and direction
    max_line_counts = df.groupby(['line_id', 'direction'])[
        'line_count'].max().reset_index()
    max_line_counts.columns = ['line_id', 'direction', 'max_line_count']

    # Merge the maximum line count back to the original DataFrame
    df = df.merge(max_line_counts, on=['line_id', 'direction'], how='left')

    return df


def add_max_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every line and direction, find the station with the maximum density and add the
    distance to this station.
    :param df: DataFrame containing the bus schedule with columns ['line_id',
    'direction', 'station_id', 'latitude', 'longitude', 'density']
    :return: DataFrame with an additional column 'distance_to_max_density_station'
    indicating the distance to the station with the maximum density
    """

    # Calculate the maximum density for each line_id and direction
    max_density_info = df.loc[df.groupby(['line_id', 'direction'])['density'].idxmax()]
    max_density_info = max_density_info[
        ['line_id', 'direction', 'station_id', 'station_index', 'latitude', 'longitude',
         'density']]
    max_density_info.columns = ['line_id', 'direction', 'max_density_station_id',
                                'max_stat_index', 'max_density_latitude',
                                'max_density_longitude', 'max_density']

    # Merge the maximum density info back to the original DataFrame
    df = df.merge(max_density_info, on=['line_id', 'direction'], how='left')

    # Calculate the distance to the station with the maximum density
    df['stations_to_max_density_station'] = df.apply(
        lambda row: row["station_index"] - row['max_stat_index'],
        axis=1
    )

    # Drop temporary columns used for calculation
    df = df.drop(
        columns=['max_density_station_id', 'max_stat_index', 'max_density_latitude',
                 'max_density_longitude'])

    return df


def add_drive_frac(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every line and direction, find the maximum line count and add it to the DataFrame.
    :param df: DataFrame containing the bus schedule with columns ['line_id',
    'direction', 'station_id']
    :return: DataFrame with an additional column 'max_line_count' indicating the
    maximum line count for each line and direction
    """

    # Calculate the maximum line count for each line_id and direction
    max_line_counts = df.groupby(['line_id', 'direction'])[
        'station_index'].max().reset_index()
    max_line_counts.columns = ['line_id', 'direction', 'num_stations']

    # Merge the maximum line count back to the original DataFrame
    df = df.merge(max_line_counts, on=['line_id', 'direction'], how='left')
    df['drive_frac'] = df['station_index'] / df['num_stations']

    return df


def save_as_csv(df: pd.DataFrame):
    df.to_csv('../../../../passengers_up_preprocessed.csv', encoding="ISO-8859-8",
              index=False)


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['arrival_time'] = pd.to_timedelta(df['arrival_time'])
    df = add_time_category(df)
    # Sort by 'trip_id' and 'station_index'
    df = add_distance_from_prev_next(df)
    # Group by trip_id_unique and sum distances
    # df = add_total_trip_distance(df)
    df = add_density(df)
    df = count_lines_per_station(df)

    df = add_total_trip_distance(df)
    df = add_max_density(df)
    df = add_drive_frac(df)

    df = df.drop(['alternative', 'door_closing_time', 'passengers_continue_menupach',
                  'prev_latitude', 'prev_longitude', 'next_latitude', 'next_longitude',
                  'latitude', 'longitude', 'station_name', 'arrival_time',
                  'mekadem_nipuach_luz', 'trip_id', 'line_id',
                  'part'
                     , 'station_id'], axis=1)

    df = pd.get_dummies(df, columns=['direction', "cluster", 'time_period'])

    return df


def adjust_rows(train_and_test, df_train, df_test) -> pd.DataFrame:
    # Remove negative values
    df_train['passengers_continue'] = abs(df_train['passengers_continue'])
    df_test['passengers_continue'] = abs(df_test['passengers_continue'])
    # distance & time from prev outliers
    # distance & time from next outliers
    columns = ['total_distance', 'passengers_continue']
    for col in columns:
        df_train, df_test = handle_outlires(train_and_test, df_train, df_test, col)

    return df_train, df_test


def handle_outlires(train_and_test, df_train: pd.DataFrame, df_test: pd.DataFrame,
                    field) -> pd.DataFrame:
    # Calculate mean and standard deviation of trip_duration_minutes
    mean_value = train_and_test[field].mean()
    std_value = train_and_test[field].std()

    # Filter out trips where duration is within 2 standard deviations from the mean
    dt_train = df_train[(df_train[field] >= mean_value - 2 * std_value) &
                        (df_train[field] <= mean_value + 2 * std_value)]

    # Identify outliers (values not within 2 standard deviations from the mean)
    is_outlier = ~((df_test[field] >= mean_value - 2 * std_value) & (
                df_test[field] <= mean_value + 2 * std_value))
    # Replace outliers with the mean value
    df_test.loc[is_outlier, field] = mean_value
    return dt_train, df_test


def create_db():
    df = load_data()
    df = preprocessing(df)
    save_as_csv(df)


if __name__ == '__main__':
    create_db()
    df = load_data('../../../../passengers_up_preprocessed.csv')

    print(df.head(10))
