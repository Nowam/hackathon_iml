import pandas as pd
import numpy as np
from util import haversine

from scipy.stats import gaussian_kde
import numpy as np

import pandas as pd


def add_density(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates to get unique station entries
    data = df.drop_duplicates('station_id').copy()

    # Extract latitude and longitude columns
    latitudes = data['latitude']
    longitudes = data['longitude']

    # Perform KDE
    values = np.vstack([longitudes, latitudes])
    kde = gaussian_kde(values)
    kde_values = kde(values).T

    # Use .loc to set the density values explicitly
    data.loc[:, 'density'] = kde_values

    # Merge the KDE values back to the original DataFrame based on 'station_id'
    df = df.merge(data[['station_id', 'density']], on='station_id', how='left')

    return df


def count_lines_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, count how many different bus lines stop at that station.
    :param df: DataFrame containing the bus schedule with columns ['station_id', 'line_id']
    :return: DataFrame with an additional column 'line_count' indicating the count of unique lines per station
    """
    # Group by 'station_id' and count unique 'line_id' values
    line_counts = df.groupby('station_id')['line_id'].nunique().reset_index()

    # Rename the column for clarity
    line_counts.columns = ['station_id', 'line_count']

    # Merge the counts back to the original DataFrame
    df = df.merge(line_counts, on='station_id', how='left')

    return df


def remove_rows(df: pd.DataFrame) -> pd.DataFrame:
    pass


def calculate_time_to_next_station():
    pass


def load_data(file_path='train_bus_schedule.csv'):
    df = pd.read_csv(file_path, encoding="ISO-8859-8")
    return df


# Calculate the number of unique stations within 1 km radius for each unique station
def calculate_nearby_stations(unique_stations):
    def wrappper(station):
        lat1, lon1 = station['latitude'], station['longitude']
        distances = unique_stations.apply(lambda row: haversine(lat1, lon1, row['latitude'], row['longitude']), axis=1)
        return (distances <= 1).sum() - 1  # Subtract 1 to exclude the station itself
    return wrappper

def add_distance_from_prev_next(df: pd.DataFrame) -> pd.DataFrame:
    # Get the arrival time of the previous station
    df['previous_arrival_time'] = df.groupby('trip_id_unique')['arrival_time'].shift(1)
    df['next_arrival_time'] = df.groupby('trip_id_unique')['arrival_time'].shift(-1)

    # Calculate the time difference in seconds
    df['time_from_prev_station'] = (((df['arrival_time'] - df['previous_arrival_time']).dt.total_seconds())
                                    .apply(lambda x: max(0, x)))
    df['time_to_next_station'] = (((df['next_arrival_time'] - df['arrival_time']).dt.total_seconds())
                                    .apply(lambda x: max(0, x)))


    # Sort the dataframe by trip_id_unique and station_index
    df = df.sort_values(by=['trip_id_unique', 'station_index'])

    # Calculate distances between consecutive stations within the same trip
    df['prev_latitude'] = df.groupby('trip_id_unique')['latitude'].shift()
    df['prev_longitude'] = df.groupby('trip_id_unique')['longitude'].shift()
    df['next_latitude'] = df.groupby('trip_id_unique')['latitude'].shift(-1)
    df['next_longitude'] = df.groupby('trip_id_unique')['longitude'].shift(-1)

    df['distance_from_prev'] = df.apply(lambda row: haversine(row['prev_latitude'], row['prev_longitude'], row['latitude'], row['longitude'])
                              if pd.notnull(row['prev_latitude']) else 0, axis=1) * 1000

    df['distance_from_next'] = df.apply(lambda row: haversine(row['latitude'], row['longitude'], row['next_latitude'], row['next_longitude'])
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
    df['start_hour'] = df.groupby('trip_id_unique')['arrival_time'].transform('min').dt.components['hours']
    df['time_period'] = df['start_hour'].apply(map_time_period)
    return df
def add_total_trip_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['trip_id_unique', 'station_index'])
    trip_distances = df.groupby('trip_id_unique')['distance_from_prev'].sum().reset_index()
    trip_distances.columns = ['trip_id_unique', 'total_distance']
    # Merge the total distances back into the original dataframe
    df = df.merge(trip_distances, on='trip_id_unique')
    return df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['arrival_time'] = pd.to_timedelta(df['arrival_time'])
    df = add_time_category(df)
    # Sort by 'trip_id' and 'station_index'
    df = add_distance_from_prev_next(df)
    # Group by trip_id_unique and sum distances
    df = add_total_trip_distance(df)
    df = add_density(df)
    df = count_lines_per_station(df)
    # dummies for lines

    return df


if __name__ == '__main__':
    df = load_data()
    preprocessing(df)
