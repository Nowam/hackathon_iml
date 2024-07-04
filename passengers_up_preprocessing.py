import pandas as pd
import numpy as np


def remove_rows(df: pd.DataFrame) -> pd.DataFrame:
    pass


def calculate_time_to_next_station():
    pass


def load_data(file_path='train_bus_schedule.csv'):
    df = pd.read_csv(file_path, encoding="ISO-8859-8")
    return df

    # Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Calculate the number of unique stations within 1 km radius for each unique station
def calculate_nearby_stations(unique_stations):
    def wrappper(station):
        lat1, lon1 = station['latitude'], station['longitude']
        distances = unique_stations.apply(lambda row: haversine(lat1, lon1, row['latitude'], row['longitude']), axis=1)
        return (distances <= 1).sum() - 1  # Subtract 1 to exclude the station itself
    return wrappper

def add_distance_from_prev_next(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['trip_id_unique', 'station_index'])

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

def add_total_trip_distance(df: pd.DataFrame) -> pd.DataFrame:
    trip_distances = df.groupby('trip_id_unique')['distance_from_prev'].sum().reset_index()
    trip_distances.columns = ['trip_id_unique', 'total_distance']
    return df
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S', errors='coerce')
    df['hour'] = df['arrival_time'].dt.hour
    # Sort by 'trip_id' and 'station_index'
    df = add_distance_from_prev_next(df)
    # Group by trip_id_unique and sum distances
    df = add_total_trip_distance(df)

    unique_stations = df[['station_id', 'latitude', 'longitude']].drop_duplicates()
    unique_stations['stations_within_1km'] =\
        unique_stations.apply(calculate_nearby_stations(unique_stations), axis=1)

    # Merge the results back into the original dataframe
    df = df.merge(unique_stations[['station_id', 'stations_within_1km']], on='station_id', how='left')

    # Merge the total distances back into the original dataframe
    df = df.merge(trip_distances, on='trip_id_unique')
    new_df = df[['time_from_prev_station', 'time_to_next_station', 'stations_within_1km']]
    print(new_df)

    print(df)
if __name__ == '__main__':
    df = load_data()
    preprocessing(df)
