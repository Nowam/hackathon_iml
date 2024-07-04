import pandas as pd

from hackathon_code.preprocess.utils import haversine


def calculate_trip_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    return Y value for trip durations
    """

    # Function to calculate duration considering possible midnight span
    def calculate_trip_duration(start_time, end_time):
        duration = end_time - start_time
        if duration < pd.Timedelta(0):
            duration += pd.Timedelta(days=1)
        return duration

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')

    # Calculate the start and end times for each trip
    start_times = df.loc[df.groupby('trip_id_unique')['station_index'].idxmin()][
        ['trip_id_unique', 'arrival_time']]
    end_times = df.loc[df.groupby('trip_id_unique')['station_index'].idxmax()][
        ['trip_id_unique', 'arrival_time']]

    # Merge the start and end times into a single dataframe
    trip_times = pd.merge(start_times, end_times, on='trip_id_unique',
                          suffixes=('_start', '_end'))

    # Calculate trip durations
    trip_times['trip_duration'] = trip_times.apply(
        lambda row: calculate_trip_duration(row['arrival_time_start'],
                                            row['arrival_time_end']), axis=1)

    trip_times['trip_duration_minutes'] = trip_times[
                                              'trip_duration'].dt.total_seconds() / 60

    trip_times = trip_times.drop(columns=['arrival_time_start', 'arrival_time_end'])

    return trip_times


def calc_trip_distance(df):
    # Sort the dataframe by trip_id_unique and station_index
    df = df.sort_values(by=['trip_id_unique', 'station_index'])

    # Calculate distances between consecutive stations within the same trip
    df['prev_latitude'] = df.groupby('trip_id_unique')['latitude'].shift()
    df['prev_longitude'] = df.groupby('trip_id_unique')['longitude'].shift()

    df['distance'] = df.apply(
        lambda row: haversine(row['prev_latitude'], row['prev_longitude'],
                              row['latitude'], row['longitude'])
        if pd.notnull(row['prev_latitude']) else 0, axis=1)

    # Group by trip_id_unique and sum distances
    trip_distances = df.groupby('trip_id_unique')['distance'].sum().reset_index()
    trip_distances.columns = ['trip_id_unique', 'total_distance']

    # Merge the total distances back into the original dataframe
    df = df.merge(trip_distances, on='trip_id_unique')
    df = df.drop(columns=['prev_latitude', 'prev_longitude'])
    return df


def calc_num_of_stations(df):
    # Calculate the number of stations per trip
    station_counts = df.groupby('trip_id_unique')['station_index'].count().reset_index()
    station_counts.columns = ['trip_id_unique', 'number_of_stations']
    df = df.merge(station_counts, on='trip_id_unique')
    return df


def filter_trip_duration_outliers(df):
    # Calculate mean and standard deviation of trip_duration_minutes
    mean_duration = df['trip_duration_minutes'].mean()
    std_duration = df['trip_duration_minutes'].std()

    # Filter out trips where duration is within 2 standard deviations from the mean
    filtered_trip_durations = df[
        (df['trip_duration_minutes'] >= mean_duration - 2 * std_duration) &
        (df['trip_duration_minutes'] <= mean_duration + 2 * std_duration)]
    return filtered_trip_durations


def add_trip_start_hour(df):
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df['start_hour'] = df.groupby('trip_id_unique')['arrival_time'].transform(
        'min').dt.hour
    return df


def calculate_total_passengers(df):
    df["total_passengers"] = df.groupby("trip_id_unique")["passengers_up"].transform(
        'sum')
    return df


def estimated_trip_duration_minutes(df, average_speed_per_cluster):
    df = df.merge(average_speed_per_cluster, on='cluster')
    # Calculate estimated trip duration based on average speed per cluster
    df['estimated_trip_duration_minutes'] = df['total_distance'] / df[
        'average_speed_kmh_per_cluster'] * 60
    return df


def analyze_trip_duration(df, y):
    y = filter_trip_duration_outliers(y)
    df = df.merge(y, on='trip_id_unique')
    df = calc_trip_distance(df)

    # Calculate speed in km/h
    df['trip_duration_hours'] = df['trip_duration_minutes'] / 60
    df['speed_kmh'] = df['total_distance'] / df['trip_duration_hours']

    # Group by cluster to calculate the average speed
    average_speed_per_cluster = df.groupby('cluster')['speed_kmh'].mean().reset_index()
    average_speed_per_cluster.columns = ['cluster', 'average_speed_kmh_per_cluster']

    return average_speed_per_cluster


# Define function to map hours to time periods
def map_time_period(hour):
    if 7 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 15:
        return 'noon'
    elif 15 <= hour < 19:
        return 'afternoon'
    elif 19 <= hour < 23:
        return 'evening'
    else:
        return 'night'


def add_time_period_dummies(df):
    # Apply function to create the new column
    df['time_period'] = df['start_hour'].apply(map_time_period)
    # Create dummy variables for time_period
    df = pd.get_dummies(df, columns=['time_period'])
    return df


def feature_extraction_passengers_up(X: pd.DataFrame):
    """
    Feature extraction
    """
    return X[['trip_id_unique_station', 'passengers_continue']]


def feature_extraction_trip_duration(X: pd.DataFrame, trip_duration_info):
    """
    Feature extraction
    1. Trip count
    """
    X = X.drop(columns=['trip_id', 'part', 'trip_id_unique_station', 'line_id',
                        'alternative', 'station_id', 'door_closing_time',
                        'arrival_is_estimated', 'passengers_continue', 'station_name',
                        'passengers_continue_menupach', 'mekadem_nipuach_luz'])
    X = calc_trip_distance(X)
    X = X.drop(columns=['distance', 'latitude', 'longitude'])
    X = calc_num_of_stations(X)
    X = add_trip_start_hour(X)
    X = calculate_total_passengers(X)
    X = X.drop(
        columns=['station_index', 'arrival_time', 'passengers_up']).drop_duplicates()
    X = estimated_trip_duration_minutes(X, trip_duration_info)
    X = X.drop(columns=['average_speed_kmh_per_cluster'])
    X = add_time_period_dummies(X)
    X = pd.get_dummies(X, columns=['cluster'])

    return X
