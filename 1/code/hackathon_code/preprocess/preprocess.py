import pandas as pd


def clean_data(X: pd.DataFrame, y: pd.DataFrame):
    """
    Pruning, column fixing, etc
    """
    return X, y


def preprocess_train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Cleaning + Row removals

    """
    return clean_data(X_train, y_train)


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
    start_times = df.loc[df.groupby('trip_id_unique')['station_index'].idxmin()][['trip_id_unique', 'arrival_time']]
    end_times = df.loc[df.groupby('trip_id_unique')['station_index'].idxmax()][['trip_id_unique', 'arrival_time']]

    # Merge the start and end times into a single dataframe
    trip_times = pd.merge(start_times, end_times, on='trip_id_unique', suffixes=('_start', '_end'))

    # Calculate trip durations
    trip_times['trip_duration'] = trip_times.apply(
        lambda row: calculate_trip_duration(row['arrival_time_start'], row['arrival_time_end']), axis=1)

    trip_times['trip_duration_minutes'] = trip_times[
                                              'trip_duration'].dt.total_seconds() / 60

    trip_times = trip_times.drop(columns=['arrival_time_start', 'arrival_time_end'])

    return trip_times


def preprocess_train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Cleaning + Row removals

    """
    return X_train, y_train
