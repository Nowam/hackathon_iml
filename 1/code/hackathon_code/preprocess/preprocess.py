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
    df['arrival_time'] = pd.to_timedelta(df['arrival_time'])

    # Function to calculate duration considering possible midnight span
    def calculate_trip_duration(times):
        duration = times.max() - times.min()
        if duration < pd.Timedelta(0):
            duration += pd.Timedelta(days=1)
        return duration

    # Group by trip_id_unique and calculate the trip duration
    trip_durations = df.groupby('trip_id_unique')['arrival_time'].apply(
        calculate_trip_duration).reset_index()

    # Rename the columns for clarity
    trip_durations.columns = ['trip_id_unique', 'trip_duration']

    trip_durations['trip_duration_minutes'] = trip_durations[
                                          'trip_duration'].dt.total_seconds() / 60

    # Merge the trip durations back into the original dataframe
    return trip_durations


def preprocess_train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Cleaning + Row removals

    """
    return X_train, y_train
