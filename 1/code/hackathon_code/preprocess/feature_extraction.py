import pandas as pd


def feature_extraction_passengers_up(X: pd.DataFrame):
    """
    Feature extraction
    """
    return X[['trip_id_unique_station', 'passengers_continue']]


def feature_extraction_trip_duration(X: pd.DataFrame):
    """
    Feature extraction
    1. Trip count

    """
    return X.trip_id_unique.value_counts().reset_index().rename(
        columns={'count': 'station_count'}
    )
