from matplotlib import pyplot as plt
from preprocess.feature_extraction import *
import numpy as np
import seaborn as sns

def number_of_passenger_to_line_freq(df):
    line_total_passengers = calculate_total_passengers(df)
    line_total_passengers = line_total_passengers.drop_duplicates('trip_id_unique')
    line_total_passengers = line_total_passengers.groupby('line_id')['total_passengers'].sum()
    print(line_total_passengers['19047'])

    trip_counts = df.groupby('line_id')['trip_id_unique'].nunique()
    summary_df = pd.DataFrame({
        'line_id': line_total_passengers.index,
        'total_passengers': line_total_passengers.values,
        'trip_counts': trip_counts.values
    })

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='trip_counts', y='total_passengers', data=summary_df, hue='total_passengers', palette='viridis',
                    size='total_passengers', sizes=(50, 200))
    plt.title('Crowded Lines based on Trip Counts and Total Passengers')
    plt.xlabel('Number of Trips')
    plt.ylabel('Total Passengers')
    plt.grid(True)
    plt.tight_layout()

    slope, intercept = np.polyfit(summary_df['trip_counts'], summary_df['total_passengers'], 1)
    x = np.array(summary_df['trip_counts'])
    y = slope * x + intercept

    # Calculate residuals (vertical distances from the line)
    residuals = np.abs(summary_df['total_passengers'] - (slope * summary_df['trip_counts'] + intercept))

    # Define outliers based on residuals (example: using 1.5 * MAD)
    MAD = np.median(residuals)
    outlier_threshold = 5 * MAD

    # Annotate only outliers
    outliers = summary_df[residuals > outlier_threshold]
    for i, txt in enumerate(outliers['line_id']):
        plt.annotate(txt, (outliers['trip_counts'].iloc[i], outliers['total_passengers'].iloc[i]))

    plt.show()


def find_time_period_activity(df, time_period):
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = add_trip_start_hour(df)
    activity = add_time_period_dummies(df)
    activity = activity[activity[time_period] == 1]
    busy_stations = activity.groupby('station_id')['passengers_up'].sum().reset_index().sort_values(by="passengers_up", ascending=False)
    print(busy_stations.head(10))
    return busy_stations


def get_busy_stations_in_noon_and_night(df):
    sns.set(style="whitegrid")
    noon_busy_stations = find_time_period_activity(df, 'time_period_noon').head(10)
    night_busy_stations = find_time_period_activity(df, 'time_period_night').head(10)

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))

    # Plot for Noon
    sns.barplot(x=noon_busy_stations["station_id"], y=noon_busy_stations["passengers_up"], ax=axes[0],
                palette='YlOrBr', order=noon_busy_stations["station_id"])
    axes[0].set_title('Top 10 Busy Stations at Noon', fontsize=16)
    axes[0].set_xlabel('Station ID', fontsize=14)
    axes[0].set_ylabel('Total Passengers', fontsize=14)
    for i, v in enumerate(noon_busy_stations["passengers_up"]):
        axes[0].text(i, v + 1, str(v), color='black', ha='center')

    # Plot for Night
    sns.barplot(x=night_busy_stations["station_id"], y=night_busy_stations["passengers_up"], ax=axes[1],
                palette='Purples_d', order=night_busy_stations["station_id"])
    axes[1].set_title('Top 10 Busy Stations at Night', fontsize=16)
    axes[1].set_xlabel('Station ID', fontsize=14)
    axes[1].set_ylabel('Total Passengers', fontsize=14)
    for i, v in enumerate(night_busy_stations["passengers_up"]):
        axes[1].text(i, v + 1, str(v), color='black', ha='center')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("data/train_bus_schedule.csv", encoding="ISO-8859-8")
    number_of_passenger_to_line_freq(df)
    get_busy_stations_in_noon_and_night(df)