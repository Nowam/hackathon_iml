from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from hackathon_code.preprocess.feature_extraction import feature_extraction_passengers_up
from hackathon_code.preprocess.preprocess import preprocess_train


def main():
    parser = ArgumentParser()
    parser.add_argument("--training_set", type=str,
                        default="hackathon_code/data/train_bus_schedule.csv",
                        help="path to the training set")
    parser.add_argument("--test_set", type=str,
                        help="path to test set, if not set, split train", required=False)
    parser.add_argument("--out", required=False, type=str,
                        help="path to the output prediction")
    args = parser.parse_args()

    df = pd.read_csv(args.training_set, encoding="ISO-8859-8")
    X, y = df.drop('passengers_up', axis=1), df[['trip_id_unique', 'passengers_up']]
    if not args.test_set:
        # We want to split out full trips
        unique_trips_train = X['trip_id_unique'].drop_duplicates().sample(
            frac=0.8, random_state=42)
        X_train = df.loc[df.trip_id_unique.isin(unique_trips_train)]
        y_train = y.loc[y.trip_id_unique.isin(unique_trips_train)].passengers_up
        X_test = df.loc[~df.trip_id_unique.isin(unique_trips_train)]
        y_test = y.loc[~y.trip_id_unique.isin(unique_trips_train)].passengers_up

    else:
        X_train, y_train = X, y.passengers_up
        X_test = pd.read_csv(args.test_set, encoding="ISO-8859-8")
        y_test = None

    X_train, y_train = preprocess_train(X_train, y_train)
    X_train = feature_extraction_passengers_up(X_train).drop(
        columns=['trip_id_unique_station'])
    X_test = feature_extraction_passengers_up(X_test)

    # Initialize the model
    model = LinearRegression()

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True,
               random_state=42)  # Change n_splits to desired number of folds

    # Calculate cross-validated scores
    mse_scores = cross_val_score(model, X_train, y_train, cv=kf,
                                 scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

    # Calculate average MSE and R2 scores
    average_mse = np.mean(-mse_scores)
    average_r2 = np.mean(r2_scores)

    print(f"Average MSE on Cross-Validation: {average_mse}")
    print(f"Average R2 Score on Cross-Validation: {average_r2}")

    # Train the model on the entire training set
    model.fit(X_train, y_train)

    # Evaluate the model on the training set
    train_score = model.score(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))

    print(f"Model Score on Train: {train_score}")
    print(f"MSE on Train: {train_mse}")

    if not args.test_set:
        # Evaluate the model on the test set
        test_score = model.score(X_test.drop(columns=['trip_id_unique_station']),
                                 y_test)
        test_mse = mean_squared_error(y_test, model.predict(
            X_test.drop(columns=['trip_id_unique_station']))
                                      )

        print(f"Model Score on Test: {test_score}")
        print(f"MSE on Test: {test_mse}")

    if args.out:
        X_test['passengers_up'] = model.predict(
            X_test.drop(columns=['trip_id_unique_station'])).round()
        X_test[['trip_id_unique_station', 'passengers_up']].to_csv(args.out,
                                                                   index=False)


if __name__ == '__main__':
    main()
