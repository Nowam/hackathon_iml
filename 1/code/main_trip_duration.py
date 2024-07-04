from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

from hackathon_code.preprocess.trip_duration_preprocess import calculate_trip_durations
from hackathon_code.preprocess.trip_duration_preprocess import feature_extraction_trip_duration
from hackathon_code.preprocess.trip_duration_preprocess import analyze_trip_duration


def main():
    parser = ArgumentParser()
    parser.add_argument("--training_set", type=str, default="hackathon_code/data/train_bus_schedule.csv",
                        help="path to the training set")
    parser.add_argument("--test_set", type=str,
                        help="path to test set, if not set, split train", required=False)
    parser.add_argument("--out", required=False, type=str,
                        help="path to the output prediction")
    args = parser.parse_args()
    print('Loading Data!')
    df = pd.read_csv(args.training_set,
                     encoding="ISO-8859-8")
    y = calculate_trip_durations(df)
    if not args.test_set:
        # Split train and test data (each training sample has multiple rows)
        y_train = y.sample(frac=0.8, random_state=42)
        X_train = df.loc[df.trip_id_unique.isin(y_train.trip_id_unique)]
        y_test = y.loc[~y.trip_id_unique.isin(y_train.trip_id_unique)]
        X_test = df.loc[df.trip_id_unique.isin(y_test.trip_id_unique)]
    else:
        X_train, y_train = df, y
        X_test = pd.read_csv(args.test_set, encoding="ISO-8859-8")
        y_test = None
    print('Preprocessing!')
    X_train, y_train = preprocess_train(X_train, y_train) # noqa
    trip_duration_info = analyze_trip_duration(X_train, y)
    print('Extracting Features!')
    X_train = feature_extraction_trip_duration(X_train, trip_duration_info)
    X_test = feature_extraction_trip_duration(X_test, trip_duration_info)
    print('Evaluating Model!')

    # Align the indices
    train = X_train.merge(y_train, on='trip_id_unique')
    X_train, y_train = (train.drop(columns=['trip_id_unique', 'trip_duration', 'trip_duration_minutes'], axis=1),
                        train['trip_duration_minutes'])
    if not args.test_set:
        test = X_test.merge(y_test, on='trip_id_unique')
        X_test, y_test = (test.drop(columns=['trip_id_unique', 'trip_duration', 'trip_duration_minutes'], axis=1),
                          test['trip_duration_minutes'])

    # Initialize the model
    model = XGBRegressor(n_estimators=200, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True,
               random_state=42)  # Change n_splits to desired number of folds

    # Calculate cross-validated scores
    mse_scores = cross_val_score(model, X_train, y_train, cv=kf,
                                 scoring='neg_mean_squared_error')

    # Calculate average MSE and R2 scores
    average_mse = np.mean(-mse_scores)

    print(f"Average MSE on Cross-Validation: {average_mse}")

    # Train the model on the entire training set
    model.fit(X_train, y_train)

    # Evaluate the model on the training set
    train_mse = mean_squared_error(y_train, model.predict(X_train))

    print(f"MSE on Train: {train_mse}")

    if not args.test_set:
        # Evaluate the model on the test set
        test_mse = mean_squared_error(y_test, model.predict(X_test))

        print(f"MSE on Test: {test_mse}")

    if args.out:
        X_test['trip_duration_in_minutes'] = model.predict(
            X_test.drop(columns=['trip_id_unique'])).round(2)
        X_test[['trip_id_unique', 'trip_duration_in_minutes']].to_csv(args.out,
                                                                      index=False)


if __name__ == '__main__':
    main()
