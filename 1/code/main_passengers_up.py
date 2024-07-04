from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from hackathon_code.preprocess.passengers_up_preprocessing import preprocessing, adjust_rows


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
    df = preprocessing(df)
    if not args.test_set:
        # We want to split out full trips
        unique_trips_train = df['trip_id_unique'].drop_duplicates().sample(
            frac=0.8, random_state=42)
        df_train = df.loc[df.trip_id_unique.isin(unique_trips_train)]
        # Concatenate X_train and y_train
        df_test = df.loc[~df.trip_id_unique.isin(unique_trips_train)]
    else:
        df_train = df
        df_test = preprocessing(pd.read_csv(args.test_set, encoding="ISO-8859-8"))
        df = pd.concat([df_train, df_test])

    df_train, df_test = adjust_rows(df, df_train.copy(), df_test.copy())
    X_train, y_train = df_train.drop(columns=['trip_id_unique', 'trip_id_unique_station', 'passengers_up'], axis=1), df_train['passengers_up']
    if not args.test_set:
        X_test, y_test = df_test.drop(columns=['trip_id_unique', 'passengers_up']), df_test['passengers_up']
    else:
        X_test = df_test.drop(columns=['trip_id_unique'])
    # Initialize the model
    model = XGBRegressor()

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True,
               random_state=42)  # Change n_splits to desired number of folds

    # Calculate cross-validated scores
    mse_scores = cross_val_score(model, X_train, y_train, cv=kf,
                                 scoring='neg_mean_squared_error')

    # Calculate average MSE and R2 scores
    average_mse = np.mean(-mse_scores)

    print(f"Average MSE on Cross-Validation: {average_mse}")

    model = XGBRegressor(n_estimators=200, max_depth=5, eta=0.1, subsample=0.7,
                         colsample_bytree=0.8)
    # Train the model on the entire training set
    model.fit(X_train, y_train)

    # Evaluate the model on the training set
    train_mse = mean_squared_error(y_train, model.predict(X_train))

    print(f"MSE on Train: {train_mse}")

    if not args.test_set:
        # Evaluate the model on the test set

        test_mse = mean_squared_error(y_test, model.predict(
            X_test)
                                      )

        print(f"MSE on Test: {test_mse}")

    if args.out:
        X_test['passengers_up'] = model.predict(
            X_test.drop(columns=['trip_id_unique_station'])).copy()
        X_test.loc[X_test['passengers_up'] < 0, 'passengers_up'] = 0
        X_test['passengers_up'] = X_test['passengers_up'].round().astype(int)
        X_test[['trip_id_unique_station', 'passengers_up']].to_csv(args.out,
                                                                   index=False)


if __name__ == '__main__':
    main()
