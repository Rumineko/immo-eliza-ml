import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
import compress_pickle as cpkl
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Read the .csv file
data = pd.read_csv("data/cleaned/data.csv")

columns_to_use = [
    "Type",
    "Province",
    "Price",
    "Kitchen Type",
    "Garden Surface",
    "Habitable Surface",
    "Terrace Surface",
    "Furnished",
    "Openfire",
    "State of Building",
    "EPC",
    "Swimming Pool",
    "Latitude",
    "Longitude",
    "Region",
]


data = data[data["Latitude"] >= 49]
data = data[data["Latitude"] <= 52]
data.loc[data["Kitchen"] == 0, "Kitchen Type"] = 0
data = data[data["Type"].isin(["HOUSE", "APARTMENT"])]
data = data[columns_to_use]

data.dropna(subset=["Price"], inplace=True)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
    transform="pandas"
)
ohetransform = ohe.fit_transform(data[["Type"]])
data = pd.concat([data, ohetransform], axis=1).drop("Type", axis=1)
ohetransform = ohe.fit_transform(data[["Province"]])
data = pd.concat([data, ohetransform], axis=1).drop("Province", axis=1)
ohetransform = ohe.fit_transform(data[["Region"]])
data = pd.concat([data, ohetransform], axis=1).drop("Region", axis=1)

X = data.drop("Price", axis=1)
y = data["Price"]

# Split the preprocessed data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=53
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "one_hot_encoding",
            OneHotEncoder(),
        ),
    ]
)

categorical_columns = X_train.select_dtypes(include=["object"]).columns

preprocessing_pipeline = ColumnTransformer(
    [
        ("categorical", categorical_pipeline, categorical_columns),
    ],
)

final_pipeline = Pipeline(
    steps=[
        (
            "model",
            RandomForestRegressor(
                n_estimators=1000, max_depth=10, min_samples_split=50, random_state=0
            ),
        ),
    ]
)


# Apply the preprocessing pipeline to the data
final_pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = final_pipeline.predict(X_test)

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print("Mean Absolute Error:", round(np.mean(errors), 2), "euros.")
print("Accuracy:", round(accuracy, 2), "%.")

print(X_train.dtypes)

# Save the model
with open("model_imputed.pkl", "wb") as file:
    pickle.dump(final_pipeline, file)
