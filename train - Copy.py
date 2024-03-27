import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.impute import KNNImputer
import pickle

# Read the .csv file
data = pd.read_csv("data/cleaned/data.csv")

data = data[data["Latitude"] >= 49]
data = data[data["Latitude"] <= 52]
data.loc[data["Kitchen"] == 0, "Kitchen Type"] = 0
columns_to_drop = [
    "Kitchen Surface",
    "Facades",
    "Living Surface",
    "Heating Type",
    "Toilet Count",
    "ID",
    "Kitchen",
    "Furnished",
    "Openfire",
    "Fireplace Count",
    "Terrace",
    "Garden Exists",
    "Cadastral Income",
    "Consumption Per m2",
    "Price per sqm",
    "Room Count",
    "url",
    "Consumption",
    "Sea view",
    "Build Year",
    "Postal Code",
]
df2 = data.drop(columns_to_drop, axis=1, errors="ignore")


ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
    transform="pandas"
)
ohetransform = ohe.fit_transform(df2[["Region"]])
df3 = pd.concat([df2, ohetransform], axis=1).drop("Region", axis=1)
ohetransform = ohe.fit_transform(df3[["Type"]])
columns_to_drop = ["Type_OFFICE", "Type_LAND", "Type_GARAGE", "Type_COMMERCIAL"]
df4 = pd.concat([df3, ohetransform], axis=1).drop("Type", axis=1)
ohetransform = ohe.fit_transform(df4[["Province"]])
df5 = pd.concat([df4, ohetransform], axis=1).drop("Province", axis=1)
df5.drop(df4[df4[columns_to_drop] == 1].dropna(how="all").index, inplace=True)
df5.drop(columns_to_drop, axis=1, inplace=True)
df5 = df5.select_dtypes(include=["int64", "float64"])
df6 = df5.dropna()
df6 = df6.dropna(subset=["Price"])
columns_to_drop = [
    "Latitude",
    "Longitude",
    "Swimming Pool",
    "Region_Flanders",
    "Region_Brussels",
    "Region_Wallonia",
    "Bedroom Count",
    "Bathroom Count",
]
dataframe = df6.drop(columns_to_drop, axis=1)
X = dataframe.drop("Price", axis=1)
y = dataframe["Price"]
X.shape, y.shape

df5 = df5.dropna(subset=["Price"])

dataframe = df5.drop(columns_to_drop, axis=1)
unimputed_X = dataframe.drop("Price", axis=1)
unimputed_y = dataframe["Price"]


unimputed_X_train, unimputed_X_test, unimputed_y_train, unimputed_y_test = (
    train_test_split(unimputed_X, unimputed_y, test_size=0.2, random_state=53)
)

impute_knn = KNNImputer(n_neighbors=5)
imputed_x_train = impute_knn.fit_transform(unimputed_X_train)
imputed_x_test = impute_knn.fit_transform(unimputed_X_test)

rf = RandomForestRegressor(n_estimators=1000, random_state=0)
rf.fit(imputed_x_train, unimputed_y_train)
unimouted_y_pred = rf.predict(imputed_x_test)

# Calculate mean absolute percentage error (MAPE)
errors = abs(unimouted_y_pred - unimputed_y_test)
mape = 100 * (errors / unimputed_y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print("Mean Absolute Error:", round(np.mean(errors), 2), "euros.")
print("Accuracy:", round(accuracy, 2), "%.")

# Save the model
with open("model_imputed.pkl", "wb") as file:
    pickle.dump(rf, file)
