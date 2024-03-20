import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=53
)

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print("Mean Absolute Error:", round(np.mean(errors), 2), "euros.")
print("Accuracy:", round(accuracy, 2), "%.")
# Save the model
pickle.dump(rf, open("model_dropped.pkl", "wb"))
