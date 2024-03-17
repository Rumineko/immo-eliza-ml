from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Read the .csv file
data = pd.read_csv("./data/cleaned/appended_data.csv")

dataframe1 = data.loc[
    :, ["Price", "Habitable Surface", "Room Count", "State of Building", "Kitchen Type"]
]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataframe1)
frame1 = pd.DataFrame(data=scaled_data, columns=dataframe1.columns)
frame1.head()
regression = LinearRegression()
X = frame1.drop("Price", axis=1)
y = frame1["Price"]
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
intercept = regression.intercept_

print(f"R^2: {r2:.2f}")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"Intercept: {intercept:.2f}")
