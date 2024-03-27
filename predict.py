import pickle
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from .preprocess import (
    load_data,
    fill_empty_data,
    append_data,
    convert_non_numeric_to_numeric,
    price_per_sqm,
    province_to_region,
)


def main():
    # Load the new CSV data
    new_data = load_data("new_data.csv")
    # We then append the data
    appended_data = append_data(new_data)
    # We then convert the non-numeric data to numeric data
    converted_data = convert_non_numeric_to_numeric(appended_data)
    # We then fill in the empty data
    filled_data = fill_empty_data(converted_data)
    # We drop some columns that we don't need
    filled_data.drop(
        columns=[
            "Sewer",
            "Terrace Orientation",
            "Garden Orientation",
            "Has starting Price",
            "Transaction Subtype",
            "Is Holiday Property",
            "Gas Water Electricity",
            "Parking count inside",
            "Parking count outside",
            "Land Surface",
        ],
        inplace=True,
    )
    # We create a new column 'Region' by applying the function 'province_to_region' to the 'Province' column
    filled_data["Region"] = filled_data["Province"].apply(province_to_region)
    # We use the price_per_sqm function to create a new column 'Price per Sqm'
    filled_data = price_per_sqm(filled_data)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
        transform="pandas"
    )
    ohetransform = ohe.fit_transform(filled_data[["Province"]])
    dataframe1 = pd.concat([filled_data, ohetransform], axis=1).drop("Province", axis=1)
    ohetransform = ohe.fit_transform(dataframe1[["Type"]])
    dataframe2 = pd.concat([dataframe1, ohetransform], axis=1).drop("Type", axis=1)
    columns_to_predict = [
        "Habitable Surface",
        "Kitchen Type",
        "Terrace Surface",
        "Garden Surface",
        "State of Building",
        "EPC",
        "Type_APARTMENT",
        "Type_HOUSE",
        "Province_ANTWERPEN",
        "Province_BRUSSEL",
        "Province_HENEGOUWEN",
        "Province_LIMBURG",
        "Province_LUIK",
        "Province_LUXEMBURG",
        "Province_NAMEN",
        "Province_OOST-VLAANDEREN",
        "Province_VLAAMS-BRABANT",
        "Province_WAALS-BRABANT",
        "Province_WEST-VLAANDEREN",
    ]
    dataframe3 = dataframe2[columns_to_predict]

    # Load the model
    model = pickle.load(open("model_imputed.pkl", "rb"))

    # Make predictions
    predictions = model.predict(dataframe3)

    # Print the predicted values
    print(predictions)

    # Save the predictions to a CSV file
    filled_data["Predicted Price"] = predictions
    filled_data.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
