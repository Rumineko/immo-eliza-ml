from __future__ import annotations
from fastapi import FastAPI
import pickle
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from preprocess import append_data_singular, convert_non_numeric_singular
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()


class Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Habitable_Surface: float = Field(alias="Habitable Surface")
    Kitchen_Type: str = Field(alias="Kitchen Type")
    Terrace_Surface: float = Field(alias="Terrace Surface")
    Garden_Surface: float = Field(alias="Garden Surface")
    EPC: str
    Type: str
    Postal_Code: int = Field(alias="Postal Code")
    Furnished: bool
    Openfire: bool
    State_of_Building: str = Field(alias="State of Building")


price_model = pickle.load(open("model_imputed.pkl", "rb"))

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
    transform="pandas"
)


@app.post("/")
async def price_prediction(input_parameters: Model):
    df = pd.DataFrame(
        [list(input_parameters.model_dump().values())],
        columns=input_parameters.dict().keys(),
    )
    df["Postal Code"] = df["Postal_Code"]
    df.drop("Postal_Code", axis=1, inplace=True)
    df["Habitable Surface"] = df["Habitable_Surface"]
    df.drop("Habitable_Surface", axis=1, inplace=True)
    df["Kitchen Type"] = df["Kitchen_Type"]
    df.drop("Kitchen_Type", axis=1, inplace=True)
    df["Terrace Surface"] = df["Terrace_Surface"]
    df.drop("Terrace_Surface", axis=1, inplace=True)
    df["Garden Surface"] = df["Garden_Surface"]
    df.drop("Garden_Surface", axis=1, inplace=True)
    df["State of Building"] = df["State_of_Building"]
    df.drop("State_of_Building", axis=1, inplace=True)
    print(df)
    df1 = append_data_singular(df)
    df2 = convert_non_numeric_singular(df1)
    df2.drop(["Postal Code", "Openfire", "Furnished"], axis=1, inplace=True)
    df2.dtypes
    prediction = price_model.predict(df2)
    return {"Price": prediction[0]}
