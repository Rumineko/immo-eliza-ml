import json
import requests
from preprocess import append_data_singular, convert_non_numeric_singular

url = "http://127.0.0.1:8000/price_prediction"

input_dict = {
    "Habitable Surface": 100,
    "Kitchen Type": "INSTALLED",
    "Terrace Surface": 10,
    "Garden Surface": 20,
    "EPC": "A",
    "Type": "APARTMENT",
    "Postal Code": 9000,
    "Furnished": False,
    "Openfire": False,
    "State of Building": "GOOD",
}

input_dict = append_data_singular(input_dict)
input_dict = convert_non_numeric_singular(input_dict)

print(input_dict)
input_json = json.dumps(input_dict)

response = requests.post(url, data=input_json)
print(response.text)
