# Model Card for HermitCrab-Predict

This model was created as part of the BeCode AI Bootcamp over the course of approximately two weeks in March 2024. It is the third phase of the project, and utilizes the data previously acquired to create a model to predict the price of Houses and Apartments in Belgium.

# Table of Contents

- [Model Card for HermitCrab-Predict](#model-card-for--model_id-)
- [Table of Contents](#table-of-contents-1)
- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
  - [Direct Use](#direct-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  - [Recommendations](#recommendations)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
    - [Preprocessing](#preprocessing)
- [Evaluation](#evaluation)
  - [Testing Data, Factors &amp; Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Factors](#factors)
    - [Metrics](#metrics)
  - [Results](#results)
- [Technical Specifications](#technical-specifications-optional)
- - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Software](#software)
- [Model Card Authors](#model-card-authors-optional)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)

# Model Details

## Model Description

This model was created as part of the BeCode AI Bootcamp over the course of approximately two weeks in March 2024. It is the third phase of the project, and utilizes the data previously acquired to create a model to predict the price of Houses and Apartments in Belgium.

- **Developed by:** [Alice Mendes](https://www.linkedin.com/in/alice-edcm/)
- **Model type:** Random Forest Regression
- **License:** openrail
- **Resources for more information:**
  - [First Part of the Project](https://github.com/karelrduran/Immo-Data-Collection)
  - [Second Part of the Project](https://github.com/Rumineko/immo-eliza-hermitcrabs-analysis)

# Uses

This model is specifically meant to be used to predict the pricing of properties in Belgium.

## Direct Use

The user can provide the model with certain pieces of data, and the model will attempt to give its best estimate on what the price of the house should be, based on said parameter values.

# Bias, Risks, and Limitations

This model has been trained with only data from properties in the category of House and Apartment in Belgium. As such, properties outside of these two categories cannot have their prices be accurately predicted.

## Recommendations

To use only properties in Belgium, that are categorized as either House or Apartment.

# Training Details

## Training Data

The model was trained from a dataset with approximately 70,000 properties in Belgium, obtained by scraping immoweb.be (see Model Description for extra GitHub Repo resources).

## Training Procedure

### Preprocessing

The preprocessing starts with loading the .csv file. It will then append some missing data using external files,  convert some non-numerical values into numerical values as well as fill some empty values using logical reasoning and. Some extra cleaning is still done during the prediction, but this is done simply to get the correct parameters that we use in the model.

# Evaluation

## Testing Data, Factors & Metrics

### Testing Data

20% of Initial Provided Data.

### Factors

- Habitable Surface
- Kitchen Type
- Terrace Surface
- Garden Surface
- State of Building
- EPC
- Type
- Province

### Metrics

Mean Absolute Error and Accuracy

## Results

Mean Absolute Error: 118486.69 euros

Accuracy: 71.89 %

# Technical Specifications

## Model Architecture and Objective

The objective of the model is to, as accurately as possible, predict the pricing of Real Estate Properties in Belgium.

## Compute Infrastructure

### Software

To run this model, you need to have python installed, as well as some additional packages, listed inside of requirements.txt

# Model Card Authors

[Alice Mendes](https://www.linkedin.com/in/alice-edcm/)

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

* create a new virtual environment by executing this command in your terminal:
  `python3 -m venv hermitcrab-predict`
* activate the environment by executing this command in your terminal:
  `source hermitcrab-predict/bin/activate`
* install the required dependencies by executing this command in your terminal:
  `pip install -r requirements.txt`
* build the model by running preprocess.py and train.py:
  `python preprocess.py`
  `python train.py`
* place your data you want to predict inside of the root folder as a file called new_data.csv, and run predict.py:
  `python predict.py`

</details>
