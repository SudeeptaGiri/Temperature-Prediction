# Temperature Prediction Model

This project implements a machine learning model for temperature prediction using historical data.

## Table of Contents

- [Temperature Prediction Model](#temperature-prediction-model)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dataset](#dataset)
  

## Description

The temperature prediction model is built using machine learning techniques to forecast temperatures based on historical data. It utilizes an LSTM (Long Short-Term Memory) neural network for time-series prediction.

## Installation

To install and run the temperature prediction model, follow these steps:

```bash
git clone https://github.com/SudeeptaGiri/Temperature-Prediction.git
cd Temperature-Prediction
pip install -r requirements.txt
```

## Usage

After installation, you can use the temperature prediction model by running the following command:

```bash
python predict_temperature.py
```
This will execute the model and provide temperature predictions based on the trained data.

## DataSet

The model is trained on a dataset containing historical temperature data. You can find the dataset in the data directory. If you have a different dataset, make sure it follows the same format for compatibility.
