# Diabetes Risk Predictor

## Overview

Welcome to the Diabetes Risk Predictor, a sophisticated tool designed to evaluate individual risk factors associated with diabetes. Leveraging advanced algorithms and a robust dataset, our predictor offers accurate assessments to aid in proactive health management.

Note that this dataset may not be reliable as this project was developed for educational purposes in the field of machine learning only and not for professional use.

A live version of the application can be found on [Streamlit Community Cloud](https://diabetes-predictor-24.streamlit.app/). 

## Installation

You can run this inside a virtual environment to make it easier to manage dependencies. I recommend using `conda` to create a new environment and install the required packages. You can create a new environment called `diabetes-risk-predictor` by running:

```bash
conda create -n diabetes-risk-predictor python=3.10 
```

Then, activate the environment:

```bash
conda activate diabetes-risk-predictor
```

Then, activate the environment:

```bash
conda activate diabetes-risk-predictor
```

To install the required packages, run:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-image.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app/main.py
```

This will launch the app in your default web browser. You can then upload an image of cells to analyze and adjust the various settings to customize the analysis. Once you are satisfied with the results, you can export the measurements to a CSV file for further analysis.
