# 💎 Diamond Price Predictor App

A machine learning-powered web application that predicts diamond prices based on physical characteristics.

## Overview

This Streamlit application uses a trained machine learning model to estimate diamond prices based on:
- **Carat Weight**: The weight of the diamond
- **Cut Quality**: Fair, Good, Very Good, Premium, Ideal
- **Color**: Diamond color grade (D-J)
- **Clarity**: Diamond clarity grade (I1-IF)
- **Depth & Table**: Percentage measurements of the diamond
- **Dimensions**: Length (x), Width (y), and Height (z) measurements

## Features

- **Interactive Web Interface**: User-friendly Streamlit interface with sliders and dropdowns
- **Real-time Predictions**: Instant price predictions as you adjust diamond features
- **Comprehensive Data**: Trained on extensive diamond dataset with multiple attributes
- **Encoded Features**: Automatic encoding of categorical variables for model prediction

## Installation

1. Clone or download this directory
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Files

- `app.py`: Main Streamlit application
- `md_diamond.ipynb`: Jupyter notebook with data exploration and model training
- `diamonds.csv`: Diamond dataset used for training
- `diamond_model.pkl`: Pre-trained model file
- `requirements.txt`: Python dependencies

## Model Information

The model is trained using the diamonds dataset and uses features including:
- Numerical features: carat, depth, table, x, y, z, and derived area
- Categorical features: cut, color, clarity (encoded as integers)

## Requirements

Python 3.7+

See `requirements.txt` for detailed dependencies.
