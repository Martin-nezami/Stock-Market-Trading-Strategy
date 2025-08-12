# Stock Market Trading Strategy

## Overview
This project implements a **Long Short-Term Memory (LSTM)** neural network model to predict stock prices based historical market data.  
The model leverages sequential learning to capture temporal dependencies in price movements, enabling more accurate forecasting for investment and trading decision-making.  

The system is trained on historical **Close** price data and evaluated on separate validation and test datasets.  
Additionally, a simple investment strategy is implemented to simulate potential wealth changes based on model predictions.

## Project Structure
The project is organized as follows:

.
├── data/
│   ├── IVV_1m_validation.csv
│   └── IVV_test_sample.csv
├── models/
│   └── lstm_stock_model.h5        # saved model
├── notebooks/
│   └── lstm_stock_prediction.ipynb # (optional) notebook version
├── src/
│   ├── dataset.py                  # create_dataset, scaling helpers
│   ├── model.py                    # build_lstm_model()
│   ├── train.py                    # training loop + early stopping
│   ├── evaluate.py                 # RMSE + plotting
│   └── backtest.py                 # wealth process logic
├── requirements.txt
└── README.md

1. **Data Loading and Preprocessing**  
   - Loads historical stock price data from CSV files.  
   - Uses only the `Close` price for modeling.  
   - Scales data to a range of [0,1] using `MinMaxScaler`.  
   - Converts time series into supervised learning format using a lookback window of 100 time steps.

2. **LSTM Neural Network**  
   - Three LSTM layers with dropout regularization and L2 weight penalties.  
   - Fully connected output layer for predicting the next closing price.  
   - Compiled with Mean Squared Error loss and Adam optimizer.

3. **Model Training**  
   - Trains the LSTM model with early stopping to prevent overfitting.  
   - Uses separate training and validation datasets.  

4. **Prediction and Evaluation**  
   - Generates predictions on training, validation, and test sets.  
   - Computes RMSE for quantitative evaluation.  
   - Visualizes predicted vs actual prices.

5. **Investment Strategy Simulation**  
   - Implements a basic trading strategy:  
     - Go long if the predicted price is higher than the previous prediction.  
     - Stay in cash otherwise.  
   - Tracks wealth changes over time and computes directional accuracy.

6. **Model Saving and Loading**  
   - Saves the trained LSTM model in `.h5` format.  
   - Loads the saved model for inference on test data.

## Setup and Installation
### Prerequisites
- Python 3.x (Python 3.7 or later recommended)
- `pip` (Python package installer)

### Required Libraries
Install the dependencies with:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Running the Code
1. Place the CSV data files (`IVV_1m_training.csv`, `IVV_1m_validation.csv`, `IVV_test_sample.csv`) in the working directory.
2. Run the Python script or Jupyter Notebook containing the provided LSTM implementation.
3. Adjust file paths and parameters as needed.

## Results and Visualizations
- Training, validation, and test predictions are plotted against actual prices.
- RMSE scores are printed for all datasets.
- Wealth process over time is visualized for the investment strategy.

## Conclusion
This project demonstrates how LSTM networks can be applied to stock price forecasting.  
While results show potential predictive capability, real-world deployment would require more robust feature engineering, transaction cost modeling, and extensive backtesting.
