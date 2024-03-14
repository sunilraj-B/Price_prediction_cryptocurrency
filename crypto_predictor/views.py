# views.py
from django.urls import reverse
from django.shortcuts import render, redirect
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import json
def table(request):
    table_url=reverse('crypto_predictor:table')
    return render(request,'table.html',{'table_url':table_url})
def update(request):
    update_url = reverse('crypto_predictor:update')
    return render(request, 'update.html', {'update_url': update_url})

def select_crypto(request):
    # Fetch a list of cryptocurrency names from an API
    api_url = 'https://api.coingecko.com/api/v3/coins/list'
    response = requests.get(api_url)
    
    if response.status_code == 200:
        # Parse the API response to get a list of cryptocurrency names
        crypto_data = response.json()
        crypto_names = [(crypto['id'], crypto['name']) for crypto in crypto_data]
    else:
        # Use default cryptocurrency names if API request fails
        crypto_names = [('bitcoin', 'Bitcoin'), ('ethereum', 'Ethereum')]

    return render(request, 'select_crypto.html', {'crypto_names': crypto_names})

def predict_crypto_prices(request):
    crypto_symbol = request.GET.get('crypto_symbol', 'bitcoin').lower()  # Default to bitcoin if not provided
    days = 30

    # Fetch historical price data from CoinGecko for the selected cryptocurrency
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart'
    params = {'vs_currency': 'usd', 'days': days}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx status codes)
        data = response.json()
        # print(data)  # Log the API response to understand its structure

        # Check if 'prices' key is present in the API response
        if 'prices' not in data:
            raise KeyError("'prices' key not found in the API response.")

        # Extract prices and timestamps
        prices = [point[1] for point in data['prices']]
        timestamps = [point[0] for point in data['prices']]

        # Create a Pandas DataFrame
        df = pd.DataFrame({'timestamp': timestamps, 'price': prices})

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Feature engineering: extracting date-related features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        # Split the data into training and testing sets
        X = df[['day_of_week', 'day', 'month']]
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create regression models
        linear_model = LinearRegression()
        decision_tree_model = DecisionTreeRegressor(random_state=42)
        random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
        svr_model = SVR(kernel='linear')  # Support Vector Regression with a linear kernel
        gradient_boosting_model = GradientBoostingRegressor(random_state=42)

        # Train models
        trained_models = [linear_model, decision_tree_model, random_forest_model, svr_model, gradient_boosting_model]
        for model in trained_models:
            model.fit(X_train, y_train)

        # Perform cross-validation for each model
        ensemble_predictions = []
        model_names = ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'SVR',
                       'Gradient Boosting Regression']

        for i, model in enumerate(trained_models):
            # Perform cross-validation and get predictions
            y_cv_pred = cross_val_predict(model, X, y, cv=5)
            ensemble_predictions.append(y_cv_pred)

            # Evaluate the model
            mae = mean_absolute_error(y, y_cv_pred)
            mse = mean_squared_error(y, y_cv_pred)
            r2 = r2_score(y, y_cv_pred)

            print(f"Model: {model_names[i]}")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Mean Squared Error (MSE): {mse:.2f}")
            print(f"R-squared (R^2) Score: {r2:.2f}")
            print("-" * 30)

        # Ensemble approach: Average the predictions from all models
        y_pred_ensemble = np.mean(ensemble_predictions, axis=0)
        mae_ensemble = mean_absolute_error(y, y_pred_ensemble)
        mse_ensemble = mean_squared_error(y, y_pred_ensemble)
        r2_ensemble = r2_score(y, y_pred_ensemble)

        print("Ensemble Approach")
        print(f"Mean Absolute Error (MAE): {mae_ensemble:.2f}")
        print(f"Mean Squared Error (MSE): {mse_ensemble:.2f}")
        print(f"R-squared (R^2) Score: {r2_ensemble:.2f}")

        # Future date prediction: Predict prices for the next n days
        n_days = 14  # Number of days to predict into the future
        future_dates = [df['timestamp'].max() + timedelta(days=i) for i in range(1, n_days + 1)]

        # Extract date-related features for future dates
        future_days_of_week = [date.weekday() for date in future_dates]
        future_days = [date.day for date in future_dates]
        future_months = [date.month for date in future_dates]

        # Create a DataFrame for future dates
        future_df = pd.DataFrame({
            'day_of_week': future_days_of_week,
            'day': future_days,
            'month': future_months
        })

        # Ensemble approach: Average the predictions from all models for future dates
        future_predictions = np.mean([model.predict(future_df) for model in trained_models], axis=0)

        # Print future price predictions
        for i, date in enumerate(future_dates):
            print(f"Predicted Price on {date}: ${future_predictions[i]:.2f}")

        # Prepare data for rendering in the template
        ensemble_models = [str(model) for model in model_names]
        future_results = list(zip(future_dates, future_predictions))
        
        response_data = {
            'ensemble_models': ensemble_models,
            'mae_ensemble': mae_ensemble,
            'mse_ensemble': mse_ensemble,
            'r2_ensemble': r2_ensemble,
            'future_dates': [date.strftime('%Y-%m-%d') for date in future_dates],
            'predicted_prices': [float(price) for price in future_predictions],
            'future_results': future_results,
        }

        return render(request, 'hello.html', response_data)

    except requests.exceptions.RequestException as e:
        # Handle HTTP request errors
        print(f'HTTP request error. Exception: {e}')
        return render(request, 'error.html', {'error_message': 'Error making request to CoinGecko API'})

    except json.JSONDecodeError as e:
        # Handle JSON decoding errors
        print(f'Error decoding JSON. Exception: {e}')
        return render(request, 'error.html', {'error_message': 'Error decoding JSON from CoinGecko API'})

    except KeyError as e:
        # Handle missing key errors
        print(f'KeyError in API response. Exception: {e}')
        return render(request, 'error.html', {'error_message': str(e)})

    except Exception as e:
        # Handle any other exception that might occur during data processing
        print(f'Error processing data. Exception: {e}')
        return render(request, 'error.html', {'error_message': 'Error processing data'})
