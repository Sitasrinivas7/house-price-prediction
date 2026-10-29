# 🏠 House Price Prediction

## 1. Project Overview

The House Price Prediction project uses Machine Learning to estimate residential property prices based on property-related features such as location, carpet area, furnishing, floor, and other details.

The main objective is to understand the real-estate dataset, perform data preprocessing, train regression models, and evaluate their prediction performance.

## 2. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

## 3. Dataset

The dataset contains residential property listings with information about property prices, locations, areas, furnishing, bathrooms, balconies, parking, and other attributes.

The original dataset contains approximately 187,531 records and 21 columns.

## 4. Data Cleaning and Preprocessing

The following steps were performed:

* Inspected dataset structure and data types.
* Identified missing values.
* Converted property prices from lakh/crore formats into rupees.
* Removed records with missing target prices.
* Investigated unusually high property prices.
* Selected relevant features for the initial model.
* Handled missing numerical and categorical values using a preprocessing pipeline.
* Encoded categorical features using OneHotEncoder.
* Split the data into training and testing sets.

For the baseline experiment, properties above the 99th percentile of the target price were excluded. This decision and its limitations should be considered when interpreting the results.

## 5. Machine Learning Models

Two regression algorithms were evaluated:

### Linear Regression

Used as the baseline model.

### Random Forest Regressor

Used to model potentially more complex relationships between property features and prices.

## 6. Model Evaluation

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R-squared (R²)

### Results

| Metric   | Linear Regression | Random Forest |
| -------- | ----------------: | ------------: |
| MAE      |       ₹54.97 lakh |   ₹14.62 lakh |
| RMSE     |       ₹83.79 lakh |   ₹35.62 lakh |
| R² Score |            0.3432 |        0.8813 |

The Random Forest model performed better than the baseline Linear Regression model on the current test set.

These results apply to the filtered dataset and should not be assumed to generalize to all properties or markets.

## 7. Sample Prediction

A sample property was passed to the trained Random Forest pipeline.

The model predicted a price of approximately ₹1.36 crore.

This is a model estimate, not a verified market valuation.

## 8. Saved Model

The trained pipeline was saved as:

`house_price_model.pkl`

The saved pipeline includes preprocessing and the Random Forest regressor.

## 9. Limitations

* Many properties have missing area and amenity information.
* Some unusually high prices were excluded from the baseline experiment.
* Prediction errors remain, particularly for some expensive properties.
* The model has not been independently validated on a separate external dataset.
* Predictions should not be treated as professional property valuations.

## 10. Future Improvements

* Improve feature engineering using property title and description.
* Investigate missing values and unusual prices further.
* Compare additional regression algorithms.
* Perform hyperparameter tuning and cross-validation.
* Evaluate performance across different price ranges and locations.
* Test the model on unseen, representative property listings.
