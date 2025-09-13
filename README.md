# Adult Census Income Predictor

This project predicts whether an individual's income exceeds $50K/year using the UCI Adult Census dataset and logistic regression.

## Features

- Cleans and preprocesses the dataset (`main.py`)
- Handles missing values and encodes categorical features
- Trains a logistic regression model (`ModelTraining.py`)
- Saves preprocessing objects and model for future use
- Provides a script for user input-based predictions (`Prediction.py`)

## Workflow

1. **Data Cleaning**  
   - Reads `adult.csv`, treats `?` as missing values  
   - Fills missing `workclass` and `occupation` with `'Unknown'`  
   - Drops rows with missing `native.country`  
   - Removes the `fnlwgt` column  
   - Saves cleaned data to `adult_cleaned.csv`

2. **Model Training**  
   - Splits data into train/test sets  
   - Scales numerical features and one-hot encodes categorical features  
   - Trains a logistic regression model  
   - Saves model and preprocessing objects with `joblib`

3. **Prediction**  
   - Loads model and preprocessing objects  
   - Accepts user input for all features  
   - Preprocesses input and predicts income class and probabilities

## Usage

1. **Install dependencies**  
   ```bash
   pip install pandas scikit-learn joblib numpy
   ```

2. **Run data cleaning**  
   ```bash
   python main.py
   ```

3. **Train the model**  
   ```bash
   python ModelTraining.py
   ```

4. **Make predictions**  
   ```bash
   python Prediction.py
   ```
   Enter the requested information when prompted.

## Files

- `main.py`: Data cleaning and preprocessing
- `ModelTraining.py`: Model training and saving
- `Prediction.py`: User input prediction script
- `adult.csv`: Raw dataset
- `adult_cleaned.csv`: Cleaned dataset
- `.joblib` files: Saved model and preprocessing objects

## Notes

- Categorical features are one-hot encoded.
- Missing values in `workclass` and `occupation` are filled with `'Unknown'`.
- Rows with missing `native.country` are dropped.
- The model achieves ~85% accuracy.

