import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load data from an Excel file
def load_data(file_path):
    return pd.read_excel(file_path)

# Perform feature engineering and preprocessing
def perform_feature_engineering(data):
    features = data[["Age", "G", "GS", "Cmp", "Att", "Yds", "ThrTD", "Int", "Y/A", "RuTD", "Tgt", "Rec", "Y/R",
                     "ReTD", "Fmb", "ToTD", "2PM", "2PP", "FantPt", "PPR", "VBD"]]

    numeric_features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'ThrTD', 'Int', 'Y/A', 'RuTD', 'Tgt', 'Rec',
                        'Y/R', 'ReTD', 'Fmb', 'ToTD', '2PM', '2PP', 'FantPt', 'PPR', 'VBD']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    features = preprocessor.fit_transform(features)

    poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly_features.fit_transform(features)

    return interaction_features

# Train regression models for each target variable
def train_models(features, target_variables):
    models = {}

    for target in target_variables:
        if target in ["G", "GS", "FantPt", "PPR", "VBD"]:
            model = LinearRegression()
        else:
            model = RandomForestRegressor()

        model.fit(features, data[target])
        models[target] = model

    return models

# Generate predictions for 2023 using trained models
def predict_stats(models, interaction_features_2023, target_variables):
    predictions_2023 = {}

    for target in target_variables:
        model = models[target]
        predictions = model.predict(interaction_features_2023)
        predictions_2023[target] = predictions

    return pd.DataFrame(predictions_2023)

# Evaluate model performance
def evaluate_model(predictions, target):
    # Calculate evaluation metrics
    mse = mean_squared_error(data[target], predictions)
    rmse = mean_squared_error(data[target], predictions, squared=False)
    r2 = r2_score(data[target], predictions)

    print(f"Evaluation for {target}:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print()

# Generate rankings based on predicted stats
def generate_rankings(predictions_2023, target_variables):
    rankings_2023 = predictions_2023.copy()

    category_weights = {
        "Age": 0.05,
        "G": 0.1,
        "GS": 0.1,
        "Cmp": 0.05,
        "Att": 0.05,
        "Yds": 0.1,
        "ThrTD": 4,
        "Int": -2,
        "Y/A": 0.2,
        "RuTD": 6,
        "Tgt": 0.5,
        "Rec": 0.5,
        "Y/R": 0.2,
        "ReTD": 6,
        "Fmb": -2,
        "ToTD": 2,
        "2PM": 2,
        "2PP": 2,
        "FantPt": 1,
        "PPR": 0.5,
        "VBD": 0.2
    }

    rankings_2023["TotalRank"] = rankings_2023.apply(
        lambda row: sum(row[category] * category_weights[category] for category in target_variables), axis=1)
    rankings_2023["PosRank"] = rankings_2023["TotalRank"].rank(ascending=False, method="min")
    rankings_2023["OvRank"] = rankings_2023["PosRank"].rank(ascending=True, method="min")

    return rankings_2023

# Main function
def main():
    file_path = "fantasy_football_stats_2022.xlsx"
    data = load_data(file_path)
    target_variables = ["Cmp", "Att", "Yds", "ThrTD", "Int", "Y/A", "RuTD", "Tgt", "Rec", "Y/R", "ReTD", "Fmb", "ToTD", "2PM", "2PP", "FantPt"]

    features = perform_feature_engineering(data)
    interaction_features_2023 = features[-1].reshape(1, -1)
    features = features[:-1]

    models = train_models(features, target_variables)

    # Evaluate models
    for target in target_variables:
        predictions = cross_val_predict(models[target], features, data[target], cv=5)
        evaluate_model(predictions, target)

    predictions_2023 = predict_stats(models, interaction_features_2023, target_variables)
    rankings_2023 = generate_rankings(predictions_2023, target_variables)

    rankings_2023.to_excel("2023_rankings.xlsx")

if __name__ == "__main__":
    main()

