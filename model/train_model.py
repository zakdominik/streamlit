#imports all used libraries
import os
import joblib
import boto3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

#loads the .env file for variables such as AWS credentials
load_dotenv()

#gets to the base directory (2 folders up from this file)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#path to the parquet file for training data
DATA_PATH = os.path.join(BASE, "data/processed/training_data.parquet")
#path where the trained model will be saved locally
MODEL_PATH = os.path.join(BASE, "model/rf_model.pkl")
#S3 bucket and where the model will get uploaded in it
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = "model/rf_model.pkl"

#features on which the model will be trained on
FEATURES = ["square_meters", "bedrooms", "bathrooms", "neighborhood", "property_type"]
#the target column which the model wants to predict
TARGET = "price"
#splits into numerical and categorical for different preprocessing
NUMERICAL = ["square_meters", "bedrooms", "bathrooms"]
CATEGORICAL = ["neighborhood", "property_type"]

#function that loads the parquet file
def load_data():
    #saves parquet as pandas df
    print(f"Loading training data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df

#builds the ML model pipeline: preprocessing and the model
def build_pipeline():
    #for numerical features any missing values are replaced with the column median
    num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")),])

    #for categorical features, null value are filled with Unknown then converts strings to integers
    #unknown_value=-1 ensures that new Neighborhood doesn't crash the model
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    #applies the right transformer to the right columns
    preprocessor = ColumnTransformer([("num", num_transformer, NUMERICAL),("cat", cat_transformer, CATEGORICAL),])

    #the full pipeline chains preprocessing and the model together into one object
    return Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=200, #number of decision trees
            max_depth=20, #how deep each tree can grow
            min_samples_leaf=5, #each leaf must have at least 5 rows which prevents memorisation
            n_jobs=-1,
            random_state=42, #fixed seed for reproducible results
        )),
    ])

#evaluation function
def evaluate(model, X_test, y_test):
    #runs the model on the test set and prints perfromance metrics - MAE, R2, MAPE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test-y_pred)/y_test))*100
    print(f"R2: {r2:.4f}")
    print(f"MAE: £{mae:,.0f}")
    print(f"MAPE: {mape:.1f}%")

#function that extracts how much each feature contributed to the predictions
def feature_importance(model):
    names = NUMERICAL+CATEGORICAL
    importances = model.named_steps["rf"].feature_importances_
    pairs = list(zip(names, importances))
    ranked = sorted(pairs, key=lambda x: x[1], reverse=True)
    print("Feature importances:")
    for name, score in ranked:
        print(f"{name}: {score:.4f}")

#main function that runs the code - loading, training, evaluation...
def main():
    #loads the training data
    df = load_data()
    #separates the features from the target
    X = df[FEATURES]
    y = df[TARGET]

    #80-20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)} rows")
    print(f"Test: {len(X_test)} rows")

    #builds the pipeline and trains it
    model = build_pipeline()
    print("Training the model")
    model.fit(X_train, y_train)
    #evaluates on the unseen test set and print feature importance
    evaluate(model, X_test, y_test)
    feature_importance(model)
    #saves the trained model as a .pkl file
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    #uploads the model to S3 so lambda can access it
    print("Uploading model to S3")
    s3 = boto3.client("s3", region_name="eu-west-2")
    s3.upload_file(MODEL_PATH, S3_BUCKET, S3_KEY)
    print("Finished uploading")

if __name__ == "__main__":
    main()
