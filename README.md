# London Property Radar

Automated property investment screening tool for London. Scrapes the 10 newest Rightmove listings every 6 hours across 10 London neighbourhoods, scores them using a trained Random Forest model and flags properties where the asking price is more than 10% below the predicted market value.

## Code

- batch_pipeline/batch_pipeline.py — PySpark pipeline that processes the raw CSV into a clean Parquet dataset
- model/train_model.py — trains the Random Forest model and uploads it to S3
- streaming/lambda_function.py — AWS Lambda handler that scrapes Rightmove, scores listings, and writes results to the database
- streaming/deploy_lambda.py — packages and deploys the Lambda function to AWS
- setup_aws.py — one-time setup script for S3, RDS tables, and IAM role
- dashboard/app.py — Streamlit dashboard that reads from the database and displays flagged deals

## Data

- data/raw/london_houses.csv — Kaggle dataset of 1000 London properties used for model training
- The trained model and processed data are stored in AWS S3 (`london-property-radar`, eu-west-2)

## Project Files

- notebooks/model_evaluation.ipynb` — model performance charts and metrics

## Video

video.com
