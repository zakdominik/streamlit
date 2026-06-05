#Disclaimer: This script was written by a generative AI model.
#Setting up the full infrastructure through the AWS web console proved too restrictive —
#certain configurations such as seeding the database tables, uploading assets to S3, and
#wiring up the IAM role could not be done cleanly through the UI alone.
#This script was written to automate the entire one-time setup process via the AWS SDK instead.

"""
One-time AWS setup script.
Run this once to:
  1. Create the S3 bucket and upload initial assets
  2. Test the RDS connection and create both database tables

After running this, use deploy_lambda.py for all subsequent Lambda deployments.
"""

import os
import boto3
import psycopg2
from dotenv import load_dotenv

load_dotenv()

AWS_REGION  = os.getenv("AWS_REGION", "eu-west-2")
S3_BUCKET   = os.getenv("S3_BUCKET_NAME")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

BASE = os.path.dirname(os.path.abspath(__file__))

# Files to upload on first setup
UPLOAD_FILES = {
    "model/rf_model.pkl":                    "model/rf_model.pkl",
    "data/processed/neighborhoods.json":     "data/neighborhoods.json",
}

CREATE_ALL_LISTINGS_SQL = """
CREATE TABLE IF NOT EXISTS all_listings (
    id               SERIAL PRIMARY KEY,
    listing_url      TEXT UNIQUE,
    address          TEXT,
    postcode         TEXT,
    borough          TEXT,
    neighborhood     TEXT,
    asking_price     INTEGER,
    bedrooms         INTEGER,
    bathrooms        INTEGER,
    floor_area_m2    FLOAT,
    property_type    TEXT,
    predicted_value  FLOAT,
    valuation_ratio  FLOAT,
    score            FLOAT,
    flagged          BOOLEAN DEFAULT FALSE,
    latitude         FLOAT,
    longitude        FLOAT,
    scraped_at       TIMESTAMP DEFAULT NOW()
);
"""

CREATE_FLAGGED_DEALS_SQL = """
CREATE TABLE IF NOT EXISTS flagged_deals (
    id               SERIAL PRIMARY KEY,
    listing_url      TEXT UNIQUE,
    address          TEXT,
    postcode         TEXT,
    borough          TEXT,
    neighborhood     TEXT,
    asking_price     INTEGER,
    bedrooms         INTEGER,
    bathrooms        INTEGER,
    floor_area_m2    FLOAT,
    property_type    TEXT,
    predicted_value  FLOAT,
    valuation_ratio  FLOAT,
    score            FLOAT,
    latitude         FLOAT,
    longitude        FLOAT,
    flagged_at       TIMESTAMP DEFAULT NOW()
);
"""


def setup_s3():
    print("\n── S3 Setup ──────────────────────────────")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )
        print(f"  Created bucket: {S3_BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  Bucket already exists: {S3_BUCKET}")
    except Exception as e:
        print(f"  Bucket note: {e}")

    for local_rel, s3_key in UPLOAD_FILES.items():
        local_path = os.path.join(BASE, local_rel)
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_rel}")
            continue
        size_mb = os.path.getsize(local_path) / 1_048_576
        print(f"  Uploading {local_rel} ({size_mb:.1f} MB) → s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"    Done.")

    print("  S3 setup complete.")


def setup_rds():
    print("\n── RDS Setup ─────────────────────────────")
    print(f"  Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=10,
        )
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute(CREATE_ALL_LISTINGS_SQL)
        print("  Table 'all_listings' created (or already exists).")

        cur.execute(CREATE_FLAGGED_DEALS_SQL)
        print("  Table 'flagged_deals' created (or already exists).")

        # Seed neighborhoods from neighborhoods.json
        neighborhoods_path = os.path.join(BASE, "data/processed/neighborhoods.json")
        if os.path.exists(neighborhoods_path):
            import json
            with open(neighborhoods_path) as f:
                neighborhoods = json.load(f)
            for name in neighborhoods:
                cur.execute(
                    "INSERT INTO neighborhoods (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                    (name,)
                )
            print(f"  Seeded {len(neighborhoods)} neighborhoods.")
        else:
            print("  WARNING: neighborhoods.json not found — run batch_pipeline.py first.")

        for table in ["neighborhoods", "postcodes", "property_types", "listings", "analysed_listings"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"  Rows in {table}: {cur.fetchone()[0]}")

        cur.close()
        conn.close()
        print("  RDS setup complete.")
    except Exception as e:
        print(f"  RDS ERROR: {e}")


if __name__ == "__main__":
    setup_s3()
    setup_rds()
    print("\nSetup complete.")
