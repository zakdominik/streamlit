#imports all used libraries
import os
import sys
import json
import glob
import boto3
from dotenv import load_dotenv

#Pyspark needs Java 17, otherwise it throws an error
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

#PySpark imports need to be imported after Java or it throws an error
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

#loads the .env file for variables such as AWS credentials
load_dotenv()

#gets to the base directory (2 folders up)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#gets the paths for input and output files
CSV_PATH = os.path.join(BASE, "data/raw/london_houses.csv")
OUT_DIR = os.path.join(BASE, "data/processed")
OUT_PATH = os.path.join(OUT_DIR, "training_data.parquet")
#S3 bucket name, from .env
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
#S3 path
S3_KEY = "data/training_data.parquet"

#manually sets up the schema for the CSV so it gets read as it should and prevents type mismatches
SCHEMA = StructType([
    StructField("Address", StringType()),
    StructField("Neighborhood", StringType()),
    StructField("Bedrooms", IntegerType()),
    StructField("Bathrooms", IntegerType()),
    StructField("Square Meters", IntegerType()),
    StructField("Building Age", IntegerType()),
    StructField("Garden", StringType()),
    StructField("Garage", StringType()),
    StructField("Floors", IntegerType()),
    StructField("Property Type", StringType()),
    StructField("Heating Type", StringType()),
    StructField("Balcony", StringType()),
    StructField("Interior Style", StringType()),
    StructField("View", StringType()),
    StructField("Materials", StringType()),
    StructField("Building Status", StringType()),
    StructField("Price (£)", IntegerType()),
])

#function that creates a local Spark session
def get_spark():
    return (SparkSession.builder
            .appName("LondonPropertyRadar-BatchPipeline")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate())

#main function running the pipeline
def main():
    #sets up the Spark session
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    print("Loading CSV")
    #converts the CSV into a Spark df using the schema from above
    df = (spark.read
          .option("header", "true")
          .option("quote", '"')
          .option("escape", '"')
          .schema(SCHEMA)
          .csv(CSV_PATH))
    print(f"Raw rows: {df.count()}") #counts the number of rows

    #drops unused columns, renames to a more standardised format and changes m^2 to double
    df = df.select(
        F.col("Address").alias("address"),
        F.col("Neighborhood").alias("neighborhood"),
        F.col("Bedrooms").alias("bedrooms"),
        F.col("Bathrooms").alias("bathrooms"),
        F.col("Square Meters").cast(DoubleType()).alias("square_meters"),
        F.col("Property Type").alias("property_type"),
        F.col("Price (£)").alias("price"),
    )

    #removes any rows where any column is null as we need every column for training
    df = df.dropna(subset=["price", "square_meters", "bedrooms", "bathrooms", "neighborhood", "property_type"])
    print(f"Rows after cleaning: {df.count()}") #counts rows after cleaning

    #convert the cleaned df to parquet (more efficient and faster to read than CSV)
    df.write.mode("overwrite").parquet(OUT_PATH)
    print(f"Saved training data to {OUT_PATH}")

    #extracts the unique list of neighborhoods from the df and saves it as a JSON
    #used later by Lambda to know which areas to search on Rightmove
    neighborhoods = sorted([r[0] for r in df.select("neighborhood").distinct().collect()])
    neighborhoods_path = os.path.join(OUT_DIR, "neighborhoods.json")
    with open(neighborhoods_path, "w") as f:
        json.dump(neighborhoods, f)
    print(f"Neighborhoods: {neighborhoods}")

    print("Uploading to S3")
    #creates an S3 client pointing at eu-west-2 where the bucket is
    s3 = boto3.client("s3", region_name="eu-west-2")

    #uploads each part file of the parquet directory to S3
    part_files = glob.glob(os.path.join(OUT_PATH, "part-*.parquet"))
    for part in part_files:
        part_name = os.path.basename(part)
        s3.upload_file(part, S3_BUCKET, f"{S3_KEY}/{part_name}")
        print(f"Uploaded {part_name} to S3")

    #uploads the neighborhoods JSON separately so lambda can access it without the parquet
    s3.upload_file(neighborhoods_path, S3_BUCKET, "data/neighborhoods.json")
    print("Uploaded neighborhoods.json to S3")
    #stops the Spark session
    spark.stop()
    print("Batch pipeline finished.")


if __name__ == "__main__":
    main()
