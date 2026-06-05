#Disclaimer: This script was written by a generative AI model.
#Deploying a Lambda function with cross-platform dependencies via the AWS console is not possible
#because the console has a 10MB direct upload limit and does not support specifying Linux-compatible
#wheel formats. This script automates the full deployment: building the package with manylinux wheels,
#uploading to S3, and configuring the Lambda function and EventBridge schedule via the AWS SDK.

"""
Packages and deploys the Lambda function to AWS.
Creates the function if it doesn't exist, updates it if it does.
Also creates the EventBridge rule to trigger it every 6 hours.
"""

import os
import sys
import json
import shutil
import subprocess
import zipfile
import boto3
from dotenv import load_dotenv

load_dotenv()

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAM_DIR  = os.path.join(BASE, "streaming")
BUILD_DIR   = os.path.join(STREAM_DIR, "build")
ZIP_PATH    = os.path.join(STREAM_DIR, "lambda_package.zip")

AWS_REGION      = "eu-west-2"
LAMBDA_NAME     = "london-property-radar-scorer"
LAMBDA_ROLE_NAME = "london-property-radar-lambda-role"
S3_BUCKET       = os.getenv("S3_BUCKET_NAME", "london-property-radar")

ENV_VARS = {
    "S3_BUCKET_NAME":   os.getenv("S3_BUCKET_NAME"),
    "DB_HOST":          os.getenv("DB_HOST"),
    "DB_PORT":          os.getenv("DB_PORT", "5432"),
    "DB_NAME":          os.getenv("DB_NAME"),
    "DB_USER":          os.getenv("DB_USER"),
    "DB_PASSWORD":      os.getenv("DB_PASSWORD"),
    "APIFY_API_TOKEN":  os.getenv("APIFY_API_TOKEN"),
}

DEPENDENCIES = [
    "scikit-learn==1.7.2",
    "pandas==2.2.3",
    "numpy==1.26.4",
    "joblib",
    "psycopg2-binary",
    "requests",
]


def build_package():
    print("Building Lambda deployment package...")
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    # Install dependencies into build dir
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        *DEPENDENCIES,
        "--target", BUILD_DIR,
        "--quiet",
        "--platform", "manylinux2014_x86_64",
        "--python-version", "3.12",
        "--only-binary=:all:",
    ])

    # scipy and scipy.libs must both stay - scikit-learn 1.7.2 requires them at import time

    # Remove __pycache__, tests, and .dist-info to save space
    for root, dirs, files in os.walk(BUILD_DIR, topdown=True):
        for d in list(dirs):
            if d in ("__pycache__", "tests", "test"):
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)
            elif d.endswith(".dist-info"):
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)

    unzipped_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(BUILD_DIR) for f in fs
    ) / 1024 / 1024
    print(f"Build dir after cleanup: {unzipped_mb:.0f} MB")

    # Copy Lambda function code
    shutil.copy(os.path.join(STREAM_DIR, "lambda_function.py"), BUILD_DIR)

    # Zip everything
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(BUILD_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                arcname   = os.path.relpath(full_path, BUILD_DIR)
                zf.write(full_path, arcname)

    size_mb = os.path.getsize(ZIP_PATH) / 1_048_576
    print(f"Package built: {ZIP_PATH} ({size_mb:.1f} MB)")
    return ZIP_PATH


def get_or_create_role(iam):
    """Create IAM role for Lambda with S3 + RDS + CloudWatch permissions."""
    try:
        role = iam.get_role(RoleName=LAMBDA_ROLE_NAME)
        print(f"Using existing role: {LAMBDA_ROLE_NAME}")
        return role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"Creating IAM role: {LAMBDA_ROLE_NAME}")
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }]
    })
    role = iam.create_role(RoleName=LAMBDA_ROLE_NAME, AssumeRolePolicyDocument=trust)
    role_arn = role["Role"]["Arn"]

    # Attach policies
    for policy in [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
    ]:
        iam.attach_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyArn=policy)

    import time; time.sleep(10)  # IAM propagation delay
    return role_arn


def deploy_lambda(zip_path, role_arn):
    s3_key = "lambda/lambda_package.zip"
    print(f"Uploading package to s3://{S3_BUCKET}/{s3_key} ...")
    from boto3.s3.transfer import TransferConfig
    s3 = boto3.client("s3", region_name="eu-west-2")
    transfer_cfg = TransferConfig(multipart_chunksize=50 * 1024 * 1024, max_concurrency=4)
    s3.upload_file(zip_path, S3_BUCKET, s3_key, Config=transfer_cfg)
    print("Upload complete.")

    lmb = boto3.client("lambda", region_name=AWS_REGION)

    try:
        # Update existing function
        lmb.update_function_code(
            FunctionName=LAMBDA_NAME,
            S3Bucket=S3_BUCKET,
            S3Key=s3_key,
        )
        import time
        waiter = lmb.get_waiter("function_updated")
        waiter.wait(FunctionName=LAMBDA_NAME)
        lmb.update_function_configuration(
            FunctionName=LAMBDA_NAME,
            Environment={"Variables": ENV_VARS},
            Timeout=300,
            MemorySize=512,
        )
        print(f"Updated Lambda function: {LAMBDA_NAME}")
    except lmb.exceptions.ResourceNotFoundException:
        # Create new function
        lmb.create_function(
            FunctionName=LAMBDA_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"S3Bucket": S3_BUCKET, "S3Key": s3_key},
            Timeout=300,
            MemorySize=512,
            Environment={"Variables": ENV_VARS},
        )
        print(f"Created Lambda function: {LAMBDA_NAME}")

    fn = lmb.get_function(FunctionName=LAMBDA_NAME)
    return fn["Configuration"]["FunctionArn"]


def setup_eventbridge(fn_arn):
    """Create EventBridge rule to trigger Lambda every 6 hours."""
    events = boto3.client("events", region_name=AWS_REGION)
    lmb    = boto3.client("lambda", region_name=AWS_REGION)

    rule_name = "london-property-radar-6h"
    rule = events.put_rule(
        Name=rule_name,
        ScheduleExpression="rate(6 hours)",
        State="ENABLED",
        Description="Triggers London Property Radar scorer every 6 hours",
    )
    rule_arn = rule["RuleArn"]
    print(f"EventBridge rule created: {rule_name}")

    # Allow EventBridge to invoke Lambda
    try:
        lmb.add_permission(
            FunctionName=LAMBDA_NAME,
            StatementId="EventBridgeTrigger",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
    except lmb.exceptions.ResourceConflictException:
        pass  # permission already exists

    events.put_targets(
        Rule=rule_name,
        Targets=[{"Id": "LambdaTarget", "Arn": fn_arn}],
    )
    print("EventBridge → Lambda trigger wired up.")


if __name__ == "__main__":
    iam      = boto3.client("iam")
    role_arn = get_or_create_role(iam)

    zip_path = build_package()
    fn_arn   = deploy_lambda(zip_path, role_arn)
    setup_eventbridge(fn_arn)

    print("\nStreaming pipeline deployed.")
    print(f"  Lambda: {LAMBDA_NAME}")
    print(f"  Schedule: every 6 hours")
    print(f"  Next step: add APIFY_API_TOKEN to .env and redeploy")
