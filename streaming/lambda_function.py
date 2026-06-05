#imports all used libraries
import os
import re
import json
import boto3
import joblib
import requests
import pandas as pd
import psycopg2

#S3 bucket and all relevant file paths, model and neighborhoods
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_S3_KEY = "model/rf_model.pkl"
NEIGHBORHOODS_S3_KEY = "data/neighborhoods.json"
#tmp is the only folder where files can be saved so the files are saved there
MODEL_TMP_PATH = "/tmp/rf_model.pkl"
NEIGHBORHOODS_TMP_PATH = "/tmp/neighborhoods.json"

#database connection variables
DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ.get("DB_PORT"))
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
#Apify credentials and the actor used for scraping
APIFY_TOKEN = os.environ["APIFY_API_TOKEN"]
APIFY_ACTOR = "automation-lab~rightmove-scraper"
APIFY_BASE_URL = "https://api.apify.com/v2"

#number of listings to be fetched per neighborhood run
ITEMS_PER_NEIGHBORHOOD = 10
#how long to wait for apify (s) to run before interrupting it (this is because there is 300s limit for lambda and we have 10 runs so 10*25=250s which is just under)
APIFY_WAIT_SECONDS = 25
#property gets flagged if it is at least 10% undervalued
VALUATION_THRESHOLD = 1.10

#downloads the model from S3 and saves it to memory
def load_model():
    print("Downloading model from S3.")
    boto3.client("s3").download_file(S3_BUCKET, MODEL_S3_KEY, MODEL_TMP_PATH)
    return joblib.load(MODEL_TMP_PATH)

#downloads the neighborhoods json from S3 and returns it as a list
def load_neighborhoods():
    print("Downloading neighborhoods from S3")
    boto3.client("s3").download_file(S3_BUCKET, NEIGHBORHOODS_S3_KEY, NEIGHBORHOODS_TMP_PATH)
    with open(NEIGHBORHOODS_TMP_PATH) as f:
        return json.load(f)

#opens a connection to the database on RDS
def get_db_conn():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,user=DB_USER, password=DB_PASSWORD, connect_timeout=10)

#loads all listing URLs in the db, used for skipping already analysed listings
def load_existing_urls(conn):
    cur = conn.cursor()
    cur.execute("SELECT listing_url FROM listings")
    urls = {row[0] for row in cur.fetchall()}
    cur.close()
    print(f"Existing listings in DB: {len(urls)}")
    return urls

#loads all 3 lookup tables (neighborhoods, property_types, postcodes) so they are saved in memory for faster processing
def load_lookup_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name, neighborhood_id FROM neighborhoods")
    neighborhoods = {row[0]: row[1] for row in cur.fetchall()}
    cur.execute("SELECT property_type, property_type_id FROM property_types")
    property_types = {row[0]: row[1] for row in cur.fetchall()}
    cur.execute("SELECT postcode, postcode_id FROM postcodes")
    postcodes = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    return neighborhoods, property_types, postcodes

#returns the postcode_id for a given postcode, if the postcode isn't in the table yet it gets inserted first
def get_or_create_postcode(cur, postcode, postcodes_cache):
    if not postcode:
        return None
    #returns postcode if postcode available
    if postcode in postcodes_cache:
        return postcodes_cache[postcode]
    #inserts new postcode to the db
    cur.execute("INSERT INTO postcodes (postcode) VALUES (%s) ON CONFLICT (postcode) DO NOTHING",(postcode,))
    #gets the postcode_id
    cur.execute("SELECT postcode_id FROM postcodes WHERE postcode = %s", (postcode,))
    postcode_id = cur.fetchone()[0]
    #adds it to the postcodes cache 
    postcodes_cache[postcode] = postcode_id
    return postcode_id


#same logic as for postcodes, just for property_types now
def get_or_create_property_type(cur, prop_type, property_types_cache):
    if prop_type in property_types_cache:
        return property_types_cache[prop_type]
    cur.execute("INSERT INTO property_types (property_type) VALUES (%s) ON CONFLICT (property_type) DO NOTHING",(prop_type,))
    cur.execute("SELECT property_type_id FROM property_types WHERE property_type = %s", (prop_type,))
    property_id = cur.fetchone()[0]
    property_types_cache[prop_type] = property_id
    return property_id


#calls the Rightmove API to scrape the 10 newest Rightmove listings for a given neighbourhood
def fetch_listings_for_neighborhood(neighborhood, headers):
    print(f"Fetching listings for: {neighborhood}")
    try:
        #POST request starts a new scraper run (max 25s as defined before), lambda needs to run under 300s
        run_resp = requests.post(
            f"{APIFY_BASE_URL}/acts/{APIFY_ACTOR}/runs",
            params={"waitForFinish": APIFY_WAIT_SECONDS},
            headers=headers,
            json={
                "searchLocation": neighborhood,
                "searchType": "sale",
                "maxItems": ITEMS_PER_NEIGHBORHOOD,
            },
            timeout=APIFY_WAIT_SECONDS,
        )
        run_resp.raise_for_status()
        run = run_resp.json().get("data", {})
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            return []

        #fetch the actual listing data from the dataset
        items_resp = requests.get(
            f"{APIFY_BASE_URL}/datasets/{dataset_id}/items",
            headers=headers,
            params={"format": "json", "limit": ITEMS_PER_NEIGHBORHOOD},
            timeout=15,
        )
        items_resp.raise_for_status()
        listings = items_resp.json()
        #add neighborhood name to each listing
        for listing in listings:
            listing["_neighborhood"] = neighborhood
        print(f"{neighborhood}: got {len(listings)} listings")
        return listings

    #skips the run if there is any error
    except Exception as e:
        print(f"Failed to fetch listings for {neighborhood}: {e}")
        return []

#Disclaimer: This function was written by a generative AI model to save time on the parsing
#extracts the UK postcode from the end of a Rightmove display address string
#e.g. "Flat 2, Broadway Market, London E8 4QJ" -> "E84QJ"
def _extract_postcode(address):
    match = re.search(r"([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})$", address.strip().upper())
    return match.group(1).replace(" ", "") if match else None

#Disclaimer: This function was also written by AI to save time on the parsing logic
#extracts floor area in m2 from Rightmove's displaySize string and handles both m2 and ft2 formats as Rightmove uses both
def _extract_floor_area(display_size):
    if not display_size:
        return None
    s = str(display_size).lower()
    #convert sq ft to m2 (1 sq ft = 0.0929 m2)
    if "sq. ft" in s or "sqft" in s:
        nums = re.findall(r"[\d,]+", display_size)
        if nums:
            return round(float(nums[0].replace(",", "")) * 0.0929, 1)
    nums = re.findall(r"[\d,]+", display_size)
    if nums:
        val = float(nums[0].replace(",", ""))
        if val > 5:
            return val
    return None


#maps Rightmove property type strings to the 3 categories used in training data, anything else is Unknown which the model can handle
def map_property_type(raw):
    #if no category, assume that it is apartment
    if not raw:
        return "Apartment"
    raw = str(raw).lower()
    if "flat" in raw or "apartment" in raw or "maisonette" in raw or "studio" in raw:
        return "Apartment"
    if "detached" in raw and "semi" not in raw:
        return "Detached House"
    if "semi" in raw:
        return "Semi-Detached"
    return "Unknown"


#takes a single Rightmove listing dict, then runs the model on it and returns a scored result dict - returns None if error or should be skipped
def score_listing(listing, model):
    try:
        price = listing.get("price")
        #if no price then we cant score it so return None
        if not price:
            return None
        price = int(price)

        #extracts all fields from the listing 
        neighborhood = listing["_neighborhood"]
        address = listing.get("displayAddress") or ""
        postcode = _extract_postcode(address) #uses the helper function to get the postcode
        bedrooms = int(listing.get("bedrooms") or 1)
        bathrooms = int(listing.get("bathrooms") or 1)
        floor_area = _extract_floor_area(listing.get("displaySize", ""))
        prop_type = map_property_type(listing.get("propertySubType"))
        url = listing.get("url") or f"https://www.rightmove.co.uk/properties/{listing.get('propertyId', '')}"
        latitude = listing.get("latitude")
        longitude = listing.get("longitude")

        #only run the model if we have a floor area, it's required and sometimes listing don't have it. it's an important feature
        if floor_area and floor_area > 0:
            features = pd.DataFrame([{
                "square_meters": floor_area,
                "bedrooms": float(bedrooms),
                "bathrooms": float(bathrooms),
                "neighborhood": neighborhood,
                "property_type": prop_type,
            }])
            predicted_value = float(model.predict(features)[0])
            valuation_ratio = predicted_value/price
        else:
            #no floor area = no prediction, listing will get stored but without prediction
            predicted_value = None
            valuation_ratio = None

        #flags property if model thinks the property is worth more than 10% above asking price
        should_flag = valuation_ratio is not None and valuation_ratio > VALUATION_THRESHOLD
        #score represents how underpriced the property is (0-100) where 100 means 30% underpriced
        if valuation_ratio is not None:
            score = max(min((valuation_ratio-1.0)/0.3, 1.0)*100, 0.0)
        else:
            score = 0.0

        #returns a dict with everything needed to write to the db
        return {
            "should_flag": should_flag,
            "listing_url": url,
            "address": address[:255],
            "postcode": postcode,
            "price": price,
            "neighborhood": neighborhood,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floor_area": floor_area,
            "property_type": prop_type,
            "predicted_value": round(predicted_value, 2) if predicted_value else None,
            "valuation_ratio": round(valuation_ratio, 4) if valuation_ratio else None,
            "score": round(score, 2),
            "latitude": float(latitude) if latitude is not None else None,
            "longitude": float(longitude) if longitude is not None else None,
        }
    except Exception as e:
        #if scoring fails for any reason, skip the listing rather than crashing the whole run
        print(f"Failed to score listing: {e}")
        return None


#writes all scored listings to the database, listings get property data and analysed_listings get the model scores etc.
def write_to_rds(conn, scored, neighborhoods_cache, property_types_cache, postcodes_cache):
    if not scored:
        print("No new listings to write.")
        return

    cur = conn.cursor()
    inserted = 0
    for r in scored:
        #resolves all foreign key IDs before inserting
        neighborhood_id = neighborhoods_cache.get(r["neighborhood"])
        property_type_id = get_or_create_property_type(cur, r["property_type"], property_types_cache)
        postcode_id = get_or_create_postcode(cur, r["postcode"], postcodes_cache)

        #insert into listings, if URL exists then it skips it
        cur.execute("""
            INSERT INTO listings
                (listing_url, address, postcode, price, neighborhood,
                 bedrooms, bathrooms, floor_area, property_type,
                 latitude, longitude)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (listing_url) DO NOTHING
            RETURNING listing_id
        """, (
            r["listing_url"], r["address"], postcode_id, r["price"], neighborhood_id,
            r["bedrooms"], r["bathrooms"], r["floor_area"], property_type_id,
            r["latitude"], r["longitude"],
        ))
        row = cur.fetchone()
        #skips writing to analysed listings if it isn't scored
        if row is None:
            continue
        listing_id = row[0]

        #inserts the ML output into analysed_listings using listing_id
        cur.execute("""
            INSERT INTO analysed_listings
                (listing_id, predicted_value, valuation_ratio, score, flagged)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            listing_id,
            r["predicted_value"], r["valuation_ratio"],
            r["score"], r["should_flag"],
        ))
        inserted += 1

    #commits all inserted rows at once
    conn.commit()
    flagged_count = sum(1 for r in scored if r["should_flag"]) #gets the count of all flagged properties
    print(f"Inserted {inserted} new listings ({flagged_count} flagged).")
    cur.close()


#the entry point that AWS calls every 6 hours
def lambda_handler(event, context):
    print("Lambda started.")

    #load model and neighbourhoods files
    model = load_model()
    neighborhoods = load_neighborhoods()
    print(f"Neighbourhoods: {neighborhoods}")
    #open DB connection and load lookup tables and existing URLs into memory
    conn = get_db_conn()
    existing_urls = load_existing_urls(conn)
    neighborhoods_cache, property_types_cache, postcodes_cache = load_lookup_tables(conn)

    #Apify requires this header for every request
    headers = {"Authorization": f"Bearer {APIFY_TOKEN}"}

    #fetch listings for each neighbourhood and filter out ones already done
    all_listings = []
    for neighborhood in neighborhoods:
        listings = fetch_listings_for_neighborhood(neighborhood, headers)
        new_listings = [l for l in listings if (l.get("url") or "") not in existing_urls]
        print(f"{neighborhood}: {len(listings)} fetched, {len(new_listings)} new")
        all_listings.extend(new_listings)
    #scores every new listing
    scored = [score_listing(l, model) for l in all_listings]
    scored = [s for s in scored if s is not None]
    print(f"Scored: {len(scored)}")
    print(f"Flagged: {sum(1 for s in scored if s['should_flag'])}")

    #writes everything to the database
    write_to_rds(conn, scored, neighborhoods_cache, property_types_cache, postcodes_cache)
    conn.close()

    #returns a summary for logging
    return {
        "statusCode": 200,
        "listings_fetched": len(all_listings),
        "listings_scored": len(scored),
        "deals_flagged": sum(1 for s in scored if s["should_flag"]),
    }
