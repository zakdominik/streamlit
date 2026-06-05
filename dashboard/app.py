#imports all used libraries
import os
import datetime
import boto3
import psycopg2
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

#loads dotenv file for credentials for local deployment
load_dotenv()

#on Streamlit Community Cloud credentials come from st.secrets so this loads it from there instead of dotenv
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

#page config
st.set_page_config(
    page_title="London Property Radar",
    page_icon="🏠",
    layout="wide",
)

#opens a connection to the RDS database using credentials
def _connect():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT")),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=10,
    )


#single query that joins all 5 normalised tables into one flat result
LISTINGS_QUERY = """
    SELECT
        l.listing_id,
        l.listing_url,
        l.address,
        p.postcode,
        n.name AS neighborhood,
        l.price,
        l.bedrooms,
        l.bathrooms,
        l.floor_area,
        pt.property_type,
        l.latitude,
        l.longitude,
        l.added_at,
        al.predicted_value,
        al.valuation_ratio,
        al.score,
        al.flagged
    FROM listings l
    JOIN analysed_listings al ON l.listing_id = al.listing_id
    JOIN neighborhoods n ON l.neighborhood = n.neighborhood_id
    JOIN property_types pt ON l.property_type = pt.property_type_id
    LEFT JOIN postcodes p ON l.postcode = p.postcode_id
"""

#loads flagged table
def load_flagged():
    conn = _connect()
    df = pd.read_sql(LISTINGS_QUERY + " WHERE al.flagged = TRUE ORDER BY al.score DESC", conn)
    conn.close()
    return df

#loads the table of all listings
def load_all_listings():
    conn = _connect()
    df = pd.read_sql(LISTINGS_QUERY + " ORDER BY l.added_at DESC, al.score DESC", conn)
    conn.close()
    return df


#reads the last Lambda CloudWatch log timestamp and adds 6 hours to estimate the next run time
def get_next_run_time():
    try:
        client = boto3.client("logs", region_name="eu-west-2")
        streams = client.describe_log_streams(
            logGroupName="/aws/lambda/london-property-radar-scorer",
            orderBy="LastEventTime",
            descending=True,
            limit=1,
        )
        ts = streams["logStreams"][0]["lastIngestionTime"]/1000
        last = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return last+datetime.timedelta(hours=6)
    except Exception:
        return None


#formatting helpers, returns a dash for missing values. good because it doesnt crash the website. plus formats the numbers
def fmt_price(v):
    return "—" if pd.isna(v) else f"£{int(v):,}"
def fmt_ratio(v):
    return "—" if pd.isna(v) else f"{v:.2f}x"
def fmt_area(v):
    return "—" if pd.isna(v) else f"{v:.0f} m²"
def fmt_score(v):
    return "—" if pd.isna(v) else f"{v:.1f}"


#formats and renders a df as a Streamlit table with clickable listing links
def render_table(df, ts_col):
    d = df.copy()
    d["price"] = d["price"].apply(fmt_price)
    d["predicted_value"] = d["predicted_value"].apply(fmt_price)
    d["valuation_ratio"] = d["valuation_ratio"].apply(fmt_ratio)
    d["floor_area"] = d["floor_area"].apply(fmt_area)
    d["score"] = d["score"].apply(fmt_score)
    d[ts_col] = pd.to_datetime(d[ts_col]).dt.strftime("%d %b %Y %H:%M")
    d["link"] = d["listing_url"]

    #renames columns for better UI experience
    d = d.rename(columns={
        "address": "Address",
        "neighborhood": "Neighbourhood",
        "postcode": "Postcode",
        "price": "Price",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "floor_area": "Area",
        "property_type": "Type",
        "predicted_value": "Model Value",
        "valuation_ratio": "Val. Ratio",
        "score": "Score",
        ts_col: "Timestamp",
        "link": "Link",
    })

    cols = ["Address", "Postcode", "Neighbourhood", "Price", "Beds", "Baths", "Area", "Type", "Model Value", "Val. Ratio", "Score", "Timestamp", "Link"]

    #flagged column is only shown on the all listings tab and not the flagged deals tab
    if "flagged" in df.columns:
        d = d.rename(columns={"flagged": "Flagged"})
        cols = ["Flagged"]+cols

    st.dataframe(
        d[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Link", display_text="View"),
            "Flagged": st.column_config.CheckboxColumn("Flagged", disabled=True),
        },
    )


#page header which has title on the left and next pipeline run countdown on the right
col_title, col_next = st.columns([3, 1])
#page title, description
with col_title:
    st.title("London Property Radar")
    st.caption("Automated Investment Valuation Tool. Refreshed every 6 hours from Rightmove")

#renders the next pipeline run countdown display
with col_next:
    next_run = get_next_run_time() #gets the next run time
    if next_run:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        delta = next_run-now
        mins = int(delta.total_seconds()//60)
        #if overdue
        if delta.total_seconds() < 0:
            run_str = "overdue, running soon"
        #if less than 60 min
        elif mins < 60:
            run_str = f"in {mins} min"
        #if more than an hour
        else:
            h, m = divmod(mins, 60)
            run_str = f"in {h}h {m}m"
        st.markdown(
            f"""
            <div style="text-align:right; padding-top:1.2rem;">
                <div style="font-size:0.8rem; color:#888; text-transform:uppercase; letter-spacing:0.05em;">Next pipeline run</div>
                <div style="font-size:1.8rem; font-weight:700; line-height:1.2;">{next_run.strftime('%H:%M')} <span style="font-size:1rem; font-weight:400;">UTC</span></div>
                <div style="font-size:1rem; color:#f59e0b; font-weight:600;">{run_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

#load data from the database 
try:
    flagged_df = load_flagged()
    all_df = load_all_listings()
#stops the app if fails to connect
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.stop()

#sidebar filters that is built from the actual data
with st.sidebar:
    st.header("Filters")
    st.caption("Applied to both tabs")
    #lets user select specific neighbourhood
    all_neighborhoods = sorted(set(flagged_df["neighborhood"].dropna().tolist()+all_df["neighborhood"].dropna().tolist()))
    selected_neighborhoods = st.multiselect("Neighbourhood", all_neighborhoods, default=[])
    #minimal score
    min_score = st.slider("Min score", 0, 100, 0, step=5)
    #how many bedrooms filter
    bedroom_opts = sorted(set(flagged_df["bedrooms"].dropna().astype(int).tolist() +all_df["bedrooms"].dropna().astype(int).tolist()))
    bedroom_filter = st.selectbox("Bedrooms", ["Any"] + [str(b) for b in bedroom_opts])
    #type of property (flat or etc.)
    type_opts = sorted(set(flagged_df["property_type"].dropna().tolist() +all_df["property_type"].dropna().tolist()))
    selected_types = st.multiselect("Property type", type_opts, default=[])


#applies the sidebar filters to a dataframe
def apply_filters(df):
    out = df.copy()
    if selected_neighborhoods:
        out = out[out["neighborhood"].isin(selected_neighborhoods)]
    out = out[out["score"].fillna(0) >= min_score]
    if bedroom_filter != "Any":
        out = out[out["bedrooms"] == int(bedroom_filter)]
    if selected_types:
        out = out[out["property_type"].isin(selected_types)]
    return out


filtered_flagged = apply_filters(flagged_df)
filtered_all = apply_filters(all_df)

#flagged deals and all listings tabs
tab1, tab2 = st.tabs([
    f"Flagged Deals ({len(filtered_flagged)})",
    f"All Listings ({len(filtered_all)})",
])

#flagged deals tab
with tab1:
    #if empty, dont show anything
    if filtered_flagged.empty:
        st.info("No flagged deals match the current filters.")
    else:
        #summary metrics at the top
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deals", len(filtered_flagged))
        c2.metric("Avg score", f"{filtered_flagged['score'].mean():.1f}")
        c3.metric("Avg val. ratio", fmt_ratio(filtered_flagged["valuation_ratio"].mean()))
        c4.metric("Median price", fmt_price(filtered_flagged["price"].median()))
        st.divider()

        #map coloured by investment score, only shown if listings have latitude and longitude
        map_df = filtered_flagged.dropna(subset=["latitude", "longitude"]).copy()
        map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})
        if map_df.empty:
            st.info("Map unavailable as no location data for current listings yet.")
        else:
            map_df["price_fmt"] = map_df["price"].apply(fmt_price)
            map_df["ratio_fmt"] = map_df["valuation_ratio"].apply(fmt_ratio)
            map_df["_size"] = 6
            fig = px.scatter_mapbox(
                map_df, lat="lat", lon="lon",
                color="score", size="_size", size_max=8,
                color_continuous_scale=[[0, "#818cf8"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                range_color=[0, 100],
                opacity=1.0,
                hover_name="address",
                hover_data={
                    "price_fmt": True,
                    "ratio_fmt": True,
                    "score": True,
                    "neighborhood": True,
                    "lat": False, "lon": False, "_size": False,
                },
                labels={
                    "price_fmt": "Price", "ratio_fmt": "Val. ratio",
                    "score": "Score", "neighborhood": "Neighbourhood",
                },
                mapbox_style="open-street-map", zoom=10,
                center={"lat": map_df["lat"].mean(), "lon": map_df["lon"].mean()},
                height=450,
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              coloraxis_colorbar_title="Score")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        #table of all flagged deals
        st.subheader("Flagged deals")
        render_table(filtered_flagged, "added_at")

#tab2 for all listings fetched
with tab2:
    if filtered_all.empty:
        st.info("No listings in the database yet. The pipeline runs every 6 hours.")
    else:
        #summary metrics at the top
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total listings", len(filtered_all))
        c2.metric("Flagged", int(filtered_all["flagged"].sum()) if "flagged" in filtered_all.columns else "—")
        c3.metric("Avg val. ratio", fmt_ratio(filtered_all["valuation_ratio"].mean()))
        c4.metric("Median price", fmt_price(filtered_all["price"].median()))

        st.divider()
        st.subheader("All scored listings")
        st.caption(
            "Every listing fetched from Rightmove, scored by the Random Forest model. "
            "Flagged means that the model predicts property is worth at least 10% above asking price."
        )
        render_table(filtered_all, "added_at")

st.divider()
st.caption(
    "**Disclaimer:** This is an analytical tool, not financial advice! "
    "Valuation predictions are based on a Random Forest model trained on London property data and carry an error of +-14% "
)
