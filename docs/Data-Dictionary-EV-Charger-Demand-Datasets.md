# Data Dictionary for EV Charger Demand Datasets

This document provides an overview of the datasets used for the SIT764 EV Charger Demand and Infrastructure Planning project. It describes the purpose of each file and outlines the associated data schema, including column names, data types, units, descriptions, requirements, source links, and known limitations.

The datasets referenced in this document are stored in the project data folder and are processed by the notebook `Collecting_Data_EV_Charger_Demand (1).ipynb`. The information in this document was reviewed against the uploaded project files on **8 June 2026** and reflects the current dataset state at the time of review.

**Note:** Some datasets are raw public datasets downloaded from external open data sources, while others are project-generated outputs created through web scraping, preprocessing, or data integration.

---

## Dataset Overview

| Dataset Group | File Name | Format | Main Purpose | Source Type |
|---|---|---:|---|---|
| Vehicle Registration Data | `AAA-Postcode-Registrations-2024.xlsx` | XLSX | EV adoption and fuel-type registration by postcode | External public data |
| Vehicle Type Data | `vehicle_type.json` | JSON | Vehicle registrations by state, fuel type, manufacturer, model, and quarter | External public / EV index style data |
| EV Charging Infrastructure | `nsw_ev_chargers.xlsx` | XLSX | Existing and planned EV charger locations in NSW | External open data |
| Traffic Station Reference | `road_traffic_counts_station_reference.csv` | CSV | Traffic station metadata and geospatial reference | External open data |
| Hourly Traffic Counts | `road_traffic_counts_hourly_permanent.csv` | CSV | Daily and hourly traffic volume counts from permanent stations | External open data |
| Weather Data | `sydney_2020_2025.csv` | CSV | Historical Sydney weather observations scraped for modelling features | Project-generated scraped data |
| Processing Notebook | `Collecting_Data_EV_Charger_Demand (1).ipynb` | IPYNB | Data extraction, cleaning, linking, and merging pipeline | Project-generated code |
| Derived Dataset | `merge_data.csv` | CSV | Integrated modelling dataset generated from the notebook | Project-generated output |

---

## Vehicle Registration Datasets

These datasets describe vehicle registrations and EV adoption patterns. They are used to estimate EV ownership and market trends that may influence charging demand.

### AAA Postcode Registration Dataset

#### Source

| Item | Detail |
|---|---|
| Source name | Australian Automobile Association - Electric Vehicle Index |
| Source link | [AAA Electric Vehicle Index](https://www.aaa.asn.au/research-data/electric-vehicle/) |
| Source description | The AAA EV Index provides a breakdown of light vehicle registrations in Australia by fuel type and postcode. |
| Project file | `AAA-Postcode-Registrations-2024.xlsx` |
| Relevant sheet | `Registration Numbers` |
| Current file size | 7,136 rows × 7 columns |
| Spatial granularity | Postcode and state |
| Temporal coverage | Registration snapshots as at 31 January 2021, 2022, 2023, and 2024 |
| Main project use | EV adoption proxy by NSW postcode |

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| Postcode | Integer | – | Australian postcode associated with vehicle registration records | Required |
| State | String | – | Australian state or territory abbreviation, such as NSW, VIC, QLD, ACT | Required |
| Fuel Type | String | – | Vehicle fuel group, including BEV, Hybrid/PHEV, and ICE | Required |
| Registrations as at 31 January 2021 | Integer | vehicles | Number of registered vehicles in the fuel group at the postcode level | Required |
| Registrations as at 31 January 2022 | Integer | vehicles | Number of registered vehicles in the fuel group at the postcode level | Required |
| Registrations as at 31 January 2023 | Integer | vehicles | Number of registered vehicles in the fuel group at the postcode level | Required |
| Registrations as at 31 January 2024 | Integer | vehicles | Number of registered vehicles in the fuel group at the postcode level | Required |

#### Project Files

| File Name | Description |
|---|---|
| `AAA-Postcode-Registrations-2024.xlsx` | Postcode-level light vehicle registration data by state and fuel type. Used to filter NSW records and calculate EV adoption indicators based on BEV and Hybrid/PHEV registrations. |

**Note:** The workbook contains a `Disclaimer` sheet and a `Registration Numbers` sheet. The modelling pipeline should explicitly read the `Registration Numbers` sheet.

---

### Vehicle Type Registration Dataset

#### Source

| Item | Detail |
|---|---|
| Source name | EV Index style vehicle registration data |
| Source link | [AAA Electric Vehicle Index](https://www.aaa.asn.au/research-data/electric-vehicle/) |
| Project file | `vehicle_type.json` |
| Current file size | 4,112 rows × 18 columns after parsing nested JSON records |
| Spatial granularity | State level |
| Temporal coverage | Quarterly data from Q1 2022 to Q1 2025 |
| Main project use | Vehicle market trend analysis by fuel type, manufacturer, model, and vehicle segment |

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| STATE | String | – | Australian state or territory name in uppercase | Required |
| VEHICLE TYPE | String | – | Vehicle body or market segment, such as Medium SUV, Small Car, Van, or Ute | Required |
| MANUFACTURER | String | – | Vehicle manufacturer name | Required |
| MODEL | String | – | Vehicle model name | Required |
| FUEL TYPE | String | – | Fuel type, including ICE, BEV, PHEV, Hybrid, and HFCEV | Required |
| Q1 2022 | Integer | vehicles | Number of registrations in quarter 1 of 2022 | Required |
| Q2 2022 | Integer | vehicles | Number of registrations in quarter 2 of 2022 | Required |
| Q3 2022 | Integer | vehicles | Number of registrations in quarter 3 of 2022 | Required |
| Q4 2022 | Integer | vehicles | Number of registrations in quarter 4 of 2022 | Required |
| Q1 2023 | Integer | vehicles | Number of registrations in quarter 1 of 2023 | Required |
| Q2 2023 | Integer | vehicles | Number of registrations in quarter 2 of 2023 | Required |
| Q3 2023 | Integer | vehicles | Number of registrations in quarter 3 of 2023 | Required |
| Q4 2023 | Integer | vehicles | Number of registrations in quarter 4 of 2023 | Required |
| Q1 2024 | Integer | vehicles | Number of registrations in quarter 1 of 2024 | Required |
| Q2 2024 | Integer | vehicles | Number of registrations in quarter 2 of 2024 | Required |
| Q3 2024 | Integer | vehicles | Number of registrations in quarter 3 of 2024 | Required |
| Q4 2024 | Integer | vehicles | Number of registrations in quarter 4 of 2024 | Required |
| Q1 2025 | Integer | vehicles | Number of registrations in quarter 1 of 2025 | Required |

#### Project Files

| File Name | Description |
|---|---|
| `vehicle_type.json` | Nested JSON file containing vehicle registration records as JSON strings. The notebook parses the inner `data` field into a structured DataFrame. |

**Note:** This dataset separates `Hybrid` and `PHEV`, while the postcode registration file groups them as `Hybrid/PHEV`. Fuel-type standardisation is required before comparing or merging the two datasets.

---

## EV Charging Infrastructure Datasets

These datasets describe the location, type, plug availability, and capacity of existing and planned EV charging infrastructure in NSW.

### NSW EV Chargers Dataset

#### Source

| Item | Detail |
|---|---|
| Source name | Transport for NSW Open Data - EV Charging Locations |
| Source link | [EV Charging Locations - Dataset](https://opendata.transport.nsw.gov.au/data/dataset/ev-charging-locations) |
| Source description | The dataset includes destination AC chargers, fast DC chargers, and chargers scheduled for future development in NSW. |
| Project file | `nsw_ev_chargers.xlsx` |
| Current workbook sheets | `Destination Chargers (AC)`, `Fast Chargers (DC)`, `Upcoming Fast Chargers` |
| Spatial granularity | Charging station or planned charging site |
| Main project use | Charger supply, charger coverage, and nearest traffic station matching |

---

### Destination Chargers (AC)

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| ObjId | Integer | – | Unique object identifier for the charger record | Required |
| Station name | String | – | Name of the EV charging location | Required |
| Station address | String | – | Street address of the charging location | Required |
| Opening hours | String | – | Opening hours or access information | Optional |
| Operator | String | – | Charging station operator | Optional |
| Number of station | Integer | stations | Number of charging stations at the location | Required |
| Number of plugs | Integer | plugs | Number of plugs available at the location | Required |
| Charger rating | String | kW | Charger power rating as text | Optional |
| Tesla | Integer | plugs | Number of Tesla-compatible plugs | Required |
| Type 2 | Integer | plugs | Number of Type 2 plugs | Required |
| J-1772 | Integer | plugs | Number of J-1772 plugs | Required |
| Latitude | Float | degrees | Latitude coordinate of the charger location | Required |
| Longitude | Float | degrees | Longitude coordinate of the charger location | Required |

#### Project Files

| File Name | Sheet Name | Description |
|---|---|---|
| `nsw_ev_chargers.xlsx` | `Destination Chargers (AC)` | Existing destination AC charging stations in NSW. Current uploaded sheet contains 1,011 records and 13 columns. |

---

### Fast Chargers (DC)

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| ObjId | Integer | – | Unique object identifier for the charger record | Required |
| Station name | String | – | Name of the fast charging location | Required |
| Station address | String | – | Street address of the fast charging location | Required |
| Opening hours | String | – | Opening hours or access information | Optional |
| Operator | String | – | Charging station operator | Optional |
| Number of station | Integer | stations | Number of charging stations at the location | Required |
| Number of plugs | Integer | plugs | Number of plugs available at the location | Required |
| Charger rating | String | kW | Charger power rating as text | Optional |
| CHAdeMO | Integer | plugs | Number of CHAdeMO plugs | Required |
| CCS/SAE | Integer | plugs | Number of CCS/SAE plugs | Required |
| Tesla(Fast) | Integer | plugs | Number of Tesla fast-charging plugs | Required |
| Latitude | Float | degrees | Latitude coordinate of the charger location | Required |
| Longitude | Float | degrees | Longitude coordinate of the charger location | Required |

#### Project Files

| File Name | Sheet Name | Description |
|---|---|---|
| `nsw_ev_chargers.xlsx` | `Fast Chargers (DC)` | Existing DC fast charging stations in NSW. Current uploaded sheet contains 252 records and 13 columns. |

---

### Upcoming Fast Chargers

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| Site Address | String | – | Address of the planned fast charging site | Required |
| lat | Float | degrees | Latitude coordinate of the planned site | Required |
| lng | Float | degrees | Longitude coordinate of the planned site | Required |
| Applicant | String | – | Organisation or applicant associated with the planned charger | Optional |
| Charging_bays | Integer | bays | Number of planned charging bays | Required |
| Charger_capacities | String | kW | Planned charger capacity information | Optional |
| ObjId | Integer | – | Unique object identifier for the planned charger record | Required |
| Postcodes | Integer | – | Postcode of the planned charging site | Required |
| ZoneType | String | – | Zone classification for the planned charger | Optional |
| Round | Integer | – | Funding or project round identifier | Optional |

#### Project Files

| File Name | Sheet Name | Description |
|---|---|---|
| `nsw_ev_chargers.xlsx` | `Upcoming Fast Chargers` | Planned fast charging infrastructure in NSW. Current uploaded sheet contains 187 records and 10 columns. |

**Important note:** In the current notebook, `pd.read_excel('/content/drive/MyDrive/SIT764/Data/nsw_ev_chargers.xlsx')` reads only the first sheet by default. To use the full charging infrastructure dataset, the pipeline should explicitly read all three sheets.

---

## Traffic Datasets

These datasets provide road traffic count information and traffic station metadata. They are used as a proxy for road usage and potential charging demand near EV charger locations.

### Traffic Station Reference Dataset

#### Source

| Item | Detail |
|---|---|
| Source name | NSW Roads Traffic Volume Counts API |
| Source link | [NSW Roads Traffic Volume Counts API - NSW Data](https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api) |
| Alternative source link | [NSW Roads Traffic Volume Counts API - Transport Open Data](https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api) |
| Source description | The Traffic Collection Station Reference table provides station descriptions, geospatial coordinates, road names, suburbs, postcodes, device types, and data quality ratings. |
| Project file | `road_traffic_counts_station_reference.csv` |
| Current file size | 1,783 rows × 42 columns |
| Spatial granularity | Traffic collection station |
| Main project use | Linking traffic station metadata to hourly traffic counts and EV charger locations |

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| the_geom | Float | – | Geometry placeholder from source export | Optional |
| cartodb_id | Integer | – | CartoDB internal record identifier | Optional |
| the_geom_webmercator | Float | – | Web Mercator geometry placeholder from source export | Optional |
| record_id | Float | – | Source record identifier; currently empty in the uploaded file | Optional |
| station_key | Integer | – | Key used to join station metadata with traffic count records | Required |
| station_id | Integer | – | Traffic station identifier used in the notebook for charger-station matching | Required |
| name | String | – | Station name or location label | Optional |
| road_name | String | – | Road name where the station is located | Optional |
| full_name | String | – | Full station description including road and relative location | Optional |
| common_road_name | String | – | Common road name | Optional |
| secondary_name | String | – | Secondary road or landmark description | Optional |
| road_name_base | String | – | Base road name without road type | Optional |
| road_name_type | String | – | Road type, such as Road, Street, Avenue, or Highway | Optional |
| intersection | String | – | Nearby intersection or reference road | Optional |
| distance_to_intersection | Integer | metres | Distance from station to nearby intersection | Optional |
| road_number | Integer | – | Road number identifier | Optional |
| link_number | Integer | – | Road link number | Optional |
| mab_way_type | String | – | Map and bridge way type | Optional |
| mab_way_number | Float | – | Map and bridge way number | Optional |
| mab_identifier | String | – | Map and bridge identifier | Optional |
| road_functional_hierarchy | String | – | Functional hierarchy of the road | Optional |
| road_on_type | String | – | Road location type, such as OnGround | Optional |
| lane_count | String | lanes | Lane count category | Optional |
| road_classification_type | String | – | Road classification type | Optional |
| road_classification_admin | String | – | Road administration classification | Optional |
| rms_region | String | – | NSW RMS region | Optional |
| lga | String | – | Local Government Area | Optional |
| suburb | String | – | Suburb where the traffic station is located | Optional |
| post_code | Integer | – | Postcode where the traffic station is located | Optional |
| device_type | String | – | Type of traffic counting device | Optional |
| heavy_vehicle_checking_station | Boolean | – | Indicates whether the station is a heavy vehicle checking station | Optional |
| permanent_station | Integer | flag | Indicates whether the station is permanent | Optional |
| vehicle_classifier | Integer | flag | Indicates whether the station has vehicle classification capability | Optional |
| lambert_easting | Integer | metres | Lambert projected easting coordinate | Optional |
| lambert_northing | Float | metres | Lambert projected northing coordinate | Optional |
| wgs84_latitude | Float | degrees | Latitude coordinate in WGS84 | Required |
| wgs84_longitude | Float | degrees | Longitude coordinate in WGS84 | Required |
| direction_seq | Integer | – | Direction sequence identifier | Optional |
| quality_rating | Integer | score | Data quality rating from the source system | Optional |
| publish | Boolean | – | Publication flag | Optional |
| md5 | String | – | Source hash identifier | Optional |
| updated_on | String | datetime | Source update timestamp | Optional |

#### Project Files

| File Name | Description |
|---|---|
| `road_traffic_counts_station_reference.csv` | Station reference file used to add station ID, coordinates, suburb, and region information to traffic count records. |

**Note:** The station reference file is metadata only. It does not contain traffic volume counts by itself.

---

### Permanent Hourly Traffic Counts Dataset

#### Source

| Item | Detail |
|---|---|
| Source name | NSW Roads Traffic Volume Counts API |
| Source link | [NSW Roads Traffic Volume Counts API - Transport Open Data](https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api) |
| Source description | The Permanent Hourly Traffic Counts table provides hourly traffic counts for each permanent station at daily level from 2006 onwards. |
| Project file | `road_traffic_counts_hourly_permanent.csv` |
| Current uploaded file size | 1,000,000 rows × 41 columns |
| Date range in uploaded file | 2006-01-01 to 2022-03-29 |
| Main project use | Traffic volume proxy for potential EV charging demand |

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| the_geom | Float | – | Geometry placeholder from source export | Optional |
| cartodb_id | Integer | – | CartoDB internal record identifier | Optional |
| the_geom_webmercator | Float | – | Web Mercator geometry placeholder from source export | Optional |
| record_id | Integer | – | Source record identifier | Optional |
| station_key | Integer | – | Key used to join traffic counts with station reference metadata | Required |
| traffic_direction_seq | Integer | – | Traffic direction sequence | Required |
| cardinal_direction_seq | Integer | – | Cardinal direction sequence | Required |
| classification_seq | Integer | – | Vehicle classification sequence used to filter traffic classes | Required |
| date | DateTime | yyyy-mm-dd | Date of traffic count record | Required |
| year | Integer | year | Calendar year of the traffic count record | Required |
| month | Integer | month | Calendar month of the traffic count record | Required |
| day_of_week | Integer | 1–7 | Day of week code from the source data | Required |
| public_holiday | Boolean | – | Indicates whether the date is a public holiday | Required |
| school_holiday | Boolean | – | Indicates whether the date is a school holiday | Required |
| daily_total | Integer | vehicles/day | Total traffic count across all hourly fields for the day | Required |
| hour_00 | Integer | vehicles/hour | Traffic count for 00:00–00:59 | Optional |
| hour_01 | Integer | vehicles/hour | Traffic count for 01:00–01:59 | Optional |
| hour_02 | Integer | vehicles/hour | Traffic count for 02:00–02:59 | Optional |
| hour_03 | Integer | vehicles/hour | Traffic count for 03:00–03:59 | Optional |
| hour_04 | Integer | vehicles/hour | Traffic count for 04:00–04:59 | Optional |
| hour_05 | Integer | vehicles/hour | Traffic count for 05:00–05:59 | Optional |
| hour_06 | Integer | vehicles/hour | Traffic count for 06:00–06:59 | Optional |
| hour_07 | Integer | vehicles/hour | Traffic count for 07:00–07:59 | Optional |
| hour_08 | Integer | vehicles/hour | Traffic count for 08:00–08:59 | Optional |
| hour_09 | Integer | vehicles/hour | Traffic count for 09:00–09:59 | Optional |
| hour_10 | Integer | vehicles/hour | Traffic count for 10:00–10:59 | Optional |
| hour_11 | Integer | vehicles/hour | Traffic count for 11:00–11:59 | Optional |
| hour_12 | Integer | vehicles/hour | Traffic count for 12:00–12:59 | Optional |
| hour_13 | Integer | vehicles/hour | Traffic count for 13:00–13:59 | Optional |
| hour_14 | Integer | vehicles/hour | Traffic count for 14:00–14:59 | Optional |
| hour_15 | Integer | vehicles/hour | Traffic count for 15:00–15:59 | Optional |
| hour_16 | Integer | vehicles/hour | Traffic count for 16:00–16:59 | Optional |
| hour_17 | Integer | vehicles/hour | Traffic count for 17:00–17:59 | Optional |
| hour_18 | Integer | vehicles/hour | Traffic count for 18:00–18:59 | Optional |
| hour_19 | Integer | vehicles/hour | Traffic count for 19:00–19:59 | Optional |
| hour_20 | Integer | vehicles/hour | Traffic count for 20:00–20:59 | Optional |
| hour_21 | Integer | vehicles/hour | Traffic count for 21:00–21:59 | Optional |
| hour_22 | Integer | vehicles/hour | Traffic count for 22:00–22:59 | Optional |
| hour_23 | Integer | vehicles/hour | Traffic count for 23:00–23:59 | Optional |
| md5 | String | – | Source hash identifier | Optional |
| updated_on | Float/String | datetime | Source update timestamp; mostly empty in current uploaded file | Optional |

#### Project Files

| File Name | Description |
|---|---|
| `road_traffic_counts_hourly_permanent.csv` | Permanent traffic count dataset containing daily total and hourly traffic counts by station and direction. The current uploaded export appears to contain 1,000,000 rows only, ending in March 2022. |

**Important limitation:** The current uploaded file does not cover the full intended modelling window of 2020–2025. If the project aims to model 2020–2025 demand, the traffic dataset should be re-exported or queried to include records beyond March 2022.

---

## Weather Dataset

The weather dataset provides historical weather observations for Sydney. It is used to create daily weather features such as average temperature, humidity, and wind speed.

### Sydney Weather History Dataset

#### Source

| Item | Detail |
|---|---|
| Source name | Timeanddate.com - Past Weather in Sydney |
| Source link | [Past Weather in Sydney, New South Wales, Australia](https://www.timeanddate.com/weather/australia/sydney/historic) |
| Collection method | Web scraping using `requests` and `BeautifulSoup` in the project notebook |
| Project file | `sydney_2020_2025.csv` |
| Current file size | 90,701 rows × 10 raw columns |
| Date range | 2020-01-01 to 2025-06-30 |
| Main project use | Daily weather features for demand modelling |

#### Data Schema

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| Date | Date | yyyy-mm-dd | Calendar date of the weather observation | Required |
| Time | String | hh:mm / text | Observation time from the weather history table | Required |
| Unnamed: 2 | Float | – | Empty column generated during web scraping or CSV export | Optional / Remove |
| Temp | String | °C | Temperature value as text, such as `20 °C` | Required |
| Weather | String | – | Text description of weather conditions | Optional |
| Wind | String | km/h | Wind speed as text, such as `35 km/h` | Optional |
| Unnamed: 6 | String | – | Wind direction symbol from the source table | Optional |
| Humidity | String | % | Humidity value as text, such as `73%` | Optional |
| Barometer | String | mbar | Barometric pressure value as text | Optional |
| Visibility | Float | – | Visibility column; empty in current uploaded file | Optional / Remove |

#### Derived Weather Features

| Derived Column | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| Temp_numeric | Float | °C | Numeric temperature extracted from the `Temp` text field | Required for modelling |
| Humidity_numeric | Float | % | Numeric humidity extracted from the `Humidity` text field | Required for modelling |
| Wind_numeric | Float | km/h | Numeric wind speed extracted from the `Wind` text field | Required for modelling |
| Avg_Temp | Float | °C/day | Daily average temperature after aggregation | Required for modelling |
| Avg_Humidity | Float | %/day | Daily average humidity after aggregation | Required for modelling |
| Avg_Wind | Float | km/h/day | Daily average wind speed after aggregation | Required for modelling |

#### Project Files

| File Name | Description |
|---|---|
| `sydney_2020_2025.csv` | Historical Sydney weather observations scraped month-by-month from Timeanddate.com for January 2020 to June 2025. |

**Note:** This file is a project-generated web-scraped dataset, not a direct official Bureau of Meteorology dataset. It is acceptable for exploratory modelling but should be described transparently in the project documentation.

---

## Processing Notebook

The notebook contains the code used to load, transform, link, and merge the datasets.

### Collecting Data Notebook

#### File Description

| Item | Detail |
|---|---|
| File name | `Collecting_Data_EV_Charger_Demand (1).ipynb` |
| File type | Jupyter Notebook |
| Main libraries | `pandas`, `numpy`, `requests`, `BeautifulSoup`, `sklearn.neighbors.NearestNeighbors`, `matplotlib` |
| Main purpose | Dataset loading, filtering, web scraping, spatial nearest-station matching, weather aggregation, and final merge |
| Output file | `merge_data.csv` |

#### Pipeline Steps

| Step | Description |
|---|---|
| 1 | Load AAA postcode registration workbook and filter NSW records |
| 2 | Select EV-related fuel types, including BEV and Hybrid/PHEV |
| 3 | Load and parse `vehicle_type.json` for market trend analysis |
| 4 | Load NSW EV charger data from `nsw_ev_chargers.xlsx` |
| 5 | Load permanent hourly traffic count records |
| 6 | Filter traffic records using `classification_seq` values `[2, 3]` |
| 7 | Join traffic counts with station reference metadata using `station_key` |
| 8 | Use nearest-neighbour matching to link EV chargers to nearby traffic stations |
| 9 | Scrape Sydney weather history from Timeanddate.com |
| 10 | Aggregate weather observations into daily average temperature, humidity, and wind speed |
| 11 | Merge charger-linked traffic data with daily weather features by date |
| 12 | Export the merged dataset as `merge_data.csv` |

#### Project Files

| File Name | Description |
|---|---|
| `Collecting_Data_EV_Charger_Demand (1).ipynb` | Main data collection and preprocessing notebook used to construct the integrated EV charger demand dataset. |

**Important note:** The current nearest-neighbour logic uses Euclidean distance on latitude and longitude and then multiplies the result by 111 to estimate kilometres. For more accurate geospatial analysis, Haversine distance or a projected coordinate system is recommended.

---

## Derived Dataset

### Integrated EV Charger Demand Dataset

This dataset is generated by the notebook and is not treated as a raw input file.

#### Project Files

| File Name | Description |
|---|---|
| `merge_data.csv` | Integrated dataset produced by merging EV charger records, nearest traffic station records, hourly traffic counts, and daily weather features. |

#### Expected Key Fields

| Column Name | Type | Unit | Description | Requirement |
|---|---|---|---|---|
| Station name | String | – | EV charging station name | Required |
| Station address | String | – | EV charging station address | Required |
| Latitude | Float | degrees | EV charger latitude | Required |
| Longitude | Float | degrees | EV charger longitude | Required |
| nearest_station_id | Integer | – | Nearest traffic station ID assigned to the EV charger | Required |
| distance_km | Float | km | Approximate distance from EV charger to nearest traffic station | Required |
| station_key | Integer | – | Traffic station key linked to hourly traffic data | Required |
| date | DateTime | yyyy-mm-dd | Date used to merge traffic and weather data | Required |
| daily_total | Integer | vehicles/day | Daily total traffic count | Required |
| hour_00 to hour_23 | Integer | vehicles/hour | Hourly traffic count columns | Optional |
| public_holiday | Boolean | – | Public holiday indicator | Required |
| school_holiday | Boolean | – | School holiday indicator | Required |
| suburb | String | – | Suburb of the linked traffic station | Optional |
| rms_region | String | – | NSW RMS region of the linked traffic station | Optional |
| Avg_Temp | Float | °C/day | Daily average temperature | Required |
| Avg_Humidity | Float | %/day | Daily average humidity | Required |
| Avg_Wind | Float | km/h/day | Daily average wind speed | Required |

**Note:** `merge_data.csv` should be regenerated whenever the raw input datasets or preprocessing logic are updated.

---

## Data Integration and Join Logic

### Key Relationships

| Dataset A | Dataset B | Join or Link Method | Purpose |
|---|---|---|---|
| `road_traffic_counts_hourly_permanent.csv` | `road_traffic_counts_station_reference.csv` | `station_key` | Add station coordinates, suburb, and RMS region to traffic count records |
| `nsw_ev_chargers.xlsx` | Traffic station reference / merged traffic data | Nearest-neighbour matching using latitude and longitude | Assign each EV charger to the nearest traffic station |
| `ev_chargers_linked` | `sydney_2020_2025.csv` daily weather output | `date` | Add weather features to charger-linked traffic records |
| `AAA-Postcode-Registrations-2024.xlsx` | Traffic station reference | `Postcode` ↔ `post_code` | Potential postcode-level EV adoption enrichment |
| `AAA-Postcode-Registrations-2024.xlsx` | Upcoming Fast Chargers | `Postcode` ↔ `Postcodes` | Potential link between EV adoption and planned infrastructure |
| `vehicle_type.json` | AAA postcode registrations | State and fuel-type mapping | Compare state-level vehicle market trends with postcode-level EV adoption |

### Recommended Standardisation

| Field | Current Issue | Recommended Treatment |
|---|---|---|
| Fuel type | `Hybrid/PHEV` in AAA file but `Hybrid` and `PHEV` separated in vehicle type JSON | Create a standard fuel category mapping |
| State names | AAA uses abbreviations such as NSW; JSON uses full uppercase names such as NEW SOUTH WALES | Create state-name mapping table |
| Charger power | Charger rating is stored as text and may include ranges | Extract numeric `min_kw` and `max_kw` fields |
| Coordinates | Latitude/longitude are used with Euclidean distance in current code | Use Haversine distance for geospatial matching |
| Date | Traffic date is timezone-aware string; weather date is converted and localised to UTC | Standardise all dates before merging |
| Weather numeric fields | Temperature, wind, and humidity are stored as strings in raw scraped file | Extract numeric values before aggregation |

---

## Data Quality Summary

| Dataset | Current Data Quality Observation | Recommended Action |
|---|---|---|
| `AAA-Postcode-Registrations-2024.xlsx` | Clean structure after reading the correct sheet; no missing values in the uploaded registration sheet | Keep as source-of-truth for postcode EV adoption indicators |
| `vehicle_type.json` | Nested JSON structure requires parsing; fuel categories differ from AAA postcode dataset | Parse inner JSON strings and standardise fuel labels |
| `nsw_ev_chargers.xlsx` | Workbook contains three sheets, but default `read_excel` reads only the first sheet | Explicitly read all relevant sheets |
| `road_traffic_counts_station_reference.csv` | Useful location metadata; `record_id` is empty in uploaded file | Use `station_key` and `station_id`, not `record_id` |
| `road_traffic_counts_hourly_permanent.csv` | Current uploaded file ends at 2022-03-29 and does not cover 2023–2025 | Re-query or re-export traffic records if full 2020–2025 modelling is required |
| `sydney_2020_2025.csv` | Weather fields are stored as text and contain extra scraped columns | Clean numeric fields and remove unnecessary columns |
| `merge_data.csv` | Derived from current notebook logic and depends on nearest-station assumptions | Regenerate after improving charger reading and distance calculation |

---

## Limitations

1. The raw datasets have different spatial granularity. Vehicle registrations are postcode-level, vehicle type registrations are state-level, EV chargers are location-level, traffic counts are station-level, and weather data is city-level for Sydney.

2. The raw datasets have different temporal granularity. AAA registration data is annual snapshot data, vehicle type data is quarterly, traffic data is daily/hourly, and weather observations are sub-daily before aggregation.

3. The current traffic count upload does not provide full coverage to 2025. This is a major limitation if the final modelling target is 2020–2025.

4. The current notebook only reads the first sheet of `nsw_ev_chargers.xlsx` by default. This means DC fast chargers and upcoming fast chargers may be excluded unless they are explicitly loaded.

5. Weather data was scraped from Timeanddate.com and is not an official Bureau of Meteorology dataset. This should be disclosed clearly when presenting the modelling methodology.

6. The nearest traffic station may not always be a good proxy for charger demand, especially when the matched station is far from the charger. A distance threshold should be applied to reduce weak matches.

---

## Source and Reference Links

| Source | Link | Used For |
|---|---|---|
| Australian Automobile Association - Electric Vehicle Index | [AAA EV Index](https://www.aaa.asn.au/research-data/electric-vehicle/) | Postcode-level vehicle registration and EV adoption indicators |
| Transport for NSW Open Data - EV Charging Locations | [EV Charging Locations Dataset](https://opendata.transport.nsw.gov.au/data/dataset/ev-charging-locations) | AC chargers, DC fast chargers, and upcoming fast chargers in NSW |
| NSW Roads Traffic Volume Counts API - NSW Data | [Traffic Volume Counts API](https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api) | Traffic station reference and traffic volume count metadata |
| NSW Roads Traffic Volume Counts API - Transport Open Data | [Transport Open Data Traffic Volume Counts API](https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api) | Permanent hourly traffic count source description |
| Timeanddate.com - Sydney Historic Weather | [Past Weather in Sydney](https://www.timeanddate.com/weather/australia/sydney/historic) | Historical weather observations scraped for daily weather features |

---

## Professional Data Source Statement

The dataset integrates multiple public and project-generated data sources to support EV charger demand and infrastructure analysis in NSW. Vehicle registration indicators are based on the Australian Automobile Association EV Index, which reports light vehicle registrations by fuel type and postcode. EV charging infrastructure data is sourced from Transport for NSW through the Transport Open Data portal, including AC destination chargers, DC fast chargers and planned charging sites. Traffic station metadata and hourly traffic count records are sourced from the NSW Roads Traffic Volume Counts API. Historical weather features were collected from Timeanddate.com for Sydney and transformed into daily average temperature, humidity and wind-speed features. Because the datasets differ in temporal and spatial granularity, preprocessing steps such as fuel-type standardisation, spatial nearest-station matching, weather aggregation and date-based merging are required before modelling.

