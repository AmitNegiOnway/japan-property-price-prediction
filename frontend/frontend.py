import streamlit as st
import requests
import pandas as pd

# ============================================================
# Page Configuration
# ============================================================
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    df = pd.read_csv(file_path, low_memory=False)
    return df

df = load_data('data/raw/df.csv')

st.set_page_config(
    page_title="Japan House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# FastAPI Configuration
# ============================================================

API_URL = "http://host.docker.internal:8000/predict_house_price"


# ============================================================
# Page Header
# ============================================================

st.title("🏠 Japan House Price Prediction")

st.write(
    "Enter the property information below to estimate the transaction price."
)


# ============================================================
# Property Information
# ============================================================

@st.cache_data
def get_districts_by_region():
    return (
        df.dropna(subset=["DistrictName"])
          .groupby("Region")["DistrictName"]
          .apply(lambda s: sorted(s.unique().tolist()))
          .to_dict()
    )



type_options = df['Type'].value_counts().index.to_list()

region_options = df['Region'].value_counts().index.to_list()

st.header("📍 Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    selected_type = st.selectbox(
        "Property Type",
        options=type_options,
        index=None,
        placeholder="Select property type"
    )

    selected_region = st.selectbox(
        "Region",
        options=region_options,
        index=None,
        placeholder="Select region"
    )

    if selected_region:
        district_list = get_districts_by_region().get(selected_region, [])
    else:
        district_list = []

    selected_district = st.selectbox(
        "District Name",
        options=district_list,
        index=None,
        placeholder="Select region first" if not selected_region else "Start typing..."
    )



nearest_station_option = df['NearestStation'].value_counts().index.to_list()

land_shape_options = {
    "Rectangular":       "Rectangular Shaped",
    "Semi-rectangular":  "Semi-rectangular Shaped",
    "Irregular":         "Irregular Shaped",
    "Semi-trapezoidal":  "Semi-trapezoidal Shaped",
    "Semi-square":       "Semi-square Shaped",
    "Trapezoidal":       "Trapezoidal Shaped",
    "Semi-shaped":       "Semi-shaped",
    "Square":            "Square Shaped",
    "Flag-shaped":       "Flag-shaped etc.",
}

structure_options = {
    "Wooden":                                     "W",
    "Steel":                                      "S",
    "Light-gauge Steel":                          "LS",
    "Reinforced Concrete":                        "RC",
    "Steel-Reinforced Concrete":                  "SRC",
    "Steel + Wooden":                             "S, W",
    "Block / Brick":                              "B",
    "Wooden + Block / Brick":                     "W, B",
    "Reinforced Concrete + Wooden":               "RC, W",
    "Wooden + Light-gauge Steel":                 "W, LS",
    "Steel + Block / Brick":                      "S, B",
    "Reinforced Concrete + Block / Brick":        "RC, B",
    "Steel + Light-gauge Steel":                  "S, LS",
    "Steel + Wooden + Block / Brick":             "S, W, B",
    "Steel + Wooden + Light-gauge Steel":         "S, W, LS",
    "Reinforced Concrete + Wooden + Light-gauge Steel": "RC, W, LS",
    "Steel + Block / Brick + Light-gauge Steel":  "S, B, LS",
    "Steel-Reinforced Concrete + Reinforced Concrete":  "SRC, RC",
    "Steel-Reinforced Concrete + Steel":          "SRC, S",
    "Steel-Reinforced Concrete + Wooden + Block / Brick": "SRC, W, B",
}


with col2:

    nearest_station = st.selectbox(
        "Nearest Station",
        options=nearest_station_option,
        index=None,
        placeholder="Start typing..."
    )

    land_shape = st.selectbox(
        "Land Shape",
        options=land_shape_options.keys(),
        index=None,
        placeholder="e.g. Rectangular"
    )

    with st.expander("ℹ️ What do these terms mean?"):
        st.markdown("""
            ### 🗺️ Land Shape

            - **Rectangular** → ideal shape, four right angles, easy to build on
            - **Square** → equal sides and right angles, ideal for building
            - **Semi-rectangular** → almost rectangular, one side slightly uneven
            - **Semi-square** → almost square, one side slightly uneven
            - **Trapezoidal** → two slanted sides (like a trapezoid), smaller usable area
            - **Semi-trapezoidal** → partly slanted sides, slightly reduced usable area
            - **Irregular** → uneven shape, harder to build on
            - **Semi-shaped** → mixed or only partially defined shape
            - **Flag-shaped** → narrow access strip leading to a wider plot at the back
            """)

    structure = st.selectbox(
        "Structure",
        options=structure_options.keys(),
        index=None,
        placeholder="e.g. Wooden"
    )

    with st.expander("ℹ️ What does Structure mean?"):
        st.markdown("""
            ### 🏗️ Structure

            **Single-material structures:**

            - **W — Wooden** → traditional wooden-frame house (most common for houses)
            - **S — Steel** → steel-frame construction (common for larger buildings)
            - **LS — Light-gauge Steel** → thin steel frame, lightweight construction
            - **RC — Reinforced Concrete** → concrete with steel bars (strong, earthquake-resistant)
            - **SRC — Steel-Reinforced Concrete** → steel frame inside concrete (strongest)
            - **B — Block / Brick** → masonry block or brick construction

            **Combined structures** — e.g. `S, W` = Steel + Wooden.
            """)




use_option = df['Use'].value_counts().index.to_list()
Purpose_option = df['Purpose'].value_counts().index.to_list()
Direction_option = df['Direction'].value_counts().index.to_list()

with col3:

    purpose = st.selectbox(
        "Purpose",
        options=Purpose_option,
        index=None,
        placeholder="e.g. House"
    )

    with st.expander("ℹ️ What does Purpose mean?"):
        st.markdown("""
            **Purpose** = the **main / primary** use of the property. Always a single category.

            - **House** → residential home (the main purpose is living)
            - **Shop** → retail store / commercial shop
            - **Office** → business / administrative office
            - **Factory** → manufacturing plant
            - **Warehouse** → storage / logistics building
            - **Other** → anything else (parking, mixed, unusual)
            """)

    use = st.selectbox(
        "Use",
        options=use_option,
        index=None,
        placeholder="e.g. House, Shop"
    )

    with st.expander("ℹ️ What does Use mean?"):
        st.markdown("""
            **Use** = the **actual current usage** of the property. It can be a **single use** or **several combined**.

            A value like `"House, Shop"` means the property is **partly a home and partly a shop**.
            `"House, Office, Warehouse"` means it is used as **all three at once**.

            **Building blocks (single uses):**
            - **House** → residential home
            - **Housing Complex** → apartment / multi-unit residential building
            - **Shop** → retail store
            - **Office** → business office
            - **Warehouse** → storage building
            - **Factory** → manufacturing plant
            - **Workshop** → small workshop / light manufacturing
            - **Parking Lot** → parking spaces
            - **Other** → anything not listed above

            **Common examples of combined uses:**
            - **House, Shop** → home with a shop on the ground floor (very common in Japan)
            - **House, Office** → home that also serves as an office
            - **House, Parking Lot** → home with an attached parking area
            - **Office, Warehouse** → office with storage space
            - **Factory, Office, Warehouse** → industrial site with all three functions
            - **Housing Complex, Shop** → apartment building with a shop on the ground floor

            **In short:** the more items in the list, the more mixed-use the property is.
            """)

    direction = st.selectbox(
        "Direction",
        options=Direction_option,
        index=None,
        placeholder="e.g. South"
    )

    with st.expander("ℹ️ What does Direction mean?"):
        st.markdown("""
            **Direction** = the direction the **front of the property** faces (toward the road).

            - **North / South / East / West** → property faces that cardinal direction
            - **Northeast / Northwest / Southeast / Southwest** → in-between directions
            - **No facing road** → interior plot, accessed by a private path

            In Japan, **south-facing** is generally the most desirable (more sunlight).
            """)


# ============================================================
# Property Classification
# ============================================================

st.header("🏢 Property Classification")

col1, col2, col3 = st.columns(3)

classification_options = {
    "National Highway — national-level main road":            "National Highway",
    "Prefectural Road — prefecture-level road":               "Prefectural Road",
    "Hokkaido Prefectural Road — Hokkaido prefectural":       "Hokkaido Prefectural Road",
    "Kyoto/Osaka Prefectural Road — Kyoto/Osaka prefectural": "Kyoto/ Osaka Prefectural Road",
    "Tokyo Metropolitan Road — Tokyo metro road":             "Tokyo Metropolitan Road",
    "City Road — city-maintained road":                       "City Road",
    "Ward Road — ward-maintained road (Tokyo)":               "Ward Road",
    "Town Road — town-maintained road":                       "Town Road",
    "Village Road — village-maintained road":                 "Village Road",
    "Agricultural Road — farm access road":                   "Agricultural Road",
    "Forest Road — forestry access road":                     "Forest Road",
    "Access Road — service / driveway access":                "Access Road",
    "Private Road — privately owned road":                    "Private Road",
    "Road — generic / unspecified road":                      "Road",
}

city_planning_options = {
    # Residential zones (low-density → high-density)
    "Category I Exclusively Low-story Residential — low-rise homes only, most restrictive": "Category I Exclusively Low-story Residential Zone",
    "Category II Exclusively Low-story Residential — low-rise homes, small shops allowed":  "Category II Exclusively Low-story Residential Zone",
    "Category I Exclusively Medium-high Residential — mid-rise homes, most restrictive":    "Category I Exclusively Medium-high Residential Zone",
    "Category II Exclusively Medium-high Residential — mid-rise homes, small shops":        "Category II Exclusively Medium-high Residential Zone",
    "Category I Residential — quiet residential area":                                      "Category I Residential Zone",
    "Category II Residential — residential, some shops allowed":                            "Category II Residential Zone",
    "Quasi-residential — mixed homes and small shops":                                      "Quasi-residential Zone",

    # Commercial zones
    "Neighborhood Commercial — small local shops, low traffic":                             "Neighborhood Commercial Zone",
    "Commercial — full commercial / retail area":                                           "Commercial Zone",

    # Industrial zones
    "Quasi-industrial — light industry + some homes":                                       "Quasi-industrial Zone",
    "Industrial — general industrial area":                                                 "Industrial Zone",
    "Exclusively Industrial — heavy industry only, no homes":                               "Exclusively Industrial Zone",

    # Outside / restricted
    "Urbanization Control Area — development restricted":                                   "Urbanization Control Area",
    "Non-divided City Planning Area — inside planning area, no zone":                       "Non-divided City Planning Area",
    "Quasi-city Planning Area — quasi-urban planning area":                                 "Quasi-city Planning Area",
    "Outside City Planning Area — outside planning, rural":                                 "Outside City Planning Area",
}

df['Municipality'] = df['Municipality'].str.replace('County', 'District').str.replace(',', ' , ')
a = df['Municipality'].value_counts().index.to_list()
b = df['MunicipalityCode'].value_counts().index.to_list()

municipality_map = {}
for i, j in zip(a, b):
    municipality_map[i] = j


with col1:

    classification = st.selectbox(
        "Road Type (Classification)",
        options=classification_options.keys(),
        index=None,
        placeholder="Select the road in front of the property"
    )

    with st.expander("ℹ️ What does Road Type mean?"):
        st.markdown("""
            **Road Type (Classification)** = the type of road the property faces.

            - **National Highway / Prefectural Road** → large, busy public roads
            - **City / Ward / Town / Village Road** → locally maintained public roads
            - **Access Road / Private Road** → driveway or private service road
            - **Agricultural / Forest Road** → roads through farmland or forest
            """)

    city_planning = st.selectbox(
        "City Planning",
        options=city_planning_options.keys(),
        index=None,
        placeholder="e.g. Urbanization Control Area"
    )

    with st.expander("ℹ️ What does City Planning mean?"):
        st.markdown("""
            **City Planning** = Japanese zoning category — controls what can be built on the land.

            - **Residential Zones** → low-rise / mid-rise housing (most restrictive)
            - **Quasi-residential** → homes with small shops
            - **Commercial Zones** → shops, offices, restaurants
            - **Industrial Zones** → factories, warehouses
            - **Urbanization Control / Outside City Planning** → rural, restricted development
            """)

    Municipality = st.selectbox(
        "Municipality",
        options=municipality_map.keys(),
        index=None,
        placeholder="e.g. Aomori City"
    )

    with st.expander("ℹ️ What does Municipality mean?"):
        st.markdown("""
            **Municipality** = the city / town / village where the property is located.

            The model uses the municipality **code** internally, but you see the name here.
            Example: `Aomori City` → code `2201`.
            """)


with col2:

    building_year = st.number_input(
        "Building Year (from 1945 - 2020)",
        min_value=1945,
        max_value=2020,
        value=2012,
        step=1
    )

    year = st.number_input(
        "Transaction Year (from 2006 - 2019)",
        min_value=2006,
        max_value=2019,
        value=2015,
        step=1
    )

with col3:

    quarter = st.number_input(
        "Quarter",
        min_value=1,
        max_value=4,
        value=1,
        step=1
    )

    MinTimeToNearestStation = st.number_input(
        "MinTimeToNearestStation in minute",
        min_value=0.0,
        max_value=120.0,
        value=5.0,
        step=1.0,
        help="Walking minutes from the property to the nearest train station."
    )


# ============================================================
# Numerical Features
# ============================================================

st.header("📐 Property Measurements")

col1, col2, col3 = st.columns(3)

with col1:

    total_floor_area = st.number_input(
        "Total Floor Area (m²)",
        min_value=10.0, max_value=2000.0, value=2000.0, step=1.0,
        help="Sum of every floor's area. Example: 3 floors × 40 m² = 120 m²."
    )

    land_area = st.number_input(
        "Land Area (m²)",
        min_value=15.0, max_value=5000.0, value=5000.0, step=1.0,
        help="Total size of the land plot."
    )

with col2:

    building_coverage_ratio = st.number_input(
        "Building Coverage Ratio (%)",
        min_value=30.0, max_value=80.0, value=50.0, step=10.0,
        help="Building Area ÷ Land Area × 100. Japan's zoning limits this (usually 30–80%)."
    )

    Frontage = st.number_input(
        "Frontage (m)",
        min_value=0.5, max_value=50.0, value=50.0, step=0.1,
        help="Width of the land plot along the road it faces."
    )

with col3:

    floor_area_ratio = st.number_input(
        "Floor Area Ratio (%)",
        min_value=50.0, max_value=800.0, value=100.0, step=10.0,
        help="Total Floor Area ÷ Land Area × 100. Tells how many 'layers' of floors are allowed."
    )

    Breadth = st.number_input(
        "Breadth (m)",
        min_value=1.0, max_value=60.0, value=50.0, step=0.1,
        help="Depth of the land plot from the road to the rear boundary."
    )


# ============================================================
# Prediction Button
# ============================================================

st.divider()

if st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validate categorical selections
    # --------------------------------------------------------

    required = {
        "Property Type":    selected_type,
        "Region":           selected_region,
        "District Name":    selected_district,
        "Nearest Station":  nearest_station,
        "Land Shape":       land_shape,
        "Structure":        structure,
        "Use":              use,
        "Purpose":          purpose,
        "Direction":        direction,
        "Classification":   classification,
        "City Planning":    city_planning,
        "Municipality":     Municipality,
    }

    missing = [name for name, value in required.items() if not value]

    if missing:
        st.warning("Please fill in: " + ", ".join(missing))
        st.stop()

    # --------------------------------------------------------
    # Convert UI labels → original dataset values
    # --------------------------------------------------------

    payload_type          = selected_type                          # already dataset value
    payload_region        = selected_region                        # already dataset value
    payload_district      = selected_district                      # already dataset value
    payload_station       = nearest_station                        # already dataset value
    payload_land_shape    = land_shape_options[land_shape]         # friendly → dataset
    payload_structure     = structure_options[structure]           # friendly → dataset
    payload_classification = classification_options[classification] # friendly → dataset
    payload_city_planning = city_planning_options[city_planning]   # friendly → dataset
    payload_municipality  = municipality_map[Municipality]         # name → code

    # --------------------------------------------------------
    # Create API payload
    # --------------------------------------------------------

    payload = {

        "Type":           payload_type,
        "Region":         payload_region,
        "DistrictName":   payload_district,
        "NearestStation": payload_station,

        "LandShape":      payload_land_shape,
        "Structure":      payload_structure,
        "Use":            use,
        "Purpose":        purpose,

        "Direction":      direction,
        "Classification": payload_classification,
        "CityPlanning":   payload_city_planning,
        "MunicipalityCode":   payload_municipality,

        "BuildingYear":   building_year,
        "Year":           year,
        "Quarter":         quarter,
        "MinTimeToNearestStation": MinTimeToNearestStation,

        "TotalFloorArea": total_floor_area,
        "Area":           land_area,
        "Frontage":       Frontage,
        "CoverageRatio": building_coverage_ratio,
        "FloorAreaRatio": floor_area_ratio,
        "Breadth":        Breadth,
    }


    # --------------------------------------------------------
    # Send request to FastAPI
    # --------------------------------------------------------

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )


        # ----------------------------------------------------
        # Successful prediction
        # ----------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed successfully!")

            st.metric(
                label="Estimated House Price",
                value=result["Price"]
            )


        # ----------------------------------------------------
        # API Error
        # ----------------------------------------------------

        else:

            st.error(
                f"Prediction failed: {response.text}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to FastAPI. "
            "Make sure your FastAPI server is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )