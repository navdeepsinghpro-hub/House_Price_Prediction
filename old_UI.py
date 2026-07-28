import streamlit as st
import pandas as pd
import joblib
import base64

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("hero.jpg")

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.markdown(f"""
<style>

/* Background */
.stApp {{
    background-color: black;
}}

/* Hero */
.hero {{
    background:
    linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
    url("data:image/jpeg;base64,{bg}");

    background-size: cover;
    background-position: center;

    padding:70px;
    border-radius:20px;
    color:white;
    text-align:center;

    margin-bottom:30px;

    box-shadow:0px 8px 20px rgba(0,0,0,.2);
}}

.hero h1 {{
    font-size:55px;
    margin-bottom:10px;
}}

.hero p {{
    font-size:22px;
}}

/* Button */

.stButton>button {{

    width:100%;
    height:55px;

    border:none;

    border-radius:12px;

    background:#2563eb;

    color:white;

    font-size:20px;

    font-weight:bold;
}}

.stButton>button:hover {{
    background:#1d4ed8;
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Model
# ---------------------------------
model = joblib.load("house_price_model.pkl")
columns = joblib.load("columns.pkl")

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:

    st.image("hero.jpg")

    st.title("🏠 House Price Predictor")

    st.write(
    """
Welcome 👋

Predict the estimated market value of residential properties using an **XGBoost Machine Learning model**.
    """
    )

    st.divider()

    st.subheader("📊 Model Performance")

    st.metric("R² Score", "57.17%")

    st.metric("MAE", "₹21.2 Lakh")

    st.metric("RMSE", "₹29.9 Lakh")

    st.divider()

    st.subheader("⚙️ Model")

    st.write("""
**Algorithm**

• XGBoost Regressor

**Features**

🏠 BHK

🏢 Property Type

📍 Location

📐 Area (Sqft)

""")

    st.divider()

    st.info(
        "This model predicts the estimated market value of a house. The final market price may vary depending on location, amenities and current market trends."
    )

# ---------------------------------
# Main Title
# ---------------------------------
st.markdown("""
<div class="hero">

<h1>🏠 House Price Prediction</h1>

<p>
Predict house prices instantly using Machine Learning & XGBoost
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Input Section
# ---------------------------------
col1, col2 = st.columns(2)

with col1:

    bhk = st.number_input(
        "🏠 Number of BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    property_type = st.selectbox(
        "🏢 Property Type",
        [
            "Flat",
            "House",
            "Villa"
        ]
    )

with col2:

    sqft = st.number_input(
        "📐 Area (Sqft)",
        min_value=100,
        value=1000,
        step=50
    )

    location = st.selectbox(
        "📍 Location",
        [
            'Ahmedabad','Amritsar','Bangalore','Bhilai','Bhopal',
            'Bhubaneswar','Bilaspur','Chennai','Coimbatore','Cuttack',
            'Dehradun','Dhanbad','Durgapur','Dwarka','Ernakulam',
            'Faridabad','Gurgaon','Guwahati','Gwalior','Haridwar',
            'Howrah','Hyderabad','Indore','Jaipur','Jalandhar',
            'Jamshedpur','Jodhpur','Kanpur','Karimnagar','Kochi',
            'Kolkata','Kozhikode','Lucknow','Ludhiana','Madurai',
            'Mangalore','Mumbai','Muzaffarpur','Mysore','Nagpur',
            'NewDelhi','Noida','Panipat','Patna','Pondicherry',
            'Pune','Raipur','Ranchi','Rishikesh','Rohini',
            'Shimla','Thane','Trichy','Udaipur','Vadodara',
            'Warangal'
        ]
    )

st.divider()

def format_price(price):
    if price >= 10000000:
        return f"₹ {price/10000000:.2f} Crore"
    elif price >= 100000:
        return f"₹ {price/100000:.2f} Lakh"
    else:
        return f"₹ {price:,.0f}"

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔍 Predict House Price", use_container_width=True):

    data = pd.DataFrame({
        "bhk": [bhk],
        "propertytype": [property_type],
        "location": [location],
        "sqft": [sqft]
    })

    data = pd.get_dummies(data)
    data = data.reindex(columns=columns, fill_value=0)

    prediction = model.predict(data)[0]

    st.success("✅ Prediction Completed!")

    st.metric(
        "💰 Estimated House Price",
        format_price(prediction)
    )

    st.caption(f"Exact Price : ₹ {prediction:,.0f}")

    st.info(
        "😊 This is an estimated house price. The actual market price may vary depending on the property's exact location, condition, amenities, market trends, and other factors."
    )

    st.divider()

    st.subheader("🏡 Property Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🏠 BHK", bhk)
    c2.metric("🏢 Type", property_type)
    c3.metric("📍 Location", location)
    c4.metric("📐 Area", f"{sqft:,} sqft")

# ---------------------------------
# About Model
# ---------------------------------
with st.expander("ℹ️ About This Project"):

    st.write("""
### Model Information

- **Algorithm:** XGBoost Regressor
- **Target:** Total House Price
- **Features Used:**
    - BHK
    - Property Type
    - Location
    - Area (Sqft)

### Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
""")

st.divider()

st.caption("Made with ❤️ by Navdeep Singh")