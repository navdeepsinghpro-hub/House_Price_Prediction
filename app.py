import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>

/* Main Background */
.stApp{
    background:black;
}

/* Main Container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Hero Card */
.hero{
    background:linear-gradient(135deg,#2563EB,#06B6D4);
    color:white;
    padding:50px;
    border-radius:20px;
    text-align:center;
    margin-bottom:35px;
    box-shadow:0 12px 30px rgba(0,0,0,.18);
}

.hero h1{
    font-size:clamp(32px,5vw,52px);
    font-weight:700;
    margin-bottom:10px;
}

.hero p{
    font-size:clamp(16px,2vw,20px);
    opacity:.95;
}

/* Input Boxes */
[data-testid="stNumberInput"],
[data-testid="stSelectbox"]{
    background:white;
    border-radius:15px;
    padding:10px;
    box-shadow:0 5px 15px rgba(0,0,0,.08);
}

/* Predict Button */
.stButton > button{
    width:100%;
    height:60px;
    border:none;
    border-radius:15px;
    background:linear-gradient(90deg,#2563EB,#1D4ED8);

    color:#FFFFFF !important;
    font-size:22px;
    font-weight:700;

    text-shadow:0 1px 3px rgba(0,0,0,.35);
    transition:0.3s;
    cursor:pointer;
}

.stButton > button:hover{
    background:linear-gradient(90deg,#1D4ED8,#1E40AF);
    transform:translateY(-2px);
    box-shadow:0 8px 20px rgba(37,99,235,.45);
}

.stButton > button *{
    color:#FFFFFF !important;
    fill:#FFFFFF !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#1E3A8A;
}

/* Metric Cards */
[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:18px;
    box-shadow:0 4px 12px rgba(0,0,0,.10);
}

/* Metric Label */
[data-testid="stMetricLabel"]{
    color:#2563EB !important;
    font-weight:600 !important;
    font-size:16px !important;
}

/* Metric Value */
[data-testid="stMetricValue"]{
    color:#111827 !important;
    font-weight:700 !important;
    font-size:30px !important;
}

/* Metric Delta (if any) */
[data-testid="stMetricDelta"]{
    color:#16A34A !important;
}

/* ============================= */
/* Blue Labels */
/* ============================= */

[data-testid="stWidgetLabel"] p{
    color:#2563EB !important;
    font-size:18px !important;
    font-weight:700 !important;
}

/* Number Input Text */
.stNumberInput input{
    color:#2563EB !important;
    font-weight:700 !important;
    font-size:18px !important;
}

/* Selectbox Text */
[data-baseweb="select"] span{
    color:#2563EB !important;
    font-weight:700 !important;
    font-size:18px !important;
}

/* Selectbox Arrow */
[data-baseweb="select"] svg{
    color:#2563EB !important;
}

/* Plus / Minus Buttons */
button[kind="secondary"]{
    color:#2563EB !important;
}

/* Metric Labels */
[data-testid="stMetricLabel"]{
    color:#2563EB !important;
}

/* Mobile Responsive */

@media (max-width: 768px){

    .block-container{
        padding:1rem;
    }

    .hero{
        padding:30px 20px;
        border-radius:15px;
    }

    .hero h1{
        font-size:34px;
    }

    .hero p{
        font-size:16px;
    }

    .stButton>button{
        height:50px;
        font-size:18px;
    }

    [data-testid="stMetric"]{
        padding:12px;
    }

    [data-testid="stMetricValue"]{
        font-size:22px !important;
    }

    [data-testid="stWidgetLabel"] p{
        font-size:16px !important;
    }
}

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

    st.markdown(f"""
        <div style="
        background:white;
        padding:35px;
        border-radius:20px;
        box-shadow:0 10px 25px rgba(0,0,0,.15);
        text-align:center;
        margin-top:20px;
        ">

        <h3 style="color:#2563EB;">
        💰 Estimated House Price
        </h3>

        <h1 style="
        color:#16A34A;
        font-size:55px;
        margin-bottom:5px;
        ">
        {format_price(prediction)}
        </h1>

        <p style="
        font-size:18px;
        color:gray;
        ">
        Exact Price : ₹ {prediction:,.0f}
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.subheader("📈 Expected Price Range")

    low = prediction - 2120000
    high = prediction + 2120000

    c1, c2 = st.columns(2)

    with c1:
        st.metric("💸 Minimum Price", format_price(low))

    with c2:
        st.metric("💰 Maximum Price", format_price(high))
        
    st.subheader("🎯 Prediction Confidence")
    confidence = 57.17
    st.progress(confidence / 100)
    st.write(f"Confidence Score : **{confidence:.2f}%**")

    st.info(
        "😊 This is an estimated house price. The actual market price may vary depending on the property's exact location, condition, amenities, market trends, and other factors."
    )

    st.divider()

    st.subheader("🏡 Property Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.success(f"🏠 **BHK**\n\n{bhk}")
        st.success(f"🏢 **Property Type**\n\n{property_type}")

    with c2:
        st.success(f"📍 **Location**\n\n{location}")
        st.success(f"📐 **Area**\n\n{sqft:,} sqft")

# ---------------------------------
# About Model
# ---------------------------------
st.subheader("⚙️ Model Details")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("🛠️ Algorithm", "XGBoost")
with c2:
    st.metric("📊 Features", "4")
with c3:
    st.metric("🎯 Task", "Regression")

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

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    ❤️ Made with Python • Streamlit • XGBoost
    
    <br>
    Developed by <b>Navdeep Singh</b>
    <br><br>
    <b>Version 2.0</b>
    </div>
    """,
    unsafe_allow_html=True
)