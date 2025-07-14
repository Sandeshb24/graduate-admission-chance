import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression # For calibration plot

# Set page configuration
st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a better look and feel ---
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f6;
        color: #333 !important; /* Ensure general text is dark */
    }
    .stApp {
        background-color: #f0f2f6;
        color: #333 !important; /* Ensure app container text is dark */
    }

    /* Streamlit specific components for text */
    /* Target h1, h2, h3, and p tags directly within the main app content */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp div {
        color: #333 !important; /* Apply dark color to all main text elements */
    }

    /* Header styling (main title) */
    /* This targets the main title and other top-level Streamlit elements */
    .st-emotion-cache-10qj07f {
        color: #333 !important; /* Ensure main title is dark */
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
        padding-top: 20px;
    }

    /* Subheader styling */
    /* This targets h2 elements specifically */
    .st-emotion-cache-10qj07f + div > h2 {
        color: #34495e !important; /* Ensure subheaders are dark */
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    /* Target the main sidebar container */
    [data-testid="stSidebar"] {
        background-color: #ecf0f1 !important; /* Ensure sidebar background is light */
        color: #333 !important; /* Ensure sidebar text is dark */
    }
    /* Target the sidebar content area */
    [data-testid="stSidebarContent"] {
        background-color: #ecf0f1 !important; /* Ensure sidebar content background is light */
        border-right: 1px solid #bdc3c7;
        padding: 20px;
        border-radius: 10px;
        color: #333 !important; /* Ensure sidebar text is dark */
    }
    /* Specific styling for elements within the sidebar if needed */
    [data-testid="stSidebarContent"] .st-emotion-cache-1c7y2kd,
    [data-testid="stSidebarContent"] .st-emotion-cache-1c7y2kd > div > div > input,
    [data-testid="stSidebarContent"] p,
    [data-testid="stSidebarContent"] h1,
    [data-testid="stSidebarContent"] h2,
    [data-testid="stSidebarContent"] h3 {
        color: #333 !important; /* Apply dark color to all text elements within sidebar */
    }


    /* Input widgets styling */
    .st-emotion-cache-1c7y2kd { /* Targets number input labels */
        color: #2c3e50 !important; /* Ensure input labels are dark */
        font-weight: bold;
    }
    .st-emotion-cache-1c7y2kd > div > div > input { /* Targets number input fields */
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333 !important; /* Ensure input text is dark */
    }
    .st-emotion-cache-1c7y2kd > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 0.2rem rgba(52,152,219,0.25);
    }

    /* Button styling */
    .st-emotion-cache-vk3305.e1g8pov61 { /* Targets the main button */
        background-color: #2ecc71;
        color: white; /* Button text should remain white for contrast */
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: bold;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-vk3305.e1g8pov61:hover {
        background-color: #27ae60;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    /* Metric cards */
    .st-emotion-cache-13ln4gm { /* Targets st.metric parent div */
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    .st-emotion-cache-13ln4gm > div > div > div > div:first-child { /* Metric label */
        color: #7f8c8d !important; /* Ensure metric labels are dark */
        font-size: 0.9em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .st-emotion-cache-13ln4gm > div > div > div > div:last-child { /* Metric value */
        color: #2c3e50 !important; /* Ensure metric values are dark */
        font-size: 1.8em;
        font-weight: bold;
    }

    /* Expander styling */
    .st-emotion-cache-p5m9p9 { /* Targets st.expander header */
        background-color: #ecf0f1;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        color: #34495e !important; /* Ensure expander header is dark */
        border: 1px solid #bdc3c7;
    }
    .st-emotion-cache-p5m9p9 > div { /* Expander content */
        padding: 15px;
        background-color: #ffffff;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        border: 1px solid #e0e0e0;
        border-top: none;
    }

    /* Plot styling */
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        padding: 10px;
        background-color: #ffffff;
    }

    /* Info/Success messages */
    .st-emotion-cache-16txtv4 { /* Targets st.info/success */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333 !important; /* Ensure info message text is dark */
    }

    /* General text */
    p {
        color: #555 !important; /* Ensure paragraph text is dark */
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data and Train Model ---
@st.cache_resource
def load_and_train_model():
    """Loads data, preprocesses it, and trains the Linear Regression model."""
    try:
        # Changed to load from GitHub raw URL for better accessibility
        df = pd.read_csv("https://raw.githubusercontent.com/Sandeshb24/graduate-admission-chance/refs/heads/main/Admission_Predict_Ver1.1.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}. Please ensure the URL is correct or the file is in the same directory if loading locally.")
        st.stop()

    # Drop 'Serial No.' as it's not a feature
    df = df.drop("Serial No.", axis=1)

    # Define features (X) and target (y)
    X = df.drop("Chance of Admit ", axis=1)
    y = df["Chance of Admit "]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # Added random_state for reproducibility

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_preds = model.predict(X_test_scaled)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_preds)
    
    # Calculate residuals
    residuals = y_test - y_preds

    return model, scaler, X_test, y_test, y_preds, mse, mae, rmse, r2, residuals, df

# Load model and data
model, scaler, X_test, y_test, y_preds, mse, mae, rmse, r2, residuals, df_full = load_and_train_model()

# --- App Title and Description ---
st.title("🎓 Graduate Admission Chance Predictor")
st.markdown("""
This interactive application predicts your chances of admission to a graduate program
based on various academic and personal factors.
Enter your details in the sidebar to get a personalized prediction.
""")

st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("Enter Your Academic Profile")
st.sidebar.markdown("Adjust the sliders and input fields to match your profile.")

gre_score = st.sidebar.slider("GRE Score (out of 340)", 290, 340, 320)
toefl_score = st.sidebar.slider("TOEFL Score (out of 120)", 90, 120, 105)
university_rating = st.sidebar.slider("University Rating (1-5)", 1, 5, 3)
sop = st.sidebar.slider("Statement of Purpose (SOP) Strength (1.0-5.0)", 1.0, 5.0, 3.5, 0.5)
lor = st.sidebar.slider("Letter of Recommendation (LOR) Strength (1.0-5.0)", 1.0, 5.0, 3.5, 0.5)
cgpa = st.sidebar.slider("CGPA (out of 10)", 6.0, 10.0, 8.0, 0.1)
research = st.sidebar.radio("Research Experience", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")

# Create a DataFrame for the user's input
user_input_df = pd.DataFrame({
    'GRE Score': [gre_score],
    'TOEFL Score': [toefl_score],
    'University Rating': [university_rating],
    'SOP': [sop],
    'LOR ': [lor], # Note the space in 'LOR ' as per the dataset
    'CGPA': [cgpa],
    'Research': [research]
})

# --- Prediction Section ---
st.header(" Your Predicted Admission Chance:")

# Scale user input
user_input_scaled = scaler.transform(user_input_df)

# Make prediction
predicted_chance = model.predict(user_input_scaled)[0]

# Ensure prediction is within a valid probability range [0, 1]
predicted_chance = np.clip(predicted_chance, 0, 1)

st.info(f"Based on your inputs, your predicted chance of admission is: **{predicted_chance:.2f}**")
st.markdown(f"*(This means a {predicted_chance*100:.0f}% probability of admission)*")

st.markdown("---")

# --- Model Performance Metrics ---
st.header("📊 Model Performance Metrics")
st.markdown("These metrics evaluate how well the trained model performs on unseen data.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    st.markdown("*(Average of the squared differences between predicted and actual values)*")
with col2:
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
    st.markdown("*(Average of the absolute differences between predicted and actual values)*")
with col3:
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
    st.markdown("*(Square root of MSE, in the same units as the target variable)*")
with col4:
    st.metric(label="R-squared (R²)", value=f"{r2:.4f}")
    st.markdown("*(Proportion of variance in the dependent variable predictable from the independent variables)*")

st.markdown("---")

# --- Visualizations ---
st.header("📈 Visualizations for Model Insights")

# 1. Actual vs Predicted Values
st.subheader("Actual vs. Predicted Admission Chances")
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(y_test, y_preds, alpha=0.6, color='#3498db')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel("Actual Chance of Admit")
ax1.set_ylabel("Predicted Chance of Admit")
ax1.set_title("Actual vs. Predicted Values")
ax1.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig1)
st.markdown("""
<p style='color:#555;'>
This scatter plot compares the actual admission chances (from the test data) against the chances predicted by the model.
A perfect model would have all points lying on the red dashed line (where Actual = Predicted).
Points clustered closely around this line indicate good model performance.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# 2. Residuals Distribution
st.subheader("Distribution of Prediction Residuals")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True, ax=ax2, color='#2ecc71', bins=30)
ax2.set_xlabel("Residuals (Actual - Predicted)")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of Residuals")
ax2.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig2)
st.markdown("""
<p style='color:#555;'>
This histogram shows the distribution of the residuals (the differences between actual and predicted values).
Ideally, residuals should be normally distributed around zero, indicating that the model's errors are random and unbiased.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# 3. TOEFL Score vs CGPA (from original notebook)
st.subheader("TOEFL Score vs. CGPA (Original Data)")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.regplot(x=df_full["TOEFL Score"], y=df_full["CGPA"], ax=ax3, scatter_kws={'alpha':0.6, 'color':'#e74c3c'}, line_kws={'color':'#c0392b'})
ax3.set_xlabel("TOEFL Score")
ax3.set_ylabel("CGPA")
ax3.set_title("TOEFL Score vs. CGPA with Regression Line")
ax3.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig3)
st.markdown("""
<p style='color:#555;'>
This plot visualizes the relationship between TOEFL scores and CGPA from the original dataset.
The regression line shows the general trend, suggesting a positive correlation between these two factors.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# 4. Calibration Plot (if possible to implement simply)
# Note: IsotonicRegression is typically used for probability calibration,
# which is more relevant for classification models. For regression,
# it might not directly apply in the same way as a "calibration curve".
# However, the notebook used it, so I'll try to replicate the plot's idea.
st.subheader("Calibration Plot (Predicted vs. Actual Probabilities)")
try:
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_preds, y_test)
    y_calibrated = calibrator.transform(y_preds)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.scatter(y_preds, y_test, alpha=0.5, label='Uncalibrated', color='#9b59b6')
    ax4.scatter(y_calibrated, y_test, alpha=0.5, label='Calibrated', color='#3498db')
    ax4.plot([0, 1], [0, 1], linestyle='--', color='red', lw=2)
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Actual Probability")
    ax4.set_title("Calibration Plot")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig4)
    st.markdown("""
    <p style='color:#555;'>
    This plot shows how well the predicted probabilities align with the actual probabilities.
    The red dashed line represents perfect calibration. Points closer to this line indicate better calibration.
    The 'Calibrated' points (if visible and different from 'Uncalibrated') show the effect of isotonic regression
    in adjusting predictions to be closer to actual outcomes.
    </p>
    """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not generate calibration plot: {e}")
    st.markdown("""
    <p style='color:#555;'>
    The calibration plot attempts to show if the predicted probabilities are well-calibrated (i.e., if a prediction of 0.8 truly means an 80% chance).
    For regression tasks, this is often interpreted as how well the predicted values align with the actual values across different ranges.
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("🚀 How to Use This App")
st.markdown("""
1.  **Adjust Inputs:** Use the sliders and radio button in the left sidebar to input your academic details.
2.  **View Prediction:** Your predicted chance of admission will update automatically.
3.  **Explore Metrics & Plots:** Review the model's performance metrics and various visualizations to understand its accuracy and behavior.
""")

st.markdown("---")
st.markdown("Developed by Sandesh using Data from 'Graduate Admission' dataset refered from UCLA Admissions datasets." )
