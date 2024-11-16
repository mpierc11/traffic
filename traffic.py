# 



# Import libraries
import streamlit as st
import pandas as pd
import numpy as np   
import pickle
import warnings
warnings.filterwarnings('ignore')


# Set up the app title and image
st.title('Traffic Volume Predictor')
st.image('traffic_image.gif', use_column_width = True, 
         caption = "Utilize our advances Machine Learning application to predict traffic volume.")

# Reading the pickle file that we created before 
model_pickle = open('traffic_volume.pickle', 'rb') 
bst_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset - for dummy encoding (properly)
default_df = pd.read_csv('Traffic_Volume.csv')

sample_df = pd.read_csv('traffic_data_user.csv') # for sample same frame to upload

st.sidebar.image('traffic_sidebar.jpg', use_column_width=True)
st.sidebar.header("Traffic Volume Predictor")
st.write("You can either upload your own data file or manually enter input features")

with st.sidebar.expander("Option 1: Upload your own CSV"):
    user_csv = st.file_uploader("Upload your CSV file here")
    st.header('Sample Data Format for Upload')
    st.dataframe(sample_df.head(5))
        
with st.sidebar.expander("Option 2: User input form"):
    with st.form("inputs_form"):
        holiday = st.selectbox('Choose wheter today is a designated holiday or not', options=['None', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Christmas Day', 'New Years Day', 'Washingtons Birthday', 'Memorial Day', 'Independence Day', 'State Fair', 'Labor Day', 'Martin Luther King Jr Day'])
        temp = st.number_input('Average Temperature in Kelvin', min_value=0.0, max_value=350.0, value=0.0, step=1.0)
        rain_1h = st.number_input('Amount of mm of rain that occured in an hour', min_value=0.0, max_value=1000.0, value=0.0, step=10.0)
        snow_1h = st.number_input('Amount of mm of snow that occured in an hour', min_value=0.0, max_value=10.0, value=0.0, step=0.01)   
        clouds_all = st.number_input('Percentage of cloud cover', min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        weather_main = st.selectbox('Choose the current weather', options=['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog','Thunderstorm', 'Snow', 'Squall', 'Smoke'])
        month = st.selectbox('Choose month', options= ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', ])
        weekday = st.selectbox('Choose day of week', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        hour = st.selectbox('Choose hour', options = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'])
        submit_button = st.form_submit_button("Predict")


# Encode the inputs for model prediction
encode_df = default_df.copy()
encode_df = encode_df.drop(columns=['traffic_volume'])

# Adjust encode_df date_time column
encode_df['date_time'] = pd.to_datetime(encode_df['date_time'], errors='coerce')  # Convert to datetime, handling any errors
encode_df['month'] = encode_df['date_time'].dt.month_name()
encode_df['year'] = encode_df['date_time'].dt.year
encode_df['weekday'] = encode_df['date_time'].dt.day_name()
encode_df['hour'] = encode_df['date_time'].dt.hour

# Ensure year, month, day of the week, and hour within specified ranges
encode_df['weekday'] = encode_df['weekday'].apply(lambda x: x.capitalize())  # Capitalize day names
encode_df['hour'] = encode_df['hour'].clip(lower=0, upper=23).astype(int)
encode_df['hour'] = encode_df['hour'].astype('object')
encode_df['year'] = encode_df['year'].astype('object')

encode_df['holiday'] = np.where(pd.isna(encode_df['holiday']), "None", encode_df['holiday'])

# Drop the original date_time column as it's no longer needed
encode_df = encode_df.drop(columns=['date_time'])
encode_df = encode_df.drop(columns=['year'])

# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]

# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df)

# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)
user_encoded_df = user_encoded_df.loc[:, ~user_encoded_df.columns.duplicated()]

# Get the prediction with its intervals
alpha = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.50, value=0.10, step=0.01)
prediction, intervals = bst_model.predict(user_encoded_df, alpha = alpha)
pred_value = prediction[0]
lower_limit = intervals[:, 0]
upper_limit = intervals[:, 1][0][0]

# Ensure limits are within [0, 1]
lower_limit = max(0, lower_limit[0][0])

# Show the prediction on the app
st.write("## Predicting Traffic Volume...")

# User input
if user_csv is not None:
    user_df = pd.read_csv(user_csv)
    user_df['holiday'] = np.where(pd.isna(user_df['holiday']), "None", user_df['holiday'])
    encode_df = pd.concat([encode_df, user_df])
    encode_dummy_df = pd.get_dummies(encode_df)
    encode_dummy_df = encode_dummy_df.loc[:, ~encode_dummy_df.columns.duplicated()]
    user_encoded_df = encode_dummy_df.tail(len(user_df))
    st.write("With a", (1-alpha)*100, "% confidence interval:")
    prediction, intervals = bst_model.predict(user_encoded_df, alpha = alpha)
    
    user_df['Predicted Traffic Volume'] = prediction
    user_df['Lower CI Limit'] = intervals[:, 0]
    user_df['Lower CI Limit'] = user_df['Lower CI Limit'].apply(lambda x: max(0, x))
    user_df['Upper CI Limit'] = intervals[:, 1]
    
    st.write(user_df)

else:
    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.2f}")
    st.write("With a", (1-alpha)*100, "% confidence interval:")
    st.write(f"**Confidence Interval**: [{lower_limit:.2f}, {upper_limit:.2f}]")


# Additional tabs for DT model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")


