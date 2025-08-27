import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time

# Page config
st.set_page_config(
    page_title="Patient No-Show Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load data and models
@st.cache_data
def load_data():
    data = pd.read_csv('patients.csv')
    return data

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

try:
    df = load_data()
    model_biased = load_model('best_lightgbm_model.pkl')
    model_unbiased = load_model('best_lgbm_model_unbiased.pkl')
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Dashboard', 'Data Analysis', 'Prediction'])

# Dashboard Page
if page == 'Dashboard':
    st.title('🏥 Patient No-Show Prediction Dashboard')
    
    # Key Metrics
    st.subheader('Key Metrics')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Appointments", len(df))
    with col2:
        no_show_rate = (df['No-show'] == 'Yes').mean() * 100
        st.metric("No-Show Rate", f"{no_show_rate:.1f}%")
    with col3:
        avg_age = df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    # Quick Insights
    st.subheader('Quick Insights')
    st.write("""
    - The dataset contains patient appointment information
    - No-show rate is calculated based on appointment attendance
    - Explore the Data Analysis section for detailed visualizations
    - Use the Prediction section to predict no-show probability for new patients
    """)

# Data Analysis Page
elif page == 'Data Analysis':
    st.title('📊 Data Analysis')
    
    # Age Distribution
    st.subheader('Age Distribution')
    fig = px.histogram(df, x='Age', color='No-show', 
                      title='Age Distribution by No-Show Status',
                      color_discrete_map={'Yes': 'red', 'No': 'green'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Gender Distribution
    st.subheader('Gender Distribution')
    gender_fig = px.pie(df, names='Gender', 
                       title='Gender Distribution',
                       hole=0.5)
    st.plotly_chart(gender_fig, use_container_width=True)
    
    # No-Show by Day of Week
    if 'ScheduledDay' in df.columns and 'AppointmentDay' in df.columns:
        try:
            df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
            df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            day_fig = px.histogram(df, x='DayOfWeek', color='No-show', 
                                 category_orders={"DayOfWeek": day_order},
                                 title='Appointments by Day of Week',
                                 color_discrete_map={'Yes': 'red', 'No': 'green'})
            st.plotly_chart(day_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create day of week visualization: {e}")

# Prediction Page
else:
    st.title('🔮 No-Show Prediction')
    st.write('Predict the probability of a patient not showing up for their appointment.')
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox('Gender', ['Female', 'Male'])
            age = st.number_input('Age', min_value=0, max_value=120, value=30)
            days_difference = st.number_input('Days between Schedule and Appointment', min_value=0, value=1)
            scholarship = st.checkbox('Has Scholarship')
            hipertension = st.checkbox('Has Hypertension')
            
        with col2:
            diabetes = st.checkbox('Has Diabetes')
            alcoholism = st.checkbox('Has Alcoholism')
            handicap = st.checkbox('Has Handicap')
            sms_received = st.checkbox('Received SMS')
            model_choice = st.radio('Model', ['Standard', 'Unbiased'])
        
        submitted = st.form_submit_button('Predict')
        
        if submitted:
            # Get current date for appointment day features
            current_date = pd.Timestamp.now()
            current_hour = current_date.hour
            
            # Prepare base features
            base_features = {
                'Age': age,
                'Scholarship': 1 if scholarship else 0,
                'Hipertension': 1 if hipertension else 0,
                'Diabetes': 1 if diabetes else 0,
                'Alcoholism': 1 if alcoholism else 0,
                'Handcap': 1 if handicap else 0,
                'SMS_received': 1 if sms_received else 0,
                'DaysDifference': days_difference,
                'WaitingTime': days_difference,  # Same as DaysDifference for the unbiased model
                'AppointmentDayOfWeek': current_date.dayofweek,  # 0=Monday, 6=Sunday
                'AppointmentMonth': current_date.month,  # 1-12
                'ScheduledHour': current_hour,  # Hour of the day
                'Gender_M': 1 if gender == 'Male' else 0  # One-hot encoded gender
            }
            
            # Select features based on model type
            if model_choice == 'Unbiased':
                expected_features = [
                    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism',
                    'Handcap', 'SMS_received', 'WaitingTime', 'AppointmentDayOfWeek', 'ScheduledHour'
                ]
            else:  # Standard model
                expected_features = [
                    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism',
                    'Handcap', 'SMS_received', 'DaysDifference', 'AppointmentDayOfWeek',
                    'AppointmentMonth', 'Gender_M'
                ]
            
            # Create input DataFrame with only the expected features
            input_df = pd.DataFrame({k: [base_features[k]] for k in expected_features})
            
            # Make prediction
            model = model_unbiased if model_choice == 'Unbiased' else model_biased
            try:
                prob_no_show = model.predict_proba(input_df)[0][1]
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error(f"Input features: {input_df.columns.tolist()}")
                st.error(f"Input shape: {input_df.shape}")
                if hasattr(model, 'feature_name_'):
                    st.error(f"Model expected features: {model.feature_name_}")
                # Set a default value for prob_no_show to continue execution
                prob_no_show = 0.5  # Neutral probability
            
            # Display result
            st.subheader('Prediction Result')
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability of No-Show", f"{prob_no_show*100:.1f}%")
            with col2:
                if prob_no_show > 0.5:
                    st.error('High Risk of No-Show')
                else:
                    st.success('Low Risk of No-Show')
            
            # Show feature importance if available
            st.subheader('Feature Importance')
            try:
                if hasattr(model, 'feature_importances_'):
                    # Get feature names - use model's feature names if available, otherwise use expected_features
                    feature_names = getattr(model, 'feature_name_', expected_features)
                    
                    # Create a DataFrame with feature importances
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Create and display the bar chart
                    fig = px.bar(importance_df, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title=f'Feature Importance ({model_choice} Model)',
                                labels={'Importance': 'Importance Score'})
                    
                    # Improve layout
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_title='Importance Score',
                        yaxis_title='Feature'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show raw importance values in a table
                    with st.expander("View Detailed Importance Scores"):
                        st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}),
                                   use_container_width=True)
                else:
                    st.info("Feature importance is not available for this model.")
            except Exception as e:
                st.warning(f"Could not display feature importance: {e}")
                if hasattr(model, 'feature_importances_'):
                    st.warning(f"Feature importances: {model.feature_importances_}")
                    st.warning(f"Expected {len(expected_features)} features, got {len(model.feature_importances_)}")

# Add some styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True) 