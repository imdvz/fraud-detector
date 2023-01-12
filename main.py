import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

pickle_in = open("lgbm_model.pkl","rb")
classifier = pickle.load(pickle_in)


def welcome():
    return "Welcome All"

def fraud_detector(category, amt, gender, dob, transaction_hour):
    temp1 = [category, amt, gender, dob, transaction_hour]
    temp2 = b = np.array(temp1, dtype=float)
    prediction = classifier.predict([temp2])
    print(prediction)
    return prediction



def main():
#     st.set_page_config(layout="wide")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card Fraud Detection ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Detect user's location
    loc_button = Button(label = "Get Location")
    loc_button.js_on_event("button_click", CustomJS(code="""
        navigator.geolocation.getCurrentPosition(
            (loc) => {
                document.dispatchEvent(new CustomEvent("Your Current Location", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
            }
        )
        """))
    result1 = streamlit_bokeh_events(
        loc_button,
        events="Your Current Location",
        key="get_location",
        refresh_on_update=False,
        override_height=40,
        debounce_time=0)
    
    # Show user's current location when he/she clicked on the "Get Location" button
    if st.button("Show Location"):
        st.success(result1)
        
    if st.button("About"):
        st.text("This is an atempt to build a very robust and accurate Fraud Detector.")
        st.text("LightGBM Classifier is the final model which is performing best.")
    
    na = st.text_input("Full Name","Type Here")
    
    # Text input for asking the age of the user
    dob = st.number_input('What is your age [in Years] ?', min_value=1, max_value=120, value=18)
    
    # Dropdown for asking user's gender
    gender = st.selectbox('Gender', ('Female', 'Male'))
    
    # Dropdown for asking the category
    category = st.selectbox( 'What was the transaction for ?', ('grocery_pos', 'gas_transport', 'health_fitness', 'misc_net', 'shopping_pos', 'shopping_net', 'home', 'food_dining', 'personal_care', 'misc_pos', 'kids_pets', 'entertainment', 'grocery_net', 'travel'))
    
    # Dropdown for asking at what hour of the day the transaction was made
    transaction_hour = st.selectbox('At what hour of the day the transaction was made ?', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23))
    
    # Text input for amount of transaction
    amt = st.text_input("Amount [in USD]","Type Here")
    
    result=""
    
    # For category attribute
    category_label = [ 8,  7,  6,  4,  0,  9, 11,  5, 10,  2, 13, 12,  3,  1]
    category_names = ['grocery_pos', 'gas_transport', 'health_fitness', 'misc_net', 'shopping_pos', 'shopping_net', 'home', 'food_dining', 'personal_care', 'misc_pos', 'kids_pets', 'entertainment', 'grocery_net', 'travel']
    iter_category=0
    for cat in category_names:
        if cat==category:
            category_num = category_label[iter_category]
        iter_category+=1
    
    # For Gender
    if gender=='Female':
        gender_num=0
    else:
        gender_num=1
        
    if st.button("Predict"):
        # Predict about the transaction
        result = fraud_detector(category_num, amt, gender_num, dob, transaction_hour)
    if result==1:
        st.success('Hi '+ na.split(" ")[0] + ', Bad News! The transaction is fraudulent.')
    elif result==0:
        st.success('Hi '+ na.split(" ")[0] + ', Good News! The transaction is not fraudulent.')
    
if __name__=='__main__':
    main()
