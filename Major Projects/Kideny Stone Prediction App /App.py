#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from streamlit.web import cli as stcli
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


df = pd.read_csv('data.csv')
df1 = df.copy()
df1['osmo_cond_ratio'] = df['osmo'] / df['cond']

# Create a new feature by subtracting calc from urea
df1['urea_calc_diff'] = df['urea'] - df['calc']

# Standardize the values of gravity, ph, osmo, cond, urea, and calc
df1[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']] = (df1[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']] - df1[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']].mean()) / df1[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']].std()

# Create a new feature by multiplying osmo by urea
df1['osmo_urea_interaction'] = df['osmo'] * df['urea']

# Categorize the values of gravity, ph, osmo, cond, urea, and calc into 5 bins each
df1['gravity_bin'] = pd.qcut(df['gravity'], 5, labels=False)
df1['ph_bin'] = pd.qcut(df['ph'], 5, labels=False)
df1['osmo_bin'] = pd.qcut(df['osmo'], 5, labels=False)
df1['cond_bin'] = pd.qcut(df['cond'], 5, labels=False)
df1['urea_bin'] = pd.qcut(df['urea'], 5, labels=False)
df1['calc_bin'] = pd.qcut(df['calc'], 5, labels=False)

# Load the trained model
model = load_model('my_model.h5')

# Define a function to make a prediction
def predict_target(gravity, ph, osmo, cond, urea, calc, osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction, gravity_bin, ph_bin, osmo_bin, cond_bin, urea_bin, calc_bin):
    
    # Create a numpy array with the input values and reshape it for the model
    input_data = np.array([[gravity, ph, osmo, cond, urea, calc, osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction, gravity_bin, ph_bin, osmo_bin, cond_bin, urea_bin, calc_bin]])
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
    # Make a prediction using the model
    prediction = model.predict(input_data)
    # Return the predicted target value
    return prediction[0][0]

#plotting
fig1 = go.Figure(data=[go.Scatter3d(
        x=df['gravity'],
        y=df['ph'],
        z=df['osmo'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['target'],
            colorscale='Viridis',
            opacity=0.8
            )
    )])

fig1.update_layout(
    scene=dict(
        xaxis_title='Gravity',
        yaxis_title='pH',
        zaxis_title='Osmo'
    )
)

from pandas.plotting import scatter_matrix
scatter_matrix(df, diagonal='hist', color='b', alpha=0.3, figsize=(15,6))
fig = go.Figure(data=px.scatter_matrix(df, color='target', dimensions=['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']))
fig.update_layout(width=800, height=800)

fig2 = px.imshow(df.corr(), color_continuous_scale='RdBu')
fig2.update_layout(
    xaxis_title='Features',
    yaxis_title='Features',
)
       
 # Add a title
st.title("Kidney Stone Predictor")

with st.sidebar:
  st.divider()
  st.header("OUR MOTIVE")
  st.caption("The aim of this project is to develop a :red[machine learning and deep learning model] that can predict the presence of kidney stones in patients based on their medical history, lab reports, and other relevant factors and can be used to skip the costly imaging tests . This will allow patients and healthcare providers to easily access the model and get quick and accurate predictions about the presence of kidney stones in patients so that they can prevent it from happening.")
  st.header("OUR GOAL")
  st.caption(":red[The ultimate goal of this project is to improve the speed and accuracy of kidney stone diagnosis], leading to better patient outcomes and reduced healthcare costs also giving the users the basic medical knowledge of how Kidney stones occur, terminologies used for predicting kidney stones and how to prevent it if one has etc so that users can better self evaluate through our app. ")
  st.divider()

tab1, tab2, tab3, tab4, tab5 ,tab6= st.tabs(["Introduction","Terminology", "Diagnosis Of Kideny Stones", "Data Visiualisation", "Kidney Stone Predictor", "Preventive Steps"])
with tab1:
  st.subheader("About")
  st.caption("Kidney stones are a common health problem that affects millions of people worldwide. Early detection and timely intervention can help prevent complications and improve treatment outcomes. However, traditional methods for diagnosing kidney stones such as imaging test like CT scans and MRIs can be invasive, time-consuming and costly.")
  st.caption("Kidney stone disease, also known as nephrolithiasis or urolithiasis, is a crystallopathy where a solid piece of material (kidney stone) develops in the urinary tract. Kidney stones typically form in the kidney and leave the body in the urine stream. A small stone may pass without causing symptoms. If a stone grows to more than 5 millimeters (0.2 inches), it can cause blockage of the ureter, resulting in sharp and severe pain in the lower back or abdomen. A stone may also result in blood in the urine, vomiting, or painful urination. About half of people who have had a kidney stone are likely to have another within ten years.")
  st.subheader('Kidney Stone Disease')

  video_file1 = open('myvideo.mp4', 'rb')
  video_bytes1 = video_file1.read()
  st.video(video_bytes1)

  st.subheader("Signs and Symtoms")
  st.caption("The hallmark of a stone that obstructs the ureter or renal pelvis is excruciating, intermittent pain that radiates from the flank to the groin or to the inner thigh. This is due to the transfer of referred pain signals from the lower thoracic splanchnic nerves to the lumbar splanchnic nerves as the stone passes down from the kidney or proximal ureter to the distal ureter. This pain, known as renal colic, is often described as one of the strongest pain sensations known")
  st.subheader("Treatment")
  col1, col2, = st.columns(2)
  with col1:
    st.caption(":blue[Ureteroscopy surgery] has become increasingly popular as flexible and rigid fiberoptic ureteroscopes have become smaller. One ureteroscopic technique involves the placement of a ureteral stent (a small tube extending from the bladder, up the ureter and into the kidney) to provide immediate relief of an obstructed kidney. Stent placement can be useful for saving a kidney at risk for postrenal acute kidney failure due to the increased hydrostatic pressure, swelling and infection (pyelonephritis and pyonephrosis) caused by an obstructing stone. Ureteral stents vary in length from 24 to 30 cm (9.4 to 11.8 in) and most have a shape commonly referred to as a 'double-J' or 'double pigtail', because of the curl at both ends. They are designed to allow urine to flow past an obstruction in the ureter.")
    st.caption("\n")
    video_file2 = open('URS.mp4', 'rb')
    video_bytes2 = video_file2.read()
    st.video(video_bytes2)
    
  with col2:
    st.caption(":blue[Shock wave lithotripsy (SWL)] is a noninvasive technique for the removal of kidney stones. Most ESWL is carried out when the stone is present near the renal pelvis. ESWL involves the use of a lithotriptor machine to deliver externally applied, focused, high-intensity pulses of ultrasonic energy to cause fragmentation of a stone over a period of around 30–60 minutes. Following its introduction in the United States in February 1984, ESWL was rapidly and widely accepted as a treatment alternative for renal and ureteral stones. It is currently used in the treatment of uncomplicated stones located in the kidney and upper ureter, provided the aggregate stone burden (stone size and number) is less than 20 mm (0.8 in) and the anatomy of the involved kidney is normal.")
    st.caption("\n")
    video_file2 = open('ESWL.mp4', 'rb')
    video_bytes2 = video_file2.read()
    st.video(video_bytes2)

with tab2:
   
    st.subheader('Urine Specific Gravity')
    st.caption("‌A urine concentration test provides the specific gravity of your urine. This :blue[measures your kidney's ability to balance water content and excrete waste]. It is important in diagnosing some health conditions that impact water content in your urine. Urine concentration test is also called a blue:[water loading test or a water deprivation test]. The specific gravity of urine refers to the electrolytes and urine osmolality. Depending on your doctor’s concerns, they give you specific instructions for eating, drinking, and taking medication prior to the test. A urine specific gravity test is used to test for diagnosing many health conditions, primarily central diabetes insipidus and nephrogenic diabetes insipidus. :blue[Both health conditions cause your body to signal excessive thirst, resulting in more urination]. However, the cause of each condition is different. Damage to the pituitary gland or hypothalamus causes primary central diabetes. A malformation of your kidneys contributes to nephrogenic diabetes insipidus. Your urine’s specific gravity isn’t explicitly bad for your health. The results do signal other health conditions that may harm your health. The normal specific gravity ranges from person to person. :blue[Your urine specific gravity is generally considered normal in the ranges of 1.005 to 1.030]. If you drink a lot of water, 1.001 may be normal. If you avoid drinking fluids, levels higher than 1.030 may be normal. Your doctor takes your specific symptoms, eating habits, and drinking habits into consideration when assessing your results.")
    st.subheader('pH Value')
    st.caption("A urine pH level test analyzes the acidity or alkalinity of a urine sample. :blue[It’s a simple and painless test]. Many diseases, your diet, and the medications you take can affect how acidic or basic (alkaline) your urine is. For instance, results that are either too high or low can indicate the likelihood that your body will form kidney stones. If your urine is at an extreme on either the low or high end of pH levels, you can adjust your diet to reduce the likelihood of painful kidney stones. In short, :blue[your urine pH is an indicator of your overall health and gives a doctor important clues as to what’s going on in your body.] In this article, we’ll go over what a normal urine pH level looks like, as well as when you need to test it, and look at the test itself.:blue[A neutral pH is 7.0]. The average urine sample tests at about 6.0, :blue[but typical urine pH may range from 4.5–8.0]. The higher the number, the more basic your urine is. The lower the number, the more acidic your urine is.")
    st.subheader('Osmoregulation')
    st.caption("Osmoregulation is the homeostatic control of the water potential of the blood. The kidneys are involved in filtering the blood and deciding which substances to reabsorb and which to excrete as waste. Tiny tubular structures known as tubules carry out this filtration. :blue[There are five main parts to the kidney tubules: the Bowman's capsule, the proximal convoluted tubule, the loop of Henle, the distal convoluted tubule and the collecting duct.] The Bowman's capsule is used for ultrafiltration as the fluid is forced out of the blood and into the tubule. Then, the proximal convoluted tubule is used for the selective reabsorption of all glucose and some salts and water. The loop of Henle then maintains a sodium gradient, before the distal convoluted tubule makes final adjustments to the amount of water and salts that are reabsorbed. The collecting duct is the final piece of the puzzle that allows water to move out by osmosis to decrease the water content of urine.")
    st.subheader('Conductivity')
    st.caption("The urine includes urea, uric acid, protein, glucose, sodium, potassium, chlorine, inorganic phosphorus and calcium. Among the above components, uric acid, sodium, potassium, chlorine, inorganic phosphorus and calcium are electrolytes that exist in form of ions in the urine, i.e., :blue[the so-called ions of the invention]. When the first electrode is electrically connected with the second electrode, positive ions in the electrical path migrate to the cathode and negative ions in the electrical path migrate to the anode to respectively generate oxidation reduction reactions. :blue[The conductivity of the urine is directly proportional to different kidney functions.] More specifically, when the kidney function is normal, a filtration rate of glomeruli in the kidneys is higher, such that the quantity of these electrolytes being filtered to enter the urine is larger. Thus, the urine has a higher conductivity. Conversely, when the kidney functions is abnormal, the filtration rate of the glomeruli in the kidneys is lower, such that the quantity of these electrolytes in the urine is smaller, thus causes the urine having a lower conductivity. Therefore, by :blue[correlating the conductivity of the urine measured by the processing unit with conductivity data associated with different kidney functions, a kidney function status associated with the urine can be determined]. Conductivity could be used as a parameter in routine urinalysis.")
    st.subheader('Urea Levels')
    st.caption("If kidney problems are the main concern, the creatinine levels in your blood will likely also be measured when your blood is tested for urea nitrogen levels. :blue[Creatinine is another waste product that healthy kidneys filter out of your body through urine.] High levels of creatinine in your blood may be a sign of kidney damage. Your doctor may also test how well your kidneys are removing waste from the blood. To do this, you may have a blood sample taken to calculate your estimated glomerular filtration rate (GFR)")
    st.subheader('Calcium Level')
    st.caption("Our bodies can't make vitamin D. We can only get vitamin D from food and by exposing our skin to sunlight. Healthy kidneys can take that vitamin D we absorb and change it to an active form. That active vitamin D then helps us absorb calcium. But in chronic kidney disease (CKD), the kidneys are less able to make active vitamin D. Without enough active vitamin D, you absorb less calcium from the food you eat, so it then becomes low in your blood. Also, extra phosphorus in the blood of people with CKD may bind to calcium in the blood. This can then lower serum calcium. :blue[A normal serum calcium level is 8.5 - 10.2 mg/dL.] A serum calcium that is either too low or too high can be dangerous and both conditions need treatment. But patients with low serum calcium, even levels at the lower end of normal, have been found to reach kidney failure faster than people with higher serum calcium levels.")


with tab3:
  st.header('How do health care professionals diagnose kidney stones?')
  st.caption('Health care professionals use your medical history, a physical exam, and lab and imaging tests to diagnose kidney stones. A health care professional will ask if you have a history of health conditions that make you more likely to develop kidney stones. The health care professional also may ask if you have a family history of kidney stones and about what you typically eat. During a physical exam, the health care professional usually examines your body. The health care professional will ask you about your symptoms.')
  st.divider()
  st.subheader('Lab tests')
  st.caption('Urine tests can show whether your urine contains high levels of minerals that form kidney stones. Urine and blood tests can also help a health care professional find out what type of kidney stones you have.')
  st.caption(":red[Urinalysis:] Urinalysis involves a health care professional testing your urine sample. You will collect a urine sample at a doctor’s office or at a lab, and a health care professional will test the sample. Urinalysis can show whether your urine has blood in it and minerals that can form kidney stones. White blood cells and bacteria in the urine mean you may have a urinary tract infection.")
  st.caption(":red[Blood tests:] A health care professional may take a blood sample from you and send the sample to a lab to test. The blood test can show if you have high levels of certain minerals in your blood that can lead to kidney stones.")
  st.subheader('Imaging tests')
  st.caption("Health care professionals use imaging tests to find kidney stones. The tests may also show problems that caused a kidney stone to form, such as a blockage in the urinary tract or a birth defect. You do not need anesthesia for these imaging tests.")
  st.caption(":red[Abdominal x-ray:] An abdominal x-ray is a picture of the abdomen that uses low levels of radiation NIH external link and is recorded on film or on a computer. An x-ray technician takes an abdominal x-ray at a hospital or outpatient center, and a radiologist reads the images. During an abdominal x-ray, you will lie on a table or stand up. The x-ray technician will position the x-ray machine over or in front of your abdomen and ask you to hold your breath so the picture won’t be blurry. The x-ray technician then may ask you to change position for additional pictures. Abdominal x-rays can show the location of kidney stones in the urinary tract. Not all stones are visible on abdominal x-ray.")
  st.caption(":red[Computed tomography (CT) scans:] CT scans use a combination of x-rays and computer technology to create images of your urinary tract. Although a CT scan without contrast medium is most commonly used to view your urinary tract, a health care professional may give you an injection of contrast medium. Contrast medium is a dye or other substance that makes structures inside your body easier to see during imaging tests. You’ll lie on a table that slides into a tunnel-shaped device that takes the x-rays. CT scans can show the size and location of a kidney stone, if the stone is blocking the urinary tract, and conditions that may have caused the kidney stone to form.")
  
  video_file4 = open('Diagnosis.mp4', 'rb')
  video_bytes4 = video_file4.read()
  st.video(video_bytes4)

with tab4:
   st.subheader("Dataset")
   st.dataframe(df1.style.highlight_max(axis=0))
   st.subheader("3D Scatter Plot")
   st.plotly_chart(fig1)
   st.subheader('PairPlots')
   st.plotly_chart(fig)
   st.subheader('Correlation Heatmap')
   st.plotly_chart(fig2)


# Add input fields for the features
with tab5:
  gravity = st.number_input("Enter gravity value")
  ph = st.number_input("Enter pH value")
  osmo = st.number_input("Enter osmo value")
  cond = st.number_input("Enter cond value")
  urea = st.number_input("Enter urea value")
  calc = st.number_input("Enter calc value")
  osmo_cond_ratio = st.number_input("Enter osmo_cond_ratio value")
  urea_calc_diff = st.number_input("Enter urea_calc_diff value")
  osmo_urea_interaction = st.number_input("Enter osmo_urea_interaction value")
  gravity_bin = st.number_input("Enter gravity_bin value")
  ph_bin = st.number_input("Enter ph_bin value")
  osmo_bin = st.number_input("Enter osmo_bin value")
  cond_bin = st.number_input("Enter cond_bin value")
  urea_bin = st.number_input("Enter urea_bin value")
  calc_bin = st.number_input("Enter calc_bin value")
   
# Add a button to make the prediction
  if st.button("Predict"):
      prediction = predict_target(gravity, ph, osmo, cond, urea, calc, osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction, gravity_bin, ph_bin, osmo_bin, cond_bin, urea_bin, calc_bin)
      st.write("Predicted Target Value: ", prediction)
      if prediction > 0.3:
          st.write("You have kidney stone.")
      else:
          st.write("You dont have kidney stone.")

with tab6:
  
  video_file5 = open('Prevention.mp4', 'rb')
  video_bytes5 = video_file5.read()
  st.video(video_bytes5)
 
  st.header('Prevention of Kidney Stones')
  st.caption("The best way to prevent kidney stones is to make sure you drink plenty of water each day to avoid becoming dehydrated. To prevent stones returning, you should aim to drink up to 3 litres (5.2 pints) of fluid throughout the day, every day.")
  st.caption("You're advised to:\n1) Drink water, but drinks like tea and coffee also count.\n2) Add fresh lemon juice to your water.\n3) Avoid fizzy drinks.\n4) Do not eat too much salt")
  st.caption("Keeping your urine clear helps to stop waste products getting too concentrated and forming stones. You can tell how diluted your urine is by looking at its colour. The darker your urine is, the more concentrated it is. Your urine is usually a dark yellow colour in the morning because it contains a build-up of waste products that your body's produced overnight. Drinks like tea, coffee and fruit juice can count towards your fluid intake, but water is the healthiest option and is best for preventing kidney stones developing. You should also make sure you drink more when it's hot or when you're exercising to replace fluids lost through sweating.Depending on the type of stones you have, your doctor may advise you to cut down on certain types of food. But do not make any changes to your diet without speaking to your doctor first.")


# In[ ]:




