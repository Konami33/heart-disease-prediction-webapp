import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression


def home():
    # create streamlit interface, and some info about the app
    st.image("image/home.jpg")

    st.write("""
             ## In just a few seconds, you can calculate your risk of developing heart disease!
             ### To predict your heart disease status:
             ###### 1- Enter the parameters that best describe you.
             ###### 2- Press the "Predict" button and wait for the result.
             """)


    st.image("image/blog2.png ")



def predict():
    st.title('Please, fill your informations to predict your heart condition')

    BMI = st.selectbox("Select your BMI", ("Normal weight BMI  (18.5-25)",
                                                   "Underweight BMI (< 18.5)",
                                                   "Overweight BMI (25-30)",
                                                   "Obese BMI (> 30)"))
    Age = st.selectbox("Select your age",
                               ("18-24",
                                "25-29",
                                "30-34",
                                "35-39",
                                "40-44",
                                "45-49",
                                "50-54",
                                "55-59",
                                "60-64",
                                "65-69",
                                "70-74",
                                "75-79",
                                "55-59",
                                "80 or older"))

    Race = st.selectbox("Select your Race", ("Asian",
                                                     "Black",
                                                     "Hispanic",
                                                     "American Indian/Alaskan Native",
                                                     "White",
                                                     "Other"
                                                     ))

    Sex = st.selectbox("Select your gender", ("Female",
                                                      "Male"))
    Smoking = st.selectbox("Have you smoked more than 100 cigarettes in"
                                   " your entire life ?)",
                                   options=("No", "Yes"))
    alcoholDink = st.selectbox("How many drinks of alcohol do you have in a week?", options=("No", "Yes"))
    stroke = st.selectbox("Did you have a stroke?", options=("No", "Yes"))

    sleepTime = st.number_input("Hours of sleep per 24h", 0, 24, 7)

    genHealth = st.selectbox("General health",
                                     options=("Good", "Excellent", "Fair", "Very good", "Poor"))

    physHealth = st.number_input("Physical health in the past month (Excelent: 0 - Very bad: 30)"
                                         , 0, 30, 0)
    mentHealth = st.number_input("Mental health in the past month (Excelent: 0 - Very bad: 30)"
                                         , 0, 30, 0)
    physAct = st.selectbox("Physical activity in the past month"
                                   , options=("No", "Yes"))

    diffWalk = st.selectbox("Do you have serious difficulty walking"
                                    " or climbing stairs?", options=("No", "Yes"))
    diabetic = st.selectbox("Have you ever had diabetes?",
                                    options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
    asthma = st.selectbox("Do you have asthma?", options=("No", "Yes"))
    kidneyDisease = st.selectbox("Do you have kidney disease?", options=("No", "Yes"))
    skinCancer = st.selectbox("Do you have skin cancer?", options=("No", "Yes"))

    #Model selection
    selectModel = st.selectbox("Select your preferable machine learning model", options=("Logistic Regression (Recommended)", "k-NN", "XGBoost", "Neural Network", "GaussianNB"))
    print(selectModel)

    dataToPredic = pd.DataFrame({
        "BMI": [BMI],
        "Smoking": [Smoking],
        "AlcoholDrinking": [alcoholDink],
        "Stroke": [stroke],
        "PhysicalHealth": [physHealth],
        "MentalHealth": [mentHealth],
        "DiffWalking": [diffWalk],
        "Sex": [Sex],
        "AgeCategory": [Age],
        "Race": [Race],
        "Diabetic": [diabetic],
        "PhysicalActivity": [physAct],
        "GenHealth": [genHealth],
        "SleepTime": [sleepTime],
        "Asthma": [asthma],
        "KidneyDisease": [kidneyDisease],
        "SkinCancer": [skinCancer]
    })

    # Mapping the data as explained in the script above
    dataToPredic.replace("Underweight BMI (< 18.5)", 0, inplace=True)
    dataToPredic.replace("Normal weight BMI  (18.5-25)", 1, inplace=True)
    dataToPredic.replace("Overweight BMI (25-30)", 2, inplace=True)
    dataToPredic.replace("Obese BMI (> 30)", 3, inplace=True)

    dataToPredic.replace("Yes", 1, inplace=True)
    dataToPredic.replace("No", 0, inplace=True)
    dataToPredic.replace("18-24", 0, inplace=True)
    dataToPredic.replace("25-29", 1, inplace=True)
    dataToPredic.replace("30-34", 2, inplace=True)
    dataToPredic.replace("35-39", 3, inplace=True)
    dataToPredic.replace("40-44", 4, inplace=True)
    dataToPredic.replace("45-49", 5, inplace=True)
    dataToPredic.replace("50-54", 6, inplace=True)
    dataToPredic.replace("55-59", 7, inplace=True)
    dataToPredic.replace("60-64", 8, inplace=True)
    dataToPredic.replace("65-69", 9, inplace=True)
    dataToPredic.replace("70-74", 10, inplace=True)
    dataToPredic.replace("75-79", 11, inplace=True)
    dataToPredic.replace("80 or older", 13, inplace=True)

    dataToPredic.replace("No, borderline diabetes", 2, inplace=True)
    dataToPredic.replace("Yes (during pregnancy)", 3, inplace=True)

    dataToPredic.replace("Excellent", 0, inplace=True)
    dataToPredic.replace("Good", 1, inplace=True)
    dataToPredic.replace("Fair", 2, inplace=True)
    dataToPredic.replace("Very good", 3, inplace=True)
    dataToPredic.replace("Poor", 4, inplace=True)

    dataToPredic.replace("White", 0, inplace=True)
    dataToPredic.replace("Other", 1, inplace=True)
    dataToPredic.replace("Black", 2, inplace=True)
    dataToPredic.replace("Hispanic", 3, inplace=True)
    dataToPredic.replace("Asian", 4, inplace=True)
    dataToPredic.replace("American Indian/Alaskan Native", 4, inplace=True)

    dataToPredic.replace("Female", 0, inplace=True)
    dataToPredic.replace("Male", 1, inplace=True)

    # Load the previously saved machine learning model
    filename = 'LogRegModel.pkl'
    filename1 = 'knn.pkl'
    filename2 = 'xgb.pkl'
    filename3 = 'NN.pkl'
    filename4 = 'GNB.pkl'

    logisticmodel = pickle.load(open(filename, 'rb'))
    knnmodel = pickle.load(open(filename1, 'rb'))
    xgbmodel = pickle.load(open(filename2, 'rb'))
    nuralmodel = pickle.load(open(filename3, 'rb'))
    gaussianmodel = pickle.load(open(filename4, 'rb'))

    finalResult = 0.0
    #print(loaded_model)
    if(selectModel == 'Logistic Regression (Recommended)'):
        Result = logisticmodel.predict(dataToPredic)
        ResultProb = logisticmodel.predict_proba(dataToPredic)
        finalResult = round(ResultProb[0][1] * 100, 2)
        print("logical regression", finalResult)

    elif(selectModel == 'k-NN'):
        knnResult = knnmodel.predict(dataToPredic)
        knnResult1 = knnmodel.predict_proba(dataToPredic)
        finalResult = round(knnResult1[0][1] * 100, 2)
        print("knn", finalResult)
    elif(selectModel == 'XGBoost'):
        xgbResult = xgbmodel.predict(dataToPredic)
        xgbResult1 = xgbmodel.predict_proba(dataToPredic)
        finalResult = round(xgbResult1[0][1] * 100, 2)
        print("xgb", finalResult)
    elif(selectModel == 'Neural Network'):
        nnresult = xgbmodel.predict(dataToPredic)
        nnResult1 = xgbmodel.predict_proba(dataToPredic)
        finalResult = round(nnResult1[0][1] * 100, 2)
        print("Neural Network", finalResult)
    else:
        gnbResult = xgbmodel.predict(dataToPredic)
        gnbResult1 = xgbmodel.predict_proba(dataToPredic)
        finalResult = round(gnbResult1[0][1] * 100, 2)
        print("Gaussian", finalResult)

    # Calculate the probability of getting heart disease
    if st.button('PREDICT'):
        # st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
        if (finalResult > 30):
            st.title(f'You have a {finalResult} % chance of getting a heart disease')
        else:
            st.title(f'You have a {finalResult} % chance of getting a heart disease')


def blog():
    with st.expander("Superfoods for a Strong and Healthy Heart"):
        st.image("image/1.jpg")
        st.write("""
            Maintaining a solid and healthy heart is crucial for overall well-being. Alongside regular exercise and a balanced lifestyle, incorporating nutrient-rich superfoods into your diet can provide remarkable benefits for cardiovascular health. In this article, we will explore the concept of superfoods and their positive impact on heart health. We'll delve into the scientific research supporting their benefits and provide links to analyzed studies for further reference.
            Berries: Nature's Antioxidant Powerhouses
            Berries such as blueberries, strawberries, and raspberries are packed with antioxidants, including anthocyanins, which help reduce inflammation and oxidative stress. A study published in the Journal of the Academy of Nutrition and Dietetics highlighted the cardioprotective effects of berries, including improved blood pressure and arterial function.
            
            Fatty Fish: A Rich Source of Omega-3 Fatty Acids
            Fatty fish like salmon, mackerel, and sardines are excellent sources of omega-3 fatty acids, which have been associated with a reduced risk of heart disease. Research published in Circulation emphasized the benefits of omega-3 fatty acids for lowering triglyceride levels, reducing blood pressure, and preventing arrhythmias.
            
            Dark Chocolate: Indulge in Heart-Healthy Flavonoids
            Dark chocolate, with a high cocoa content, contains flavonoids with antioxidant and anti-inflammatory properties. A systematic review and meta-analysis published in Nutrition Reviews found that regular consumption of dark chocolate was associated with improved cardiovascular health, including reduced blood pressure.
            
            Leafy Greens: Nourish Your Heart with Vital Nutrients
            Leafy greens such as spinach, kale, and Swiss chard are rich in vitamins, minerals, and dietary nitrates that support heart health. A study published in the Journal of Nutrition highlighted the beneficial effects of dietary nitrate-rich vegetables on blood pressure and arterial function.
            
            Nuts: Crunch on Heart-Healthy Goodness
            Nuts, including almonds, walnuts, and pistachios, are packed with heart-healthy nutrients such as unsaturated fats, fiber, and antioxidants. A review published in the Journal of the American College of Cardiology concluded that Nut consumption is associated with a lower risk of cardiovascular disease and improved lipid profiles.
            
            Incorporating superfoods into your diet can be a powerful strategy for promoting a strong and healthy heart. The scientific research analyzed in this article highlights the positive impact of superfoods on various aspects of cardiovascular health, including reducing inflammation, improving blood pressure, and supporting overall heart function.
            You can nourish your heart with vital nutrients and protective compounds by embracing a diet rich in berries, fatty fish, dark chocolate, leafy greens, and nuts. Remember to consult with a healthcare professional or nutritionist for personalized guidance and recommendations tailored to your specific needs.
        """)

    with st.expander("All You Should Know About Cardiovascular Disease"):
       st.image("image/2.jpg")
       st.write("""
           Cardiovascular disease (CVD), often synonymous with heart disease, refers to conditions involving the heart or blood vessels. It is a global health concern, being the leading cause of death worldwide. Understanding the nature, risk factors, prevention strategies, and treatment options for CVD can empower us to better care for our heart health.

            Understanding Cardiovascular Disease
            
            CVD encompasses a range of conditions, including coronary artery disease, heart failure, arrhythmias, and heart valve problems. The most common form, coronary artery disease, involves the narrowing or blockage of the coronary arteries, usually due to atherosclerosis - a process characterized by the buildup of fatty plaques in the artery walls.
            
            Risk Factors
            
            CVD risk factors are the following: modifiable and non-modifiable. Non-modifiable risk factors include age, gender, and genetic predisposition. On the other hand, modifiable risk factors are aspects of our lifestyle and medical conditions that we can change or control. These include high blood pressure, high cholesterol, obesity, physical inactivity, diabetes, unhealthy diet, and tobacco use.
            
            Prevention Strategies
            
            Primary prevention strategies aim to prevent the onset of CVD in individuals without clinical evidence of the disease. This involves managing modifiable risk factors through lifestyle changes such as a heart-healthy diet, regular exercise, maintaining a healthy weight, and not smoking.
            
            Diet plays a crucial role in heart health. A diet rich in fruits, vegetables, lean proteins, healthy fats, and whole grains can help lower the risk of CVD. Reducing sodium and added sugar intake, along with moderation in alcohol consumption, is also recommended.
            
            Regular physical activity is another pillar of heart health. The American Heart Association recommends at least 150 minutes per week of moderate-intensity aerobic activity or 75 minutes per week of vigorous aerobic activity, or a combination of both, preferably spread throughout the week.
            
            Treatment Options
            
            Treatment of CVD depends on the specific type of heart disease a person has. It may involve lifestyle changes, medication, or possibly surgery. Medications can treat many forms of heart disease, such as statins for high cholesterol or beta-blockers for high blood pressure. In some cases, surgical procedures such as coronary angioplasty or heart bypass surgery might be required.
            
            Understanding cardiovascular disease is key to prevention and successful treatment. If you have any concerns about your heart health, it is essential to consult with a healthcare professional. With the proper knowledge and resources, leading a heart-healthy life is possible.""")

    with st.expander("The Impact of Sodium on Heart Health: How to Reduce Your Intake"):
       st.image("image/3.jpg")
       st.write("""
            The role of dietary sodium in our health is a topic of considerable interest and debate. While sodium is a necessary mineral for body functions such as nerve transmission and fluid balance, excessive intake can have severe implications for our heart health. Understanding how sodium impacts heart health and practical ways to reduce intake can be vital steps to a healthier life.

            Sodium directly influences our body's fluid balance. It helps to maintain the right balance of fluids, aids in transmitting nerve impulses, and controls the contraction and relaxation of muscles. However, when consumed in excess, sodium can have a detrimental effect on the cardiovascular system. The American Heart Association (AHA) recommends no more than 2,300 milligrams (mg) daily, with an ideal limit of 1,500 mg daily for most adults.
            
            This recommendation is due to sodium's ability to influence blood pressure. Too much sodium in the bloodstream can pull more water into your blood vessels, increasing the total amount (volume) of blood inside them. With more blood flowing through, blood pressure rises, causing the heart to work harder. Over time, this added stress to the heart can lead to heart disease, heart failure, and stroke. A study published in the 'New England Journal of Medicine' reinforced this idea, finding that higher sodium intake was associated with an increased risk of cardiovascular diseases.
            
            Reducing sodium intake is a critical step in maintaining heart health. Here are practical strategies to achieve this:
            
            1. Understand Food Labels: Sodium can sneak into processed and packaged foods. Understanding how to read food labels can help you make lower sodium choices. Foods labeled as 'sodium-free' or 'very low sodium' are the best choices.
            
            2. Cook at Home More Often: By cooking at home, you have more control over the ingredients and, thus, the sodium levels in your meals. Fresh fruits, vegetables, lean meats, and whole grains are naturally low in sodium.
            
            3. Limit Use of Sodium-Rich Condiments: Condiments such as soy sauce, ketchup, and salad dressings can be high in sodium. Opt for low-sodium versions, or make your own to control the sodium content.
            
            4. Eat More Fresh Foods: Fresh fruits and vegetables are naturally low in sodium. Including them more in your meals reduces your sodium intake and increases your intake of heart-healthy nutrients like potassium, which can help lower blood pressure.
            
            Remember, making dietary changes can take time and patience. It's about making small, gradual changes and sticking to them. Consulting with a healthcare professional or a registered dietitian can provide you with additional personalized strategies to reduce your sodium intake and improve your heart health.
            
            In conclusion, while sodium is an essential mineral in our diet, too much can lead to adverse heart health outcomes. Understanding its impact and implementing strategies to reduce intake can positively contribute to your heart health.
            """)

    with st.expander("Heart-Stopping Moments: Demystifying Panic Attacks and Heart Attacks"):
           st.image("image/4.jpg")
           st.write("""
                It's natural to feel concerned when experiencing chest pain or discomfort, as these symptoms can be associated with serious medical conditions. According to Heartify's statistics, these conditions are the most widely common among users. While panic attacks and heart attacks may share certain symptoms, it is crucial to understand the differences between the two in order to respond appropriately and seek the right medical attention.

                Take control of your heart health with Heartify:
                Real-time insights at your fingertips
                Gain valuable, real-time insights into your heart health with Heartify's advanced monitoring technology, providing you with accurate and up-to-date information.
                Personalized recommendations
                Heartify provides personalized recommendations based on your heart health data, helping you make informed lifestyle choices to improve your overall cardiovascular well-being.
                Easy-to-use and convenient
                Heartify's user-friendly interface and seamless integration with your Apple Watch make it easy and convenient to monitor your heart health anytime, anywhere.
                
                Click here to download.
                
                Causes:
                
                Panic attacks are typically caused by psychological factors, such as high levels of stress, anxiety, or a history of panic disorder. External triggers, such as specific phobias or traumatic experiences, can also lead to panic attacks.
                On the other hand, heart attacks, medically known as myocardial infarctions, occur when the blood flow to the heart muscle is blocked. The most common cause of this blockage is the buildup of plaque in the coronary arteries, which restricts blood flow and oxygen supply to the heart. This plaque can rupture, leading to the formation of blood clots that further impede blood flow. Heart attacks are primarily related to cardiovascular health and can be influenced by risk factors like high blood pressure, high cholesterol, smoking, obesity, and diabetes.
                
                The Heartify app provides valuable insights and recommendations based on the collected data. By analyzing heart measurements, the app can suggest personalized lifestyle modifications that promote heart health. These suggestions may include exercise routines, dietary recommendations, stress management techniques, and sleep hygiene practices. By following these guidelines and making conscious efforts to improve heart health, individuals can significantly reduce the likelihood of panic attacks and heart attacks.
                
                Symptoms:
                
                · The symptoms of a panic attack can mimic those of a heart attack, often causing considerable distress. Common symptoms include a rapid heartbeat or palpitations, chest pain or discomfort, shortness of breath, dizziness or lightheadedness, trembling or shaking, sweating, a feeling of impending doom, and a sense of being out of control.
                · While heart attack symptoms can also include chest pain or discomfort, there are additional signs that are more specific to cardiac issues. These may include pain or pressure in the chest that may radiate to the arm, jaw, or back. Other symptoms can include shortness of breath, nausea or vomiting, cold sweats, lightheadedness, and extreme fatigue. By tracking your key heart measurements with Heartify, you can identify early signs of problems and take preventive measures.
                
                The Heartify app allows users to measure their heart rate, HRV, and other vital signs on a daily basis. By establishing a routine of measuring these parameters, individuals can detect any abnormalities or irregularities early on. Sudden changes in numbers can serve as warning signs for potential heart issues or increased stress levels, which, if addressed promptly, can help prevent panic attacks and reduce the risk of heart attacks.
                
                Differentiating Factors:
                
                Several factors can help differentiate between panic attacks and heart attacks:
                Onset: Panic attacks typically occur suddenly and reach their peak within a few minutes. Heart attacks, on the other hand, may have a more gradual onset, with symptoms intensifying over time.
                Triggers: Panic attacks are often triggered by specific situations or phobias, while heart attacks can occur at any time, regardless of external circumstances.
                Action Steps: Given the potential seriousness of both panic attacks and heart attacks, it is important to respond appropriately to the symptoms presented.
                If you suspect you are experiencing a panic attack, try to remain calm and remind yourself that it will pass. Focus on deep breathing and relaxation techniques. If you have a history of panic attacks, you may already have strategies recommended by a mental health professional. Seeking therapy or counseling to address underlying anxiety issues is advisable.
                If you or someone around you experiences symptoms that could indicate a heart attack, it is essential to seek emergency medical attention.
                
                Regular monitoring of heart health through the Heartify app enables users to track trends over time. By observing patterns and fluctuations in heart rate and blood pressure, individuals can identify potential triggers for panic attacks or recognize factors that contribute to an increased risk of heart attacks. This information can empower users to make necessary lifestyle adjustments, such as managing stress levels, adopting healthier eating habits, or incorporating exercise into their daily routine.
                
                The Heartify app can play a crucial role in preventing panic attacks and reducing the risk of heart attacks. By regularly tracking heart health, individuals can detect abnormalities early, make informed lifestyle choices, and receive personalized recommendations for maintaining a healthy heart. Remember, while the app provides valuable insights, it is always essential to consult healthcare professionals for comprehensive evaluations and guidance regarding heart health.
                """)


def about():
    st.write("""
        The app is built based on the 2020 annual CDC survey data of 400k  adults related to their health status,
        using machine learning algorithm called logistic regression with an accuracy of 88%.
        To learn more about the data, check the following link: [Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease). 
        If you are interested to check my code, check my github using the following link: [Github](https://github.com/Konami33/heart-disease-prediction-webapp). Note: this results are not equivalent to a medical diagnosis!  
        """)