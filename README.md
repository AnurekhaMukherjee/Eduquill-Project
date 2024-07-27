# Eduquill-Project
Hospital Readmission Project
Forecasting Readmissions to Hospitals
# Contents Table 
Project Synopsis
Stating The Problem
Solutions
Work Flow
Describing the data
Issues Faced during Solving
Technicalities Used For Solving the Problem
# Project Synopsis
This project's main objective is to develop a predictive model that can identify patients who, within 30 days after their initial release, are at high risk of being readmitted to the hospital. With the use of this predictive model, healthcare professionals will be able to tailor their treatments and reduce readmission rates while also improving patient outcomes.

# Stating The Problem
Accurately anticipating hospital readmissions is a challenge for the healthcare sector, and the result could be higher healthcare expenditures and lower patient satisfaction. This study uses machine learning approaches to identify patients who are at high risk of hospital readmission by creating a prediction model based on patient data.

# Solutions
1. Data preprocessing involves cleaning and preparing the medical data, dealing with missing values, encoding categorical categories, and guaranteeing consistency and quality of the data.
2. Using the patient's demographics, medical history, past hospital stays, and other clinically significant information, feature engineers can create features that are pertinent to the patient from the data that is already accessible.
3. Model Building: Creating a statistical or machine learning model that can forecast the chance of a readmission to the hospital in less than 30 days.
4. Model Evaluation: Using common binary classification evaluation measures including accuracy, precision, recall, F1-score, ROC curve, and AUC, evaluate the prediction model's performance.

# Describing the data
The following columns are included in the dataset:
1.Patient_ID: A special number that only each patient has.
2.The patient's age expressed in years.
3.Gender: The patient's gender (Male, Female, Other, etc.).
4.Admission_Type: The category of admission, such as elective, urgent, or emergency.
5. Diagnosis: The patient's initial diagnosis upon admission (e.g., Diabetes, Heart Disease, Injury, Infection).
6. Num_Lab_Procedures: The total number of laboratory tests carried out while the patient was in the hospital.
7. Num_Medications: The total number of drugs that the patient has been prescribed.
8. Num_Outpatient_Visits: The total number of outpatient appointments made before the present hospitalization.
9. Num_Inpatient_Visits: The total number of inpatient visits that took place before the present hospitalization.
10. Num_Emergency_Visits: The total number of Emergency room visits made before the present hospital admission.
11. Number of diagnoses, or Num_Diagnoses.
12. A1C_Result: The A1C test result (Normal, Abnormal), if available.
13. Readmitted: This indicates (e.g., Yes, No) if the patient was readmitted to the hospital within a certain period of time.

# Issues Faced during Solving
A particular issue with this project was the large number of null entries in the A1C_Result column. In order to solve this, a classification model that makes use of additional characteristics was created in order to forecast and complete the missing values in the A1C_Result column. Using the finished dataset, a predictive model for hospital readmission status was constructed after the missing variables were imputed.

# Technicalities Used For Solving the Problem
Python Scikit-learn Seaborn Streamlit Pandas NumPy



