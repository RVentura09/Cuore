# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#pip install dash-bootstrap-components
import pandas as pd

import dash
from dash import Dash
from dash import dcc
from dash import html

import numpy as np
import pickle
from dash.dependencies import Input, Output, State
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import dash_bootstrap_components as dbc
from PIL import Image

app = dash.Dash("cuore", external_stylesheets=[dbc.themes.COSMO, dbc.icons.BOOTSTRAP] )

dataset_colnames = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
sample = None   # DataFrame with the data that the user has input in the webpage

#Relative paths respect to current file
THIS_FILE_PATH = str(Path(__file__).parent.absolute())+"/"
filename_to_load = THIS_FILE_PATH + "trained_model_cuore.pickle"

# Load trained model
loaded_model = None
with open(filename_to_load, "rb") as readFile:
    loaded_model = pickle.load(readFile)
    
ALLOWED_TYPES = ("text", "number", "password", "email", "search", "tel", "url", "range", "hidden",)


pil_image = Image.open("prueba2.png")

app.layout = html.Div(
                children = [
                    
                    
                    html.Center(
                            html.Img(src=pil_image , style= { 'height':'10rem', 'width':'89rem','margin':'auto'} , alt='image') 
                            
                            ),

                    
                    
                    html.H1(children ="CUORE", className="display-4 m-4 fs-0.2 text fw-bold text-center border-top border-bottom shadow-sm p-4 mb-5 bg-white rounded ",
                                        style={'color':'#A40606'}),
                    

                    
                    html.P(children = "Welcome to Cuore Clinical Decision Support System!", 
                          className=" display-5 fw-bold fst-italic fs-4  text-center lead" 
                        ), 
                    
                    html.P(children = "The system intended to help doctors with their cardiological patients clinical asessment", 
                          className="fst-italic fs-4  text-center lead" 
                        ), 
                    
                    
                    html.P(children = "Please insert the required values to predict the health status for your patient:", 
                          className="fs-4  text-center lead" 
                        ), 
                    
                    #AGE
                    html.P(children = "1. Age", className="fs-5 mt-0 m-4 fw-bold", style={'color':'#11577D'},
                        ),
                    
                    dcc.Input(
                                id='age'.format("number"),
                                type="number",
                                className=" mt-0 m-4 mb-4 border",
                                value = "",
                                style={'width':'23.5%'},
                                placeholder="Input Age ".format("number"),
                            ),
                    
                    #SEX
                    
                    
                    html.P(children = "2. Sex", className="fs-5 mt-0 m-4 fw-bold", style={'color':'#11577D'},
                        ),
                    
                    
                    html.P(children = "(0 -> Female, 1 -> Male)",
                           className = "mt-0 m-3 fst-italic"
                        ),
                    
                    
                    
                    dcc.Dropdown([0,1], id='Sex', value = "", className= "mt-0 m-4 mb-4", style={'width':'50%'}),
        
                    
                    
                   #CHEST PAIN 
                   html.P(children = "3. Chest Pain Type", className="fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                       ),
                   
                   html.P(children = "(1 -> Typical Angina, 2 -> Atypical Angina, 3 -> Non-Anginal, 4 -> Asymptomatic)", 
                          className= " mt-0 m-3 fst-italic"
                       ),
                   
                   dcc.Dropdown([1, 2, 3, 4], id='cp', value = "", className= "mt-0 m-4 mb-4", style={'width':'50%'}),
                   
                   
                   
                   #RESTING BLOOD PRESSURE
                    html.P(children = "4. Resting Blood Pressure", className=" fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                        ),
                    
                    dcc.Input(
                               id="rbp".format("number"),
                               type="number",
                               value = "",
                               className = "mt-0 m-4 mb-4 border",
                               style={'width':'23.5%'},
                               placeholder="Input trestbps ".format("number"),
                           ),
                    
                    
                  #SERUM CHOLESTEROL
                   html.P(children = "5. Serum Cholesterol in mg/dl ", className="fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                       ),
                   
                   dcc.Input(
                              id="sc".format("number"),
                              type="number",
                              className = "mt-0 m-4 mb-4 border",
                              style={'width':'23.5%'},
                              value = "",
                              placeholder="Input Chol ".format("number"),
                          ),
                   
                   
                   #FASTING BLOOD SUGAR
                   html.P(children = "6. Fasting blood sugar", className=" fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                       ),
                   html.P(children = "(If it is greater than 120 mg/dl select 1 otherwise 0)", 
                          className= "mt-0 m-3 fst-italic"
                       ),
                   
                   dcc.Dropdown([0,1], id='fbp', value = "", className = "mt-0 m-4 mb-4 ", style={'width':'50%'}),
                   
                   
                   
                   #RESTING ELECTROCARDIOGRAPHIC RESULT
                   html.P(children = "7. Resting Electrocardiographic Result", className="fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                       ),
                   html.P(children = "(0 -> normal, 1 -> St-T wave abnormality, 2 -> probable or definite hypertropy)",
                          className = "mt-0 m-3 fst-italic"
                       ),
                   
                   dcc.Dropdown([0,1,2], id='ecg', value = "", className = "mt-0 m-4 mb-4 ", style={'width':'50%'}),
                   
                   
                   #MAXIMUM HEART RATE ACHIEVED
                   html.P(children = "8. Maximum Heart Rate Achieved", className=" fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                       ),
                   
                   dcc.Input(
                              id="thalach".format("number"),
                              type="number",
                              value = "",
                              className = "mt-0 m-4 mb-4 border",
                              style={'width':'23.5%'},
                              placeholder="Input thalach".format("number"),
                          ),
                   
                   
                  #EXERCISE INDUCED ANGINA 
                  html.P(children = "9. Exercise Induced Angina", className="fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                      ),
                  html.P(children = "((1 = Yes, 0 = No)", className = "mt-0 m-3 fst-italic"
                      ),
                  
                  dcc.Dropdown([0,1], id='exercise_ang', value = "", className = "mt-0 m-4 mb-4 ", style={'width':'50%'} ), 
                  
                  
                  #ST DEPRESSION INDUCED BY EXERCISE RELATIVE TO REST
                  html.P(children = "10. ST Depression Induced by Exercise Relative to Rest", className=" fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                      ),
                  
                  dcc.Input(
                             id="OLDPEAK".format("number"),
                             type="number",
                             value = "",
                             className = "mt-0 m-4 mb-4 border",
                             style={'width':'23.5%'},
                             placeholder="Input oldpeak".format("number"),
                         ),
                  
                  
                  #SLOPE OF THE PEAK EXERCISE ST SEGMENT 
                  html.P(children = "11. Slope of the Peak Exercise ST Segment", className=" fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                      ),
                  html.P(children = "(1 -> Upslopping, 2 -> Flat, 3 -> Downslopping)", className= "mt-0 m-3 fst-italic"
                      ),
                  
                  dcc.Dropdown([1,2,3], id='slope', value = "", className = "mt-0 m-4 mb-4 ", style={'width':'50%'}), 
                  
                  
                  #NUMBER OF MAJOR VESSELS COVERES BY FLOUROSCOPY 
                  html.P(children = "12. Number of Major Vessels (0-3) covered by Flourosopy", className="fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                      ),
                 
                  dcc.Dropdown([0,1,2,3],id='vessels', value = "", className = "mt-0 m-4 mb-4 ", style={'width':'50%'}),  
                  
                  
                  #THALASSEMIA 
                  html.P(children = "13. Thalassemia", className=" fs-5 m-4 fw-bold ", style={'color':'#11577D'}
                      ),
                  html.P(children = "(3 -> Normal, 6 -> Fixed Defect, 7 -> Reversible Defect)", className= "mt-0 m-3 fst-italic"
                      ),
                  
                  dcc.Dropdown([3,6,7], id ='th', value = "", className = "mt-0 m-4 mb-4 ", style={'width':'50%'}), 
                  
                  
                  
                  html.Center( 
                      html.Div([
                          html.Br(),   
                          html.H4(html.B('Classification result', id='classification-result', style={'color':'#305252'}, 
                                 className = "mt-0 m-2 mb-4 ")),
                          html.Button('Predict for patient', id='submit', style={'margin':'0 auto', 'width':'30%', 'color':'#11577D' },
                              className = "mt-0 m-4 mb-4 border-primary border-2 border rounded-pill shadow p-2 mb-5 bg-white rounded"
                              ),
                
                       
                          ])
                      ),
                  
                  
                  dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"),
                             "Warning! This is not a final diagnosis. Doctors will use these results for clinical asessment "]
                            ,color="warning",
                            className="d-flex align-items-center text-center",
                            
                            
                  
                  )
                  
                  
                  
                  
                  
                  
                   ])

@app.callback(    
    Output(component_id='classification-result', component_property='children'),
    [Input(component_id='submit', component_property='n_clicks')],
    [State('age', 'value'),
    State('Sex', 'value'),
    State('cp', 'value'),
    State('rbp', 'value'),
    State('sc', 'value'),
    State('fbp', 'value'),
    State('ecg', 'value'),
    State('thalach', 'value'),
    State('exercise_ang', 'value'),
    State('OLDPEAK', 'value'),
    State('slope', 'value'),
    State('vessels', 'value'),
    State('th', 'value'),
    ]
)
def execute_classification(n_clicks, age, sex, cp,trestbps, chol, fbs, restecg, thalach,
       exang, oldpeak, slope, ca, thal):
    """
    Main method. Loads the trained model, applies the input data and returns a class
    """
    
    if(n_clicks == None): # When the application open
        return "If you have filled out all patient information, click predict for patient"
    else:
        # The sliders' values are already parsed to numeric values
        # Here we create a DataFrame with the input data
        data_from_user = [age, sex, cp,trestbps, chol, fbs, restecg, thalach,
               exang, oldpeak, slope, ca, thal]
        global sample
        
                
        sample = pd.DataFrame(data=[data_from_user], columns=dataset_colnames)

        numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        X = sample[numerical] 
        scaler=MinMaxScaler() #change to min max scaler option --> this one is better 
        #scaler = StandardScaler() #standarscaler option 
        df_standard = scaler.fit_transform(X) #fit and transform in the same row
        df_standard = pd.DataFrame(df_standard, columns=numerical)
        
        standardized = sample.copy()
        standardized[numerical] = df_standard[numerical]








        # Execute the prediction using the loaded trained model.
        prediction = loaded_model.predict(standardized)

        # Return final message
        prediction_labels = ["Healthy", "Not Healthy"]
        return "The patient is: ["+ str(prediction[0]) +":" + prediction_labels[prediction[0]] + "]"


















if __name__ == "__main__":
    app.run_server(debug=True)