from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

def prediction(list):
    with open('Model/price_predictor.pickle', 'rb') as file:
        model= pickle.load(file)
    #  # Create a DataFrame for the feature list with the appropriate column names
        columns = ['Ram', 'Weight', 'Touchscreen', 'Ips', 'Company_Acer', 'Company_Apple',
                    'Company_Asus', 'Company_Dell', 'Company_HP', 'Company_Lenovo',
                    'Company_MSI', 'Company_Other', 'Company_Toshiba',
                    'TypeName_2 in 1 Convertible', 'TypeName_Gaming', 'TypeName_Netbook',
                    'TypeName_Notebook', 'TypeName_Ultrabook', 'TypeName_Workstation',
                    'OpSys_Linux', 'OpSys_Mac', 'OpSys_Other', 'OpSys_Windows',
                    'cpu_name_AMD', 'cpu_name_Intel Core i3', 'cpu_name_Intel Core i5',
                    'cpu_name_Intel Core i7', 'cpu_name_Other', 'gpu_name_AMD',
                    'gpu_name_Intel', 'gpu_name_Nvidia']
    
    input_data = pd.DataFrame([list], columns=columns)  # Use column names for the DataFrame
    pred_value = model.predict(input_data) 
    return pred_value


@app.route('/', methods=['POST','GET'])
def index():
    pred_value = 0
    if request.method == 'POST':
        
        ram = request.form['ram']
        weight = request.form['weight']
        company = request.form['company']
        typename = request.form['typename']
        opsys = request.form['opsys']
        cpu = request.form['cpuname']
        gpu = request.form['gpuname']
        touchscreen = request.form.get('touchscreen')
        ips = request.form.get('ips')
        

        # Prepare the feature list in the correct order as per the training set
        feature_list = [
            int(ram),           # RAM
            float(weight),      # Weight
            int(touchscreen == 'Yes'),  # Touchscreen (convert to 1 or 0)
            int(ips == 'Yes')   # IPS (convert to 1 or 0)
        ]

        company_list = ['acer','apple','asus','dell','hp','lenovo','msi','other','toshiba']
        typename_list = ['2in1convertible','gaming','netbook','notebook','ultrabook','workstation']
        opsys_list = ['linux','mac','other','windows']
        cpu_list = ['amd','intelcorei3','intelcorei5','intelcorei7','other']
        gpu_list = ['amd','intel','nvidia']

        def getValue(list,value):
            for item in list:
                if item == value:
                    feature_list.append(1)
                else:
                    feature_list.append(0)
        
        getValue(company_list,company)
        getValue(typename_list,typename)
        getValue(opsys_list,opsys)
        getValue(cpu_list,cpu)
        getValue(gpu_list,gpu)
        
        
        pred_value = prediction(feature_list)
        pred_value = np.round(pred_value[0],2)*317.17
        print(pred_value)


    return render_template("index.html", pred_value=pred_value)

if __name__=='__main__':
    app.run(debug=True)