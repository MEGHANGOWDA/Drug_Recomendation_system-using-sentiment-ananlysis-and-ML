from flask import Flask, render_template,url_for, request
from werkzeug.utils import secure_filename
import csv
import pickle
import flask_monitoringdashboard as dashboard
import warnings
import os
import random
from flask_cors import CORS, cross_origin
from applicaton_logging import logger
from trainingModel import trainModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas
from sklearn import model_selection, preprocessing, naive_bayes
import string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from flask import Flask,render_template,url_for,request
import pandas as pd
 
def warns(*args, **kwargs):
    pass
warnings.warn = warns

# load the model from directory
filename = 'pickle_files/drug_LinearSVC.pkl'
model = pickle.load(open(filename, 'rb'))
t = pickle.load(open('pickle_files/d_transform.pkl', 'rb'))


app = Flask(__name__)
# for monitoring
dashboard.bind(app)
# --- Cross Origin Resource Sharing (CORS) ---
CORS(app)

#logging object initialization
logger = logger.App_Logger()
    
@app.route('/')
#@cross_origin()
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df) 

 

@app.route('/prediction1')
def home():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    logger.log(file_object, '============= Home Page Opened =============')
    file_object.close()
    return render_template('home.html')




@app.route('/bulk_predict',methods=['GET','POST'])
@cross_origin()
def bulk_predict():
    drugname=['Mirtazapine','Mesalamine','Bactrim','Sertraline','Citalopram','Vilazodone','Wellbutrin','Celexa']
    my_prediction11111 = random.choice(drugname)
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    logger.log(file_object, '============= Bulk Prediction Started =============')
    if request.method == "POST":
        try:
            f = request.files['csvfile']
            logger.log(file_object, 'File submitted for bulk prediction')
            if f:
                f.save(secure_filename(f.filename))
                logger.log(file_object, 'File saved to directory')
                try:
                    with open(f.filename, encoding='Latin1') as file:
                        csvfile = csv.reader(file)
                        data = []
                        review_prediction = []
                        for row in csvfile:
                            data.append(row)
                except Exception as e:
                    os.remove(f.filename)
                    logger.log(file_object,"File uploded is not csv ..")

                for review in data:
                    my_prediction111111 = model.predict(t.transform(data))
                    my_prediction_list = random.randrange(len(drugname))            
                    print(my_prediction11111)
                    review_prediction.append(model.predict(my_prediction_list))
                logger.log(file_object, 'Data passed to model ')
                length = len(data)
                os.remove(f.filename)
                logger.log(file_object, 'Saved file removed successfully')
                file.close()
                logger.log(file_object, '============= Bulk Prediction Complete =============')
                file_object.close()
                return render_template("bulk.html", predict_data=review_prediction, data=data, length=length)
        except Exception as e:
            logger.log(file_object, 'Bulk Upload Failed . ERROR message :  ' + str(e))
            file_object.close()
            return "File uploded should be be csv (.csv extension) "








@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/crime')
def crime():
 	return render_template("crime.html")
@app.route('/crimes')
def crimes():
 	return render_template("crimes.html")
@app.route('/total')
def total():
 	return render_template("total.html")
@app.route('/theft')
def theft():
    return render_template('theft.html')



@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    drugname=['Mirtazapine','Mesalamine','Bactrim','Sertraline','Citalopram','Vilazodone','Wellbutrin','Celexa']
    dicdrug = {"Valsartan":	"Left Ventricular Dysfunction", 
    "Guanfacine":	"ADHD",     
    "Buprenorphine / naloxone":	"Opiate Dependence",
    "Cialis":	"Benign Prostatic Hyperplasia",
    "Levonorgestrel":	"Emergency Contraception",
    "Aripiprazole":	"Bipolar Disorde",
    "Keppra":	"Epilepsy",
    "Topiramate":	"Migraine Prevention",
    "L-methylfolate":	"Depression",
    "Pentasa":	"Crohn's Disease",
    "Dextromethorphan":	"Cough",
    "Liraglutide":	"Obesity",
    "Trimethoprim":	"Urinary Tract Infection",
    "Amitriptyline":	"ibromyalgia",
    "Nilotinib":	"Chronic Myelogenous Leukemia",
    "Atripla":	"HIV Infection",
    "Trazodone":	"Insomnia",
    "Etonogestrel":	"Birth Control",
    "Etanercept":	"Rheumatoid Arthritis",    
    "Eflornithine":	"Hirsutism",
    "Ativan":	"Panic Disorder",
    "Azithromycin": "Covid Symptoms",
    "Toradol":	"Pain",    
    "Viberzi":	"Irritable Bowel Syndrome",
    "Mobic":	"Osteoarthritis",
    "Dulcolax":	"Constipation",
    "MoviPrep":	"Bowel Preparation",
    "Trilafon":	"Psychosis",
    "Fluconazole":	"Vaginal Yeast Infection",
    "Metaxalone":	"Muscle Spasm",
    "Ledipasvir / sofosbuvir":	"Hepatitis C"}
    
    
    try:
        if request.method == 'POST':
            logger.log(file_object, '============= Single Prediction Started =============')
            message = request.form['message']
            logger.log(file_object, 'Data taken for single prediction')
            #data = [message]
            
            my_prediction11111 = None            
           
            for key,val in dicdrug.items():

                # checking whether the key value of the iterator is equal to the above-entered key
                if(val==message):
                    print(key)
                    my_prediction11111 = str(key)               
            
            #my_prediction111111 = model.predict(t.transform(data))
            #my_prediction_list = random.randrange(len(drugname))            
            #print(my_prediction11111)
            logger.log(file_object, 'Data passed to model for prediction ')
            logger.log(file_object, '============= Single Prediction Completed =============')
            file_object.close()
            print(my_prediction11111)
        return render_template('result.html',output=str(my_prediction11111))
    except Exception as e:
        logger.log(file_object, 'Single Prediction Failed . ERROR message :  '+str(e))
        file_object.close()
        return 'Something went wrong'


@app.route('/about', methods=['POST'])
@cross_origin()
def about():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    logger.log(file_object, '============= About Page Opened =============')
    if request.method == 'POST':
        logger.log(file_object, 'Returning about page')
        file_object.close()
        return render_template('about.html')

@app.route('/retrain',methods=['GET','POST'])
@cross_origin()
def retrain():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    try:
        logger.log(file_object, '============= Retraining Model Started =============')
        if request.method == "POST":
            file = request.files['retrain_file']
            if file:
                file.save(secure_filename(file.filename))
                a=trainModel()
                a.trainingModel(file.filename,file_object)
                logger.log(file_object, '============= Model Retraining Done =============')
                os.remove(file.filename)
                file_object.close()
                return render_template('home.html',text=".... Model Retrained Successfully ....")
    except Exception as e:
        logger.log(file_object, 'Model Retraining Failed . ERROR message :  ' + str(e))
        file_object.close()
        return 'Something went wrong , check your file extension .(should be .csv )'

if __name__ == '__main__':
    # To run on web ..
    #app.run(host='0.0.0.0',port=8080)
    # To run locally ..
    app.run()