from flask import Flask,render_template,request,redirect,url_for,session,flash
import mysql.connector
import os
from werkzeug.utils import secure_filename
#les outils pour IA
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing  import image 
import numpy as np 
import random
# config de l'app 
app=Flask(__name__)
#clé secréte
app.secret_key = "super_cle_secrete_projet_ai"

#sauv img
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#chargement du modele
model = load_model('vgg16_skin_cancer.h5')

#creà du dossier
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

#conn à base de données
#https://console.clever-cloud.com/organisations/user_0b13559e-b081-4d51-ab9e-28631f9e4263/addons/mysql/addon_a8a3476d-205e-4c84-a7db-adc7d8dc30e6
def get_db_connection():
    conn = mysql.connector.Connect(
        host='bqynuhxhafc1xfwlg1r2-mysql.services.clever-cloud.com',
        user='umx9khy5fnq1ovdr',
        password='YJoEAII15UbiNPSIC0DG',
        database='bqynuhxhafc1xfwlg1r2'
    )
    return conn

#test page d'accueil
@app.route('/')
def home():
    if 'loggedin' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# les route
@app.route('/login',methods=['GET','POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form :
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s',(username,password))
        account = cursor.fetchone()
        cursor.close()
        conn.close()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect(url_for('dashboard'))
        else :
            msg = 'Nom d\'utilisateur ou mot de passe incorect !'
    return render_template('login.html',msg=msg)

@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE username = %s",(username,))
        existing_user = cursor.fetchone()

        if existing_user:
            return render_template('signup.html',error="ce nom d'utilisateur est déja utilisé.")
        sql = "INSERT INTO users (username,password) VALUES (%s,%s)"
        val = (username, password)
        cursor.execute(sql,val)
        conn.commit()

        cursor.close()
        conn.close()

        return redirect(url_for('login'))
    return render_template('signup.html')
@app.route('/dashboard')
def dashboard():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html',username=session['username'])

@app.route('/logout')
def logout():
    session.pop('loggedin',None)
    session.pop('id',None)
    session.pop('username',None)
    return redirect(url_for('login'))

@app.route('/predict',methods=['POST'])
def predict():
    patient_name = request.form.get('name')
    patient_age = request.form.get('age')
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(filepath)

        img = image.load_img(filepath,target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array,axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        probability = float(prediction[0][0])
        

        if probability > 0.5:
            result_text = "Malignant"
        else:
            result_text = "Benign"
        
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = "INSERT INTO patients (name,age,result,probability,image_path) VALUES (%s,%s,%s,%s,%s)"
        val = (patient_name,patient_age,result_text,probability,filepath)

        cursor.execute(sql,val)
        conn.commit()
        cursor.close()
        conn.close()

        return render_template('result.html',result=result_text , probability=probability , image_path=filepath)
    return redirect(url_for('dashboard'))
@app.route('/patients')
def list_patients():
    # 1. Connexion à la base de données
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True) # dictionary=True permet d'accéder aux colonnes par leur nom
    
    # 2. Récupérer tous les patients (du plus récent au plus ancien)
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    patients_list = cursor.fetchall()
    
    # 3. Fermer la connexion
    cursor.close()
    conn.close()
    
    # 4. Envoyer les données au fichier HTML
    return render_template('patients.html', patients=patients_list)

@app.route('/clear_history',methods=['post'])
def clear_history():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("TRUNCATE TABLE patients")
    conn.commit()

    cursor.close()
    conn.close()

    return redirect(url_for('list_patients'))

if __name__ == '__main__':
    app.run(debug=True)