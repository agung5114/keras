from flask import Flask, render_template, request, redirect, flash, url_for,send_from_directory, jsonify
# import main
# from main import getPrediction
import urllib.request
from werkzeug.utils import secure_filename
import os
import sys
import requests
import numpy as np

import keras
import joblib

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import decode_predictions

from PIL import Image
sys.modules['Image'] = Image 
 
UPLOAD_FOLDER = 'E:/Gaspoltech/keras/static/uploads'
# UPLOAD_FOLDER = 'E:/CODING/datathon/maritime/ikanapa/static/uploads'

app = Flask(__name__)
model = keras.models.load_model('E:/Gaspoltech/keras/fish_classification.h5')
datagen = joblib.load('E:/Gaspoltech/keras/data_generator.joblib')

def getPrediction(filename):
    image = load_img('static/uploads/'+filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = yhat[0]
    # np.round(yhat[2],2)]
    # [np.round(model.predict(image),2)]
    # label = decode_predictions(yhat)
    print(f'''jenis ikan: {np.round(yhat,2)}''')
    return np.round(label[0],2),np.round(label[1],2),np.round(label[2],2)
    # label = label[0][0]
    # print('%s (%.2f%%)' % (label[1], label[2]*100))
    # return label[1], label[2]*100

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/') 
def home():
    return render_template("Home.html")
    # return render_template("index.html")

# @app.route('/market')
# def market():
#     return redirect('http://fismart.gaspol.tech')

@app.route('/fismartbi')
def fismartbi():
    return render_template("fismartbi.html")

@app.route('/fishingbi')
def fishingbi():
    return render_template("fishingbi.html")

@app.route('/price')
def price():
    return render_template("price.html")

@app.route('/weather')
def weather():
    return render_template("weather.html")

@app.route('/fishrec')
def fishrec():
    return render_template("fishrec.html")

@app.route('/govdash')
def govdash():
    return render_template("govdash.html")

@app.route('/productivity')
def productivity():
    return render_template("productivity.html")

@app.route('/fishingwatch')
def fishingwatch():
    return render_template("fishingwatch.html")

@app.route('/finfunds')
def finfunds():
    return render_template("finfunds.html")

@app.route('/harga')
def harga():
    return render_template("Page-1.html")

@app.route('/fishrec', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            # return redirect(request.url)
            return redirect('/fishrec')
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            # return redirect(request.url)
            return redirect('/fishrec')
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            getPrediction(filename)
            bawal,bluefin,lainnya = getPrediction(filename)
            flash(bawal)
            flash(bluefin)
            flash(lainnya)
            # label, acc = getPrediction(filename)
            # flash(label)
            # flash(acc)
            # flash(filename)
            # return redirect(url_for('uploaded_file',filename=filename))
            return render_template('/fishrec.html',filename=filename)

# @app.route('/harga/display/<filename>')
@app.route('/uploads/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' +filename), code=301)
    # return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run (debug = True)
