import os
import pathlib
import requests
import math
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import IPython
import numpy as np
import seaborn as sns
from pydub import AudioSegment, effects

import noisereduce as nr
import librosa
import soundfile as sf

from PIL import Image
import cv2

import uuid

import tensorflow as tf
#import tensorflow_hub as hub


from flask import Flask, request, render_template, redirect, url_for, session

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])


lista_pajaros = [
    ('Gavilán común', 'Accipiter_nisus'),
    ('Mito común', 'Aegithalos_caudatus'),
    ('Alondra común', 'Alauda_arvensis'),
    ('Perdiz roja', 'Alectoris_rufa'),
    ('Ánade real', 'Anas_platyrhynchos'),
    ('Bisbita común', 'Anthus_pratensis'),
    ('Vencejo común', 'Apus_apus'),
    ('Garza real', 'Ardea_cinerea'),
    ('Búho chico', 'Asio_otus'),
    ('Busardo ratonero', 'Buteo_buteo'),
    ('Jilguero europeo', 'Carduelis_carduelis'),
    ('Cetia ruiseñor', 'Cettia_cetti'),
    ('Chorlitejo chico', 'Charadrius_dubius'),
    ('Aguilucho cenizo', 'Circus_pygargus'),
    ('Paloma bravía', 'Columba_livia'),
    ('Cuervo grande', 'Corvus_corax'),
    ('Cuco común', 'Cuculus_canorus'),
    ('Herrerillo común', 'Cyanistes_caeruleus'),
    ('Avión común', 'Delichon_urbicum'),
    ('Escribano soteño', 'Emberiza_cirlus'),
    ('Petirrojo europeo', 'Erithacus_rubecula'),
    ('Cernícalo vulgar', 'Falco_tinnunculus'),
    ('Papamoscas cerrojillo', 'Ficedula_hypoleuca'),
    ('Cogujada común', 'Galerida_cristata'),
    ('Golondrina común', 'Hirundo_rustica'),
    ('Alcaudón común', 'Lanius_meridionalis'),
    ('Terrera común', 'Lullula_arborea'),
    ('Ruiseñor común', 'Luscinia_megarhynchos'),
    ('Abejaruco europeo', 'Merops_apiaster'),
    ('Milano negro', 'Milvus_migrans'),
    ('Papamoscas gris', 'Muscicapa_striata'),
    ('Oropéndola europea', 'Oriolus_oriolus'),
    ('Carbonero común', 'Parus_major'),
    ('Gorrión común', 'Passer_domesticus'),
    ('Abejero europeo', 'Pernis_apivorus'),
    ('Colirrojo tizón', 'Phoenicurus_ochruros'),
    ('Mosquitero común', 'Phylloscopus_collybita'),
    ('Urraca común', 'Pica_pica'),
    ('Chorlito dorado europeo', 'Pluvialis_apricaria'),
    ('Chova piquirroja', 'Pyrrhocorax_pyrrhocorax'),
    ('Reyezuelo sencillo', 'Regulus_regulus'),
    ('Tarabilla común', 'Saxicola_rubicola'),
    ('Becada común', 'Scolopax_rusticola'),
    ('Tórtola europea', 'Streptopelia_turtur'),
    ('Estornino pinto', 'Sturnus_vulgaris'),
    ('Vencejo real', 'Tachymarptis_melba'),
    ('Chochín común', 'Troglodytes_troglodytes'),
    ('Mirlo común', 'Turdus_merula'),
    ('Abubilla europea', 'Upupa_epops'),
    ('Avefría europea', 'Vanellus_vanellus')
]



# Crear el diccionario
diccionario_especies = {}
diccionario_imagenes = {}


for i in range(0, len(lista_pajaros)):
    diccionario_especies[i] = lista_pajaros[i]
    diccionario_imagenes[i] = "static/imagenes_especies/"+ str(i+1) +".jpg"

# Configura la clave secreta para usar sesiones
app.secret_key = 'supersecretkey'  # Cambia esto a algo más seguro en producción





def cargar_modelo():
    image_size = (255, 255)  # Size to resize images to after they are read from disk
    input_shape = (*image_size, 3)  # Arbitrary
    include_top = False  # Whether to include the layers at the top of the network


    model =  tf.keras.applications.xception.Xception(include_top=include_top, input_shape=input_shape)
    model.trainable = True

    for layer in model.layers[:110]: #400
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    #resize = tf.keras.Sequential([
    #  tf.keras.layers.Resizing(256, 256),
    #])

    #x = resize(inputs)

    x = tf.keras.applications.xception.preprocess_input(inputs)

    #x = tf.keras.layers.BatchNormalization()(x)

    x = model(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    units = 50
    outputs = tf.keras.layers.Dense(units, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    lr = 0.001
    optimizer = tf.keras.optimizers.Adam(lr)  # Optimizer instance
    metrics = ["accuracy"]  # List of metrics to be evaluated by the model during training and testing
    loss = "sparse_categorical_crossentropy"  # Loss function

    model.compile(optimizer, loss, metrics)

    model.load_weights('static\\models\\Modelo_xception_50_aumento_datos_division_class_weights_72%.weights.h5')
    return model

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['filepath'] = filepath
        
        # Procesar el archivo de audio

        
        
        return redirect(url_for('loading'))
    return redirect(request.url)


@app.route('/loading')
def loading():
    return render_template('loading.html', redirect_url=url_for('result'))

@app.route('/result')
def result():
    inferencia()
    conclusion = session.get('conclusion', '')
    img_paths = session.get('img_paths', '')
    return render_template('result.html', conclusion=conclusion, image_paths=img_paths)

def inferencia():
    filepath = session.get('filepath', '')
    
    imagenes_path = process_audio(filepath)
    pred = []
    rutas_static_img = []
    dict_count = {}
    conclusion = []
    conclusion_par = []

    for image_path in imagenes_path:
        imagen = tf.keras.preprocessing.image.load_img(image_path, target_size=(255, 255))  # Ajusta el tamaño según las necesidades de tu modelo
        imagen_array = tf.keras.preprocessing.image.img_to_array(imagen)
        imagen_array = np.expand_dims(imagen_array, axis=0)
        pred.append(np.argmax(model.predict(imagen_array)))
    
    for p in pred:
        if not p in dict_count.keys():
            dict_count[p] = 0
        dict_count[p] += 1
    
    for bird in set(dict_count.keys()):
        conclusion_par.append((dict_count[bird] / len(pred) * 100, f"{(dict_count[bird] / len(pred) * 100):.2f}% Nombre común: {diccionario_especies[bird][0]} Nombre científico: {diccionario_especies[bird][1]}", bird))

    conclusion_par.sort(key=lambda x: x[0], reverse = True)
        
    for bird in conclusion_par:
        conclusion.append(bird[1])
        rutas_static_img.append(diccionario_imagenes[bird[2]])

    session['conclusion'] = conclusion
    session['img_paths'] = rutas_static_img

def process_audio(filepath):
    # Usamos librosa para cargar el archivo de audio y calcular su duración
    audio, sr = librosa.load(filepath)

    audio, sr = reduce_sampling_rate(audio, sr, 16000)

    audio = reduce_noise(audio, sr)

    imagenes = get_mel_spectogram(audio=audio, sr=sr)

    return imagenes

    
def reduce_sampling_rate(audio, sr, target_sr):

    y_resampled = librosa.util.normalize(audio)

    # Reducir la tasa de muestreo
    y_resampled = librosa.resample(y_resampled, orig_sr=sr, target_sr=target_sr)

    # Guardar el archivo de audio con la nueva tasa de muestreo
    return y_resampled, target_sr

def reduce_noise(audio, sr):

    # Reducir el ruido del audio
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)

    # Guardar el archivo de audio filtrado
    return reduced_noise

def get_mel_spectogram(audio, sr):
    #Empiezo a trocear
    index_trim = librosa.effects.split(audio, top_db=10, frame_length=8192, hop_length=4096)
    image_paths = []
    unique_id = uuid.uuid4()
    i = 0
    for index in index_trim:
        # Calcular el espectrograma mel
        espectrograma = librosa.feature.melspectrogram(y=audio[index[0]:index[1]], sr=sr)

        # Convertir el espectrograma a decibeles (dB)
        espectrograma_dB = librosa.power_to_db(espectrograma, ref=np.max)

        # Visualizar el espectrograma mel
        plt.figure(figsize=(32, 32))
        img_buffer = io.BytesIO()
        librosa.display.specshow(espectrograma_dB, sr=sr)
        plt.savefig(img_buffer, bbox_inches='tight', pad_inches=0, format = 'jpg')
        img_buffer.seek(0)

        img = Image.open(img_buffer)
        image_path = os.path.join(app.config['RESULT_FOLDER'], f'image_{unique_id}_{i}.jpg')
        img.save(image_path)
        image_paths.append(f'static/results/image_{unique_id}_{i}.jpg')
        plt.close('all')
        i+=1

    return image_paths

if __name__ == "__main__":
    model = cargar_modelo()
    app.run(debug=True, port=5001)

