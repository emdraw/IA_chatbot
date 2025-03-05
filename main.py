import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import json
import random
import pickle


# Inicializar el stemmer y el analizador de sentimientos
stemmer = LancasterStemmer()
nltk.download('vader_lexicon')

# Cargar los datos
with open(r"conversations.json", encoding='utf-8') as archivo:
    data = json.load(archivo)

# Cargar o preparar datos de entrenamiento
try:
    with open(r"variables.pickle", "rb") as archivoPickle:
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except FileNotFoundError:
    palabras = []
    tags = []
    auxX = []
    auxY = []

    for contenido in data["intents"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])
            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida = []
    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        fila_salida = salidaVacia[:]
        fila_salida[tags.index(auxY[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(fila_salida)

    entrenamiento = np.array(entrenamiento)
    salida = np.array(salida)
    with open(r"variables.pickle", "wb") as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

# Crear y entrenar el modelo con TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(entrenamiento[0]),)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(len(salida[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(entrenamiento, salida, epochs=5000, batch_size=300, verbose=1)
model.save(r"saved_model.tflearn")


# Función principal del bot
def mainbot():
    while True:
        entrada = input("[+]Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1
        resultados = model.predict(np.array([cubeta]))
        resultadosIndices = np.argmax(resultados)
        tag = tags[resultadosIndices]

        for tagAux in data["intents"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]

        print("[+]Bot: ", random.choice(respuesta))

mainbot()
