import io, base64 # Para decodificar la imagen recibida
import numpy as np # Cálculos numéricos
from PIL import Image # Manejo de imágenes
from flask import Flask, request, jsonify, render_template # Servidor web
from tensorflow.keras.models import load_model # Para cargar la CNN

app = Flask(__name__) # Crea la aplicación
modelo = load_model("modelo_cnn.keras") # Carga la CNN entrenada
ACCURACY = 98.61 # Accuracy de la CNN en el conjunto de test (%)

def preparar(data_url): # Convierte el dibujo en la entrada de la CNN
    datos = base64.b64decode(data_url.split(",")[1]) # Decodifica la imagen enviada
    img = Image.open(io.BytesIO(datos)).convert("L") # Pasa a escala de grises
    arr = np.array(img) # Imagen como matriz
    ys, xs = np.where(arr > 30) # Ubica los píxeles del trazo
    if len(xs) == 0: # Si no hay dibujo
        return None # No hay nada que predecir
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1] # Recorta el dígito
    img = Image.fromarray(arr) # Vuelve a imagen
    escala = 20 / max(img.size) # Para que el dígito mida 20 píxeles como máximo
    img = img.resize((max(1, round(img.width * escala)), max(1, round(img.height * escala)))) # Reduce manteniendo proporción
    lienzo = Image.new("L", (28, 28), 0) # Lienzo negro de 28x28
    lienzo.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2)) # Centra el dígito
    x = np.array(lienzo, dtype="float32") / 255.0 # Normaliza entre 0 y 1
    return x.reshape(1, 28, 28, 1) # Forma que espera la CNN

@app.route("/") # Página principal
def inicio():
    return render_template("index.html") # Muestra el formulario

@app.route("/predecir", methods=["POST"]) # Ruta que recibe el dibujo
def predecir():
    x = preparar(request.json["imagen"]) # Prepara la imagen
    if x is None: # Si el recuadro está vacío
        return jsonify({"error": "Dibuja un número primero"}) # Avisa al usuario
    prob = modelo.predict(x, verbose=0)[0] # Probabilidades de cada dígito
    return jsonify({"digito": int(np.argmax(prob)), "certeza": round(float(prob.max()) * 100, 1), "accuracy": ACCURACY}) # Respuesta

if __name__ == "__main__": # Para ejecutar de forma local
    app.run(debug=True) # Inicia el servidor