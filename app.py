from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import efficientnet.keras as efn
from tensorflow.keras import backend, layers
from tensorflow.keras.models import load_model

app = Flask(__name__)

class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

# Load model with custom FixedDropout layer
model = load_model('model/raw_batik_v2.1_EfficientNetB3_epoch_70.h5', 
                   custom_objects={'FixedDropout': FixedDropout(rate=0.2)})

# Define class names according to what you obtained from the collab result
class_names = [
    'Bali_Barong', 'Bali_Merak','DKI_Ondel_Ondel','JawaBarat_Megamendung',
    'JawaTimur_Pring', 'Kalimantan_Dayak', 'Madura_Mataketeran',
    'Maluku_Pala', 'Papua_Asmat', 'Papua_Cendrawasih', 'Papua_Tifa',
    'Solo_Parang', 'SulawesiSelatan_Lontara', 'SumateraUtara_Boraspati', 'Yogyakarta_Kawung'
]

class_desc = [
    'Motif ini menggambarkan sosok Barong, makhluk mitos dalam budaya Bali yang melambangkan kekuatan dan kesaktian. Biasanya menggunakan warna-warna cerah dan berani, seperti merah, kuning, dan hitam. Batik Barong sering dipakai dalam upacara adat dan ritual keagamaan.',
    'Motif ini menampilkan burung merak, yang melambangkan keindahan, keanggunan, dan keabadian. Batik Merak biasanya menggunakan warna-warna cerah seperti biru, hijau, dan ungu, dengan detail bulu merak yang rumit dan indah. Batik ini sering dipakai dalam acara-acara resmi dan pernikahan.',
    'Motif ini terinspirasi dari boneka Ondel-ondel, ikon budaya Betawi yang khas. Batik Ondel-ondel biasanya menggunakan warna-warna cerah dan ceria, seperti merah, kuning, dan biru, dengan gambar boneka Ondel-ondel yang besar dan mencolok. Batik ini sering dipakai dalam acara-acara budaya dan festival di Jakarta.',
    'Motif ini menggambarkan awan mendung yang menyelimuti langit, melambangkan kekuatan alam dan kesabaran. Batik Megamendung biasanya menggunakan warna-warna biru tua dan hitam, dengan corak awan yang meliuk-liuk. Batik ini sering dipakai dalam acara-acara formal dan resmi.',
    'Motif ini menampilkan pohon bambu, melambangkan kesederhanaan, kekuatan, dan kelenturan. Batik Pring biasanya menggunakan warna-warna hijau dan coklat, dengan corak batang bambu yang ramping dan daun bambu yang menari-nari. Batik ini sering dipakai dalam acara-acara sehari-hari dan kasual.',
    'Motif ini terinspirasi dari budaya suku Dayak di Kalimantan, yang kaya akan corak dan simbol-simbol adat. Batik Dayak biasanya menggunakan warna-warna cerah dan berani, seperti merah, kuning, dan hitam, dengan corak hewan hutan, motif abstrak, dan ukiran khas Dayak. Batik ini sering dipakai dalam upacara adat dan ritual keagamaan.',
    'Motif ini menggambarkan mata kerbau, melambangkan kewaspadaan, kehati-hatian, dan ketekunan. Batik Mataketeran biasanya menggunakan warna-warna coklat dan hitam, dengan corak mata kerbau yang besar dan mencolok. Batik ini sering dipakai dalam acara-acara formal dan resmi.',
    'Motif ini menampilkan buah pala, yang merupakan salah satu rempah-rempah khas Maluku. Batik Pala biasanya menggunakan warna-warna coklat dan putih, dengan corak buah pala yang detail dan realistis. Batik ini sering dipakai sebagai souvenir dan hadiah dari Maluku.',
    'Motif ini terinspirasi dari seni ukir suku Asmat di Papua, yang terkenal dengan corak dan simbol-simbol adat yang unik. Batik Asmat biasanya menggunakan warna-warna coklat dan hitam, dengan corak patung dan ukiran khas Asmat yang rumit dan indah. Batik ini sering dipakai dalam upacara adat dan ritual keagamaan.',
    'Motif ini menggambarkan burung Cendrawasih, yang merupakan ikon Papua yang melambangkan keindahan, kebebasan, dan kebahagiaan. Batik Cendrawasih biasanya menggunakan warna-warna cerah dan ceria, seperti merah, kuning, dan biru, dengan gambar burung Cendrawasih yang menari-nari di antara pepohonan. Batik ini sering dipakai dalam acara-acara budaya dan festival di Papua.',
    'Motif ini menampilkan alat musik tradisional Papua yang disebut Tifa, yang melambangkan semangat, kebersamaan, dan kegembiraan. Batik Tifa biasanya menggunakan warna-warna coklat dan hitam, dengan corak Tifa yang detail dan realistis. Batik ini sering dipakai dalam acara-acara adat dan ritual keagamaan.',
    'Motif ini terkenal dengan corak garis-garis diagonal yang tajam, melambangkan keberanian, kekuatan, dan ketegasan. Batik Parang biasanya menggunakan warna-warna coklat dan hitam, dengan variasi motif parang yang beragam, seperti parang grid, parang klitik, dan parang salira. Batik ini sering dipakai dalam acara-acara formal dan resmi, terutama oleh para pria.',
    'Motif ini terinspirasi dari aksara Lontara, aksara tradisional Bugis yang unik dan indah. Batik Lontara biasanya menggunakan warna-warna coklat dan putih, dengan corak aksara Lontara yang tersusun rapi dan artistik. Batik ini sering dipakai dalam acara-acara adat dan ritual keagamaan, dan juga merupakan salah satu ciri khas budaya Bugis.',
    'Motif ini menggambarkan pohon Boras, yang merupakan pohon khas Sumatera Utara. Batik Boraspat biasanya menggunakan warna-warna coklat dan hitam, dengan corak pohon Boras yang detail dan realistis. Batik ini sering dipakai dalam acara-acara adat dan ritual keagamaan, dan juga merupakan salah satu ciri khas budaya Batak.',
    'Motif ini menampilkan bentuk kawung (kolang-kaling) yang tersusun rapi, melambangkan kesatuan, keseimbangan, dan kesempurnaan. Batik Kawung biasanya menggunakan warna-warna coklat dan putih, dengan variasi motif kawung yang beragam, seperti kawung pecah, kawung sri, dan kawung luncu. Batik ini sering dipakai dalam acara-acara adat dan ritual keagamaan, dan juga merupakan motif batik Yogyakarta yang paling terkenal.'
]

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure request is POST
    if request.method == 'POST':
        # Ensure there is a file part in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # Ensure the file is not empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Ensure the file is an image
        if file and allowed_file(file.filename):
            # Read the image and do preprocessing
            img = Image.open(file)
            img = img.resize((300, 300))  # Resize image
            img_array = np.array(img) / 255.0  # Normalize
            
            # Expand dimensions and add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            # Predict class
            prediction = model.predict(img_array)
            print(prediction)
            predicted_class_index = int(np.argmax(prediction))
            predicted_class_name = class_names[predicted_class_index]
            batik_desc = class_desc[predicted_class_index]
            
            # Provide response with predicted class label
            return jsonify({'class_index': predicted_class_index, 'batikName': predicted_class_name, 'batikDesc': batik_desc})
        else:
            return jsonify({'error': 'Invalid file type'})




if __name__ == '__main__':
    app.run(debug=True)
