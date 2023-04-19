import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS, cross_origin
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['MODEL_PATH'] = 'Saved_models/inceptionV3_model.h5'

# Load the pre-trained model
model = load_model(app.config['MODEL_PATH'])

# Function to check if the uploaded file is valid
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# to show the uploaded file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Home page
@app.route('/')
@cross_origin()
def index():
    return render_template('home.html')

# Predict function
@app.route('/predict', methods=['POST'])
@cross_origin() 
def predict():
    # Get the uploaded file from the request
    file = request.files['image']
    # Check if the file is valid
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Load the image and preprocess it
        image = load_img(file_path, target_size=(75,75))
        test_image = img_to_array(image)
        test_image = test_image.astype('float32') / 255.0
        test_image = np.expand_dims(test_image, axis = 0)
        
        # Make a prediction
        preds = model.predict(test_image)
        # Decode the predictions and get the class name and probability
        class_idx = preds.argmax(axis=-1)[0]
        class_name = cifar10_classes[class_idx]
        prob = preds[0][class_idx]
        # Return the prediction result
        return render_template('prediction.html', filename=filename, class_name=class_name, prob=prob)
    else:
        return 'Invalid file format'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
