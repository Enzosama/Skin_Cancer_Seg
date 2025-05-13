from flask import Flask, render_template, request, jsonify
import os
import base64
import time
import random
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from graph import compiled_graph
import asyncio

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_predictions(image_path):    
    classes = {
        4: ('nv', 'melanocytic nevi'), 
        6: ('mel', 'melanoma'), 
        2: ('bkl', 'benign keratosis-like lesions'), 
        1: ('bcc', 'basal cell carcinoma'), 
        5: ('vasc', 'pyogenic granulomas and hemorrhage'), 
        0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'), 
        3: ('df', 'dermatofibroma')
    }
    model_path = r'static/best_model.keras'
    
    # Check if model exists
    if not os.path.exists(model_path):
        return [{"label": f"Error: Model not found at {model_path}", "confidence": 0.0}]
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return [{"label": f"Error loading model: {str(e)}", "confidence": 0.0}]
    
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        return [{"label": "Error: Could not read image", "confidence": 0.0}]
    
    # Resize for model input
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    try:
        result = model.predict(np.expand_dims(img, axis=0))
    except Exception as e:
        return [{"label": f"Error during prediction: {str(e)}", "confidence": 0.0}]
    
    # Process the prediction results
    predictions = []
    # If the model output is already normalized (sums to 1), skip this step
    if np.sum(result[0]) > 1.1:  # Check if values need normalization
        from tensorflow.keras.activations import softmax
        result = softmax(result).numpy()
    # Sort results by confidence (descending)
    sorted_indices = np.argsort(result[0])[::-1]
    for i in range(min(5, len(sorted_indices))):
        class_ind = int(sorted_indices[i])
        # Ensure class_ind is a valid key in the classes dictionary
        if class_ind not in classes:
            continue
            
        confidence = float(result[0][class_ind])
        class_code, class_name = classes[class_ind]
        confidence_pct = min(confidence, 100.0)
        predictions.append({
            "label": f"{class_name} ({class_code})",
            "confidence": confidence_pct
        })
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Get predictions from AI model
        predictions = get_predictions(file_path)
        
        return jsonify({
            'success': True,
            'image_url': f'/static/uploads/{filename}',
            'predictions': predictions
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    # Chuẩn bị state đầu vào cho pipeline
    state = {
        "query": user_message,
        "semantic_result": "",
        "rag_result": "",
        "cot_rag_result": "",
        "final_result": "",
        "route": "",
    }
    # Gọi pipeline xử lý bất đồng bộ
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(compiled_graph.ainvoke(state))
        loop.close()
        response_text = result.get("final_result", "Không có kết quả.")
    except Exception as e:
        response_text = f"Lỗi xử lý: {str(e)}"
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)