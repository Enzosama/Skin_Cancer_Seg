from flask import Flask, render_template, request, jsonify
import os
import base64
import time
import random
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import asyncio
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from optimized_rag import faiss_index

try:
    from graph import compiled_graph
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    compiled_graph = None

# Function to preload FAISS index
def faiss_index():
    try:
        from optimized_rag import create_optimized_rag
        from rag.llm import hugging_face_embedding
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Initialize RAG with FAISS index
        rag = create_optimized_rag(
            working_dir="./rag_cache",
            llm_model_func=None,
            embedding_func=hugging_face_embedding,
            enable_cache=True,
            use_sqlite=True
        )
        print("[INFO] FAISS index preloaded successfully")
        return rag
    except Exception as e:
        print(f"[ERROR] Failed to preload FAISS index: {e}")
        return None

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
    # Load model if not already loaded
    model_path = 'static/complete_model.h5'
    if not hasattr(get_predictions, 'model'):
        try:
            print("[INFO] Attempting to load model...")
            
            # Method 1: Custom InputLayer class to handle batch_shape
            class CompatibleInputLayer(tf.keras.layers.InputLayer):
                def __init__(self, **kwargs):
                    # Convert batch_shape to input_shape for compatibility
                    if 'batch_shape' in kwargs:
                        batch_shape = kwargs.pop('batch_shape')
                        if batch_shape and len(batch_shape) > 1:
                            kwargs['input_shape'] = batch_shape[1:]
                    super().__init__(**kwargs)
            
            custom_objects = {
                'InputLayer': CompatibleInputLayer,
            }
            
            try:
                get_predictions.model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
                print("[INFO] Model loaded successfully with compatible InputLayer")
            except Exception as e1:
                print(f"[WARNING] Compatible InputLayer method failed: {e1}")
                
                # Method 2: Try with TensorFlow compatibility mode
                try:
                    import tensorflow.compat.v1 as tf_v1
                    tf_v1.disable_v2_behavior()
                    get_predictions.model = tf.keras.models.load_model(
                        model_path, 
                        compile=False
                    )
                    print("[INFO] Model loaded with TF v1 compatibility mode")
                except Exception as e2:
                    print(f"[WARNING] TF v1 compatibility failed: {e2}")
                    
                    # Method 3: Try loading with older Keras format
                    try:
                        # Force use of legacy format
                        get_predictions.model = tf.keras.models.load_model(
                            model_path, 
                            compile=False,
                            options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                        )
                        print("[INFO] Model loaded with legacy format options")
                    except Exception as e3:
                        print(f"[WARNING] Legacy format failed: {e3}")
                        
                        # Method 4: Manual model reconstruction
                        get_predictions.model = reconstruct_model_from_weights(model_path)
                        if get_predictions.model is None:
                            raise Exception(f"All loading methods failed. You need to re-export your model with current TensorFlow version.")
                        else:
                            print("[INFO] Model reconstructed from weights")
                    
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return [{
                'label': 'Model Load Error',
                'confidence': 0.0,
                'error': f'Model loading failed: {str(e)}. Please re-export your model with TensorFlow {tf.__version__}'
            }]
    
    # Define class indices and descriptions
    class_indices = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }
    class_descriptions = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma', 
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesions'
    }
    
    try:
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = get_predictions.model.predict(img_array, verbose=0)
        if np.max(predictions[0]) > 1.0:
            predictions = tf.nn.softmax(predictions).numpy()
            print("[INFO] Applied softmax to predictions")
        
        # Process results
        results = []
        for i, pred in enumerate(predictions[0]):
            confidence = float(pred) 
            class_name = list(class_indices.keys())[i]
            results.append({
                'label': class_descriptions.get(class_name, class_name),
                'class_name': class_name,
                'confidence': round(confidence, 2)
            })
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"[INFO] Top prediction: {results[0]['label']} ({results[0]['confidence']:.2f}%)")
        return results[:5]
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
        return [{
            'label': 'Prediction Error',
            'confidence': 0.0,
            'error': f'Prediction failed: {str(e)}'
        }]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Ensure upload directory exists
            upload_dir = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            predictions = get_predictions(file_path)
            return jsonify({
                'success': True,
                'image_url': f'/static/uploads/{filename}',
                'predictions': predictions
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400            
    except Exception as e:
        print(f"[ERROR] Upload handling failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload processing failed: {str(e)}'}), 500

# Model reconstruction function for incompatible models
def reconstruct_model_from_weights(model_path):
    try:
        # Try to extract weights from the H5 file
        import h5py

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 classes for skin lesions
        ])
        
        # Try to load weights (this might fail if architecture doesn't match)
        try:
            model.load_weights(model_path)
            print("[INFO] Weights loaded into reconstructed model")
            return model
        except:
            print("[WARNING] Could not load weights into simple architecture")
            
            # Try with a more complex architecture (EfficientNet-like)
            try:
                base_model = tf.keras.applications.EfficientNetB0(
                    weights=None,  # Don't load ImageNet weights
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
                
                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                predictions = tf.keras.layers.Dense(7, activation='softmax')(x)
                
                model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                model.load_weights(model_path)
                print("[INFO] Weights loaded into EfficientNet architecture")
                return model
            except Exception as e:
                print(f"[WARNING] EfficientNet reconstruction failed: {e}")
                return None
                
    except Exception as e:
        print(f"[ERROR] Model reconstruction failed: {e}")
        return None


@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Kiểm tra dữ liệu đầu vào
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        # Gọi pipeline xử lý bất đồng bộ
        if not GRAPH_AVAILABLE or compiled_graph is None:
            print(f"[ERROR] Graph không khả dụng: GRAPH_AVAILABLE={GRAPH_AVAILABLE}, compiled_graph={'Có' if compiled_graph else 'Không'}")
            response_text = "Xin lỗi, hệ thống RAG hiện tại không khả dụng. Vui lòng thử lại sau hoặc liên hệ quản trị viên."
        else:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Sử dụng compiled_graph để xử lý truy vấn
                state = {
                    "query": user_message,
                    "semantic_result": "",
                    "rag_result": "",
                    "cot_rag_result": "",
                    "final_result": "",
                    "route": "",
                    "engine": "google"
                }
                
                # Gọi compiled_graph để xử lý truy vấn
                result = loop.run_until_complete(asyncio.wait_for(
                    compiled_graph.ainvoke(state),
                    timeout=600.0  # 600 giây timeout
                ))
                
                # Đóng loop nếu chúng ta đã tạo mới
                if not loop.is_running():
                    loop.close()
                
                # Lấy kết quả từ graph.py
                response_text = result.get("final_result", "Không có kết quả.")
                
                # Đảm bảo response_text là string
                if not isinstance(response_text, str):
                    response_text = str(response_text)
            
                template_instructions = [
                    "Provide a comprehensive definition.", 
                    "Provide a clear process explanation.", 
                    "Explain the causal relationship.", 
                    "Provide treatment recommendations.", 
                    "Describe the symptoms and their significance.", 
                    "Provide diagnostic guidance.", 
                    "Provide a comprehensive response.",
                    "Based on the medical context and available information, provide the specific definition requested.",
                    "Explain the specific process or mechanism step by step.",
                    "Clearly explain the causal relationship and underlying mechanisms.",
                    "Present the specific treatment options and recommendations.",
                    "List and explain the specific symptoms and their clinical significance.",
                    "Present the diagnostic criteria and assessment methods.",
                    "Based on the analysis above, provide a specific answer to the question.",
                    "Let's think step by step:",
                    "Step 1:", "Step 2:", "Step 3:", "Step 4:", "Step 5:"
                ]
                
                # Loại bỏ template instructions nếu có
                for instr in template_instructions:
                    if response_text.startswith(instr):
                        response_text = response_text[len(instr):].strip()
                        break
                    
                # Kiểm tra và loại bỏ các template instructions khác có thể xuất hiện trong văn bản
                if any(instr in response_text for instr in template_instructions):
                    for instr in template_instructions:
                        if instr in response_text:
                            response_text = response_text.replace(instr, "").strip()                
                if response_text.strip().lower() == "final answer:" or response_text.strip() == "":
                    if result.get("cot_rag_result") and isinstance(result.get("cot_rag_result"), str) and result.get("cot_rag_result").strip() != "":
                        response_text = result.get("cot_rag_result")
                    elif result.get("rag_result") and isinstance(result.get("rag_result"), str) and result.get("rag_result").strip() != "":
                        response_text = result.get("rag_result")
                    else:
                        response_text = "Không thể tạo câu trả lời. Vui lòng thử lại với câu hỏi khác."
                    
            except asyncio.TimeoutError:
                print(f"[ERROR] Timeout khi xử lý câu hỏi: {user_message}")
                response_text = "Xin lỗi, hệ thống đang bận. Vui lòng thử lại sau."
            except ImportError as e:
                print(f"[ERROR] Lỗi import module: {str(e)}")
                response_text = f"Lỗi import module: {str(e)}. Vui lòng kiểm tra các dependencies."
            except Exception as e:
                print(f"[ERROR] Lỗi xử lý: {str(e)}")
                response_text = f"Lỗi xử lý: {str(e)}"
            
        return jsonify({'response': response_text})
        
    except Exception as e:
        if 'response_text' in locals() and response_text:
            return jsonify({'response': response_text})
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint kiểm tra sức khỏe của hệ thống"""
    health_status = {
        'status': 'ok',
        'timestamp': time.time(),
        'components': {
            'graph': {
                'available': GRAPH_AVAILABLE,
                'compiled': compiled_graph is not None
            },
            'tensorflow': {
                'available': 'tf' in globals() or 'tensorflow' in globals()
            }
        }
    }
    # Kiểm tra xem tất cả các thành phần có hoạt động không
    all_components_ok = all([
        health_status['components']['graph']['available'],
        health_status['components']['graph']['compiled'],
        health_status['components']['tensorflow']['available']
    ])
    
    if not all_components_ok:
        health_status['status'] = 'degraded'
    
    return jsonify(health_status)

if __name__ == '__main__':
    # Preload FAISS index at startup
    _preloaded_rag = faiss_index()
    # Kiểm tra trạng thái của graph khi khởi động
    if not GRAPH_AVAILABLE or compiled_graph is None:
        print("[WARNING] Graph không khả dụng hoặc chưa được biên dịch. Chức năng chat sẽ bị vô hiệu hóa.")
        print(f"GRAPH_AVAILABLE = {GRAPH_AVAILABLE}")
        print(f"compiled_graph = {'Có' if compiled_graph else 'Không'}")
    else:
        print("[INFO] Graph đã sẵn sàng. Chức năng chat đã được kích hoạt.")
    app.run(host="0.0.0.0", port=7860, debug=True)