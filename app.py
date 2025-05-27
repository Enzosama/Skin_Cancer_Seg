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

# Thử import graph module, nếu không có thì đặt GRAPH_AVAILABLE = False
try:
    from graph import compiled_graph
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    compiled_graph = None

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
                # Tạo event loop mới nếu không có sẵn
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
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
                    timeout=90.0  # 90 giây timeout
                ))
                
                # Đóng loop nếu chúng ta đã tạo mới
                if loop != asyncio.get_event_loop():
                    loop.close()
                
                # Lấy kết quả từ graph.py
                response_text = result.get("final_result", "Không có kết quả.")
                
                # Đảm bảo response_text là string
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                
                # Kiểm tra nếu response_text chứa template instructions
                template_instructions = [
                    "Provide a comprehensive definition.", 
                    "Provide a clear process explanation.", 
                    "Explain the causal relationship.", 
                    "Provide treatment recommendations.", 
                    "Describe the symptoms and their significance.", 
                    "Provide diagnostic guidance.", 
                    "Provide a comprehensive response."
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
    # Kiểm tra trạng thái của graph khi khởi động
    if not GRAPH_AVAILABLE or compiled_graph is None:
        print("[WARNING] Graph không khả dụng hoặc chưa được biên dịch. Chức năng chat sẽ bị vô hiệu hóa.")
        print(f"GRAPH_AVAILABLE = {GRAPH_AVAILABLE}")
        print(f"compiled_graph = {'Có' if compiled_graph else 'Không'}")
    else:
        print("[INFO] Graph đã sẵn sàng. Chức năng chat đã được kích hoạt.")
    
    app.run(host="0.0.0.0", port=5001, debug=True)