document.addEventListener('DOMContentLoaded', function() {
    // Xử lý chuyển tab
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            
            // Xóa active class từ tất cả các tab
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Thêm active class cho tab được chọn
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Xử lý tải lên ảnh
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    
    // Xử lý sự kiện kéo và thả
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('drag-over');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('drag-over');
        });
    });
    
    dropArea.addEventListener('drop', handleDrop);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFiles(files[0]);
        }
    }
    
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFiles(e.target.files[0]);
        }
    });
    
    function handleFiles(file) {
        uploadFile(file);
    }
    
    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayAnalysisResults(data);
            } else {
                alert('Lỗi: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Đã xảy ra lỗi khi tải lên ảnh');
        });
    }
    
    function displayAnalysisResults(data) {
        // Hiển thị panel phân tích
        const analysisPanel = document.getElementById('analysis-panel');
        analysisPanel.style.display = 'block';
        
        // Hiển thị kết quả phân tích
        const predictionResults = document.getElementById('prediction-results');
        predictionResults.innerHTML = '';
        
        data.predictions.forEach(prediction => {
            const li = document.createElement('li');
            li.textContent = `${prediction.label}: ${prediction.confidence.toFixed(2)}%`;
            predictionResults.appendChild(li);
        });
        
        // Hiển thị kết luận
        const conclusionText = document.getElementById('conclusion-text');
        if (data.predictions.length > 0) {
            const topPrediction = data.predictions[0];
            conclusionText.textContent = `Kết quả này cho thấy có sự bất thường trong biểu mô cổ tử cung ở mức độ cao. Điều này nhấn mạnh sự cần thiết phải theo dõi và can thiệp y tế ngay lập tức để ngăn ngừa nguy cơ tiến triển thành ung thư cổ tử cung. Cần tư vấn với các bác sĩ chuyên khoa để có quyết định điều trị phù hợp nhất.`;
        } else {
            conclusionText.textContent = 'Không có kết quả phân tích';
        }
    }
    
    // Tải danh sách ảnh bệnh nhân
    fetch('/api/patient_images')
    .then(response => response.json())
    .then(data => {
        displayPatientImages(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
    
    function displayPatientImages(images) {
        const patientGrid = document.getElementById('patient-images-grid');
        
        images.forEach(image => {
            const card = document.createElement('div');
            card.className = 'patient-card';
            card.innerHTML = `
                <img src="${image.image_url}" alt="Ảnh bệnh nhân" class="patient-image">
                <div class="patient-details">
                    <div class="patient-id">ID: ${image.id}</div>
                    <div class="patient-date">${image.date}</div>
                    <div class="patient-status">${image.status}</div>
                </div>
            `;
            
            card.addEventListener('click', () => {
                window.location.href = '/patient_history';
            });
            
            patientGrid.appendChild(card);
        });
    }
    
    // Xử lý các nút hành động
    document.getElementById('explain-btn').addEventListener('click', () => {
        alert('Chức năng giải thích kết quả phân tích sẽ được triển khai sau');
    });
    
    document.getElementById('compare-btn').addEventListener('click', () => {
        alert('Chức năng so sánh với kết quả trước sẽ được triển khai sau');
    });
    
    document.getElementById('recommend-btn').addEventListener('click', () => {
        alert('Chức năng đề xuất theo dõi sẽ được triển khai sau');
    });
    
    document.getElementById('view-patient-btn').addEventListener('click', () => {
        window.location.href = '/patient_history';
    });
    
    // Xử lý chat
    const chatInput = document.getElementById('chat-message');
    const sendChatBtn = document.getElementById('send-chat');
    
    sendChatBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });
    
    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
            alert('Phản hồi từ trợ lý: ' + data.response);
            chatInput.value = '';
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    // Thêm xử lý cho nút "Upload New Image"
    const newUploadBtn = document.getElementById('newUploadBtn');
    if (newUploadBtn) {
        newUploadBtn.addEventListener('click', () => {
            // Hiển thị lại khu vực upload và ẩn khu vực kết quả
            const uploadArea = document.getElementById('uploadArea');
            const resultsArea = document.getElementById('resultsArea');
            
            if (uploadArea && resultsArea) {
                uploadArea.style.display = 'flex';
                resultsArea.style.display = 'none';
            }
            
            // Reset form nếu cần
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                fileInput.value = '';
            }
        });
    }
});