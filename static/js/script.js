document.addEventListener('DOMContentLoaded', function() {
    // Các phần tử DOM
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadArea = document.getElementById('uploadArea');
    const resultsArea = document.getElementById('resultsArea');
    const newUploadBtn = document.getElementById('newUploadBtn');
    const previewImage = document.getElementById('previewImage');
    const predictionsList = document.getElementById('predictionsList');
    const conclusionText = document.getElementById('conclusionText');
    const imageId = document.getElementById('imageId');
    const imageDate = document.getElementById('imageDate');
    const imageStatus = document.getElementById('imageStatus');
    const patientImagesList = document.getElementById('patientImagesList');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatMessages = document.getElementById('chatMessages');

    // Đếm số lượng ảnh
    let imageCount = 0; // Đặt ở đầu file hoặc ngoài hàm uploadFile

    // Xử lý sự kiện click nút Browse
    if (browseBtn) {
        browseBtn.addEventListener('click', function() {
            fileInput.click();
        });
    }
    
    // Xử lý sự kiện khi chọn file
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                uploadFile(fileInput.files[0]);
            }
        });
    }
    
    // Xử lý sự kiện kéo thả
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                uploadFile(e.dataTransfer.files[0]);
            }
        });
    }
    
    // Xử lý sự kiện nút Upload New Image
    if (newUploadBtn) {
        newUploadBtn.addEventListener('click', function() {
            // Hiển thị lại khu vực upload và ẩn khu vực kết quả
            uploadArea.style.display = 'flex';
            resultsArea.style.display = 'none';
            
            // Reset form
            fileInput.value = '';
        });
    }
    
    // Xử lý sự kiện gửi tin nhắn chat
    if (sendBtn && chatInput) {
        sendBtn.addEventListener('click', sendChatMessage);
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }
    
    // Hàm gửi tin nhắn chat
    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (message) {
            // Hiển thị tin nhắn của người dùng
            appendChatMessage(message, 'user');
            
            // Gửi tin nhắn đến server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                // Hiển thị phản hồi từ bot
                appendChatMessage(data.response, 'bot');
            })
            .catch(error => {
                console.error('Error:', error);
                appendChatMessage('Xin lỗi, đã xảy ra lỗi khi xử lý tin nhắn của bạn.', 'bot');
            });
            
            // Xóa nội dung input
            chatInput.value = '';
        }
    }
    
    // Hàm thêm tin nhắn vào khung chat
    function appendChatMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = message;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Cuộn xuống tin nhắn mới nhất
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Hàm upload file
    function uploadFile(file) {
        // Kiểm tra loại file
        const fileType = file.type.split('/')[1];
        const allowedTypes = ['jpeg', 'jpg', 'png', 'tiff', 'gif'];
        
        if (!allowedTypes.includes(fileType.toLowerCase())) {
            alert('Chỉ chấp nhận file ảnh (JPG, PNG, TIFF, GIF)');
            return;
        }
        
        // Tạo FormData
        const formData = new FormData();
        formData.append('file', file);
              
        // Gửi file lên server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Cập nhật số lượng ảnh
                imageCount++;
                document.querySelector('.image-count').textContent = `${imageCount} ảnh`;
                
                // Hiển thị ảnh đã tải lên
                previewImage.src = data.image_url;
                
                // Gán ID theo thứ tự upload
                imageId.textContent = imageCount;
                
                // Hiển thị ngày hiện tại
                const today = new Date();
                const dateStr = today.toLocaleDateString('vi-VN');
                imageDate.textContent = dateStr;
                
                // Cập nhật trạng thái
                imageStatus.textContent = 'Đã phân tích';
                
                // Hiển thị kết quả dự đoán
                predictionsList.innerHTML = '';
                data.predictions.forEach(pred => {
                    const li = document.createElement('li');
                    const confidence = (pred.confidence * 100).toFixed(2);
                    li.innerHTML = `<span class="prediction-label">${pred.label}:</span> <span class="prediction-value">${confidence}%</span>`;
                    predictionsList.appendChild(li);
                });
                
                // Tạo kết luận dựa trên kết quả dự đoán cao nhất
                if (data.predictions.length > 0) {
                    const topPrediction = data.predictions[0];
                    let conclusion = '';
                    
                    if (topPrediction.label.includes('melanoma')) {
                        conclusion = 'Kết quả này cho thấy có sự bất thường trong biểu mô cổ tử cung ở mức độ cao. Điều này nhấn mạnh sự cần thiết phải theo dõi và can thiệp y tế ngay lập tức để ngăn ngừa nguy cơ tiến triển thành ung thư cổ tử cung. Cần tư vấn với các bác sĩ chuyên khoa để có quyết định điều trị phù hợp nhất.';
                    } else {
                        conclusion = 'Kết quả phân tích cho thấy không có dấu hiệu bất thường nghiêm trọng. Tuy nhiên, vẫn nên tiếp tục theo dõi định kỳ theo khuyến nghị của bác sĩ.';
                    }
                    
                    conclusionText.textContent = conclusion;
                }
                
                // Thêm ảnh vào danh sách ảnh bệnh nhân
                const thumbnailDiv = document.createElement('div');
                thumbnailDiv.className = 'patient-image-item';
                thumbnailDiv.innerHTML = `
                    <img src="${data.image_url}" alt="Patient image">
                    <div class="image-item-info">
                        <div>ID: ${imageCount}</div>
                        <div>${dateStr}</div>
                        <div class="status">Đã phân tích</div>
                    </div>
                `;
                patientImagesList.appendChild(thumbnailDiv);
                
                // Ẩn khu vực upload và hiển thị khu vực kết quả
                uploadArea.style.display = 'none';
                resultsArea.style.display = 'block';
                
                // Thêm tin nhắn chào mừng vào chat
                appendChatMessage('Tôi đã phân tích xong ảnh của bạn. Bạn có thể hỏi tôi về kết quả phân tích.', 'bot');
            } else {
                // Hiển thị lỗi
                alert('Lỗi: ' + data.error);
                // Khôi phục giao diện upload
                resetUploadArea();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Đã xảy ra lỗi khi tải lên ảnh');
            // Khôi phục giao diện upload
            resetUploadArea();
        });
    }
    
    // Hàm reset khu vực upload
    function resetUploadArea() {
        uploadArea.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-cloud-upload-alt"></i>
            </div>
            <h3>Upload Images</h3>
            <p>Drag and drop files here or click to browse. Supported formats: JPG, PNG, TIFF</p>
            <button class="upload-btn" id="browseBtn">Upload Images</button>
        `;
        
        // Gán lại sự kiện cho nút browse
        document.getElementById('browseBtn').addEventListener('click', function() {
            fileInput.click();
        });
    }

// Thêm xử lý sự kiện cho nút "Upload New Image"
document.addEventListener('DOMContentLoaded', function() {
    // Lấy tham chiếu đến nút "Upload New Image"
    const newUploadBtn = document.getElementById('newUploadBtn');
    
    // Thêm sự kiện click cho nút
    if (newUploadBtn) {
        newUploadBtn.addEventListener('click', function() {
            // Ẩn khu vực kết quả
            resultsArea.style.display = 'none';
            
            // Hiển thị lại khu vực tải lên
            uploadArea.style.display = 'flex';
            
            // Đặt lại nội dung khu vực tải lên
            resetUploadArea();
            
            // Xóa ảnh hiện tại trong preview
            previewImage.src = '';
            
            // Đặt lại các thông tin
            imageId.textContent = '--';
            imageDate.textContent = '--';
            imageStatus.textContent = '--';
            
            // Xóa danh sách dự đoán
            predictionsList.innerHTML = '';
            
            // Đặt lại kết luận
            conclusionText.textContent = 'Waiting for analysis results...';
        });
    }
});
});