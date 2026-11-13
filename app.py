"""
Flask Web Application for Pneumonia Detection
Integrates the trained ResNet50 model with a web interface
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy import ndimage
import io
import base64
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for cloud deployment
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# Model Configuration
# ============================================================================

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (224, 224)
    CLASSES = ["NORMAL", "PNEUMONIA"]
    MODEL_PATH = "best_pneumonia_model.pth"

# ============================================================================
# Model Architecture
# ============================================================================

class PneumoniaDetectionModel(nn.Module):
    """Transfer learning model using ResNet50 backbone."""
    
    def __init__(self, pretrained=True):
        super(PneumoniaDetectionModel, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        
        self.fc = nn.Sequential(
            nn.Linear(resnet50.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ============================================================================
# Global Model Instance
# ============================================================================

model = None
device = Config.DEVICE
model_loaded = False

def load_model():
    """Load the trained model."""
    global model, model_loaded
    
    if model_loaded:
        return True
    
    try:
        model = PneumoniaDetectionModel(pretrained=False).to(device)
        
        # Try to load weights if they exist
        if os.path.exists(Config.MODEL_PATH):
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
            model_loaded = True
            model.eval()
            print(f"✓ Model loaded successfully from {Config.MODEL_PATH}")
            return True
        else:
            print(f"⚠ Model weights not found at {Config.MODEL_PATH}")
            print("Using untrained model for demonstration only")
            model_loaded = True
            model.eval()
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

# Initialize and load model on server startup (required for production WSGI/Gunicorn servers)
print("Initializing model...")
load_model()

# ============================================================================
# Prediction Functions
# ============================================================================

def get_transform():
    """Get image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def validate_chest_xray(image):
    """
    Comprehensive validation to check if the uploaded image is a valid chest/lung X-ray.
    Uses multiple characteristics specific to medical X-rays:
    - Grayscale nature (not color photography)
    - Appropriate contrast and intensity levels
    - Characteristic histogram distribution
    - Edge patterns typical of X-rays
    - Dimension ratio (chest X-rays are typically taller than wide or nearly square)
    """
    try:
        # Convert image to grayscale
        img_gray = np.array(image.convert('L'))
        height, width = img_gray.shape
        
        # ===== Check 1: Image dimensions =====
        # X-rays typically have reasonable minimum dimensions
        if height < 50 or width < 50:
            return False
        
        # ===== Check 2: Aspect ratio =====
        # Most chest X-rays are taller than wide (portrait orientation) 
        # or nearly square, but definitely not extremely wide rectangles
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 1.5 or aspect_ratio < 0.3:
            # Too wide or too tall - unlikely to be an X-ray
            return False
        
        # ===== Check 3: Intensity statistics =====
        mean_intensity = np.mean(img_gray)
        std_intensity = np.std(img_gray)
        
        # X-rays have moderate intensity with good contrast
        # Too dark (< 20) or too bright (> 230) indicates non-medical image
        if mean_intensity < 20 or mean_intensity > 230:
            return False
        
        # X-rays must have sufficient contrast (std > 12)
        # Solid colors or photos have low contrast
        if std_intensity < 12:
            return False
        
        # ===== Check 4: Histogram distribution =====
        # X-rays typically have a bimodal or multi-modal histogram
        # (light areas: background/bones, dark areas: lungs/soft tissue)
        histogram, _ = np.histogram(img_gray, bins=256, range=(0, 256))
        
        # Count peaks in histogram (modes)
        # X-rays typically have 2-5 distinct peaks
        peaks = 0
        for i in range(1, 255):
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                peaks += 1
        
        # If there are too few peaks, it's likely not an X-ray (e.g., solid color photo)
        if peaks < 2:
            return False
        
        # ===== Check 5: Edge detection (Sobel) =====
        # X-rays have characteristic edges from anatomical structures
        # Calculate gradients to detect edges
        sx = ndimage.sobel(img_gray, axis=0)
        sy = ndimage.sobel(img_gray, axis=1)
        edges = np.sqrt(sx**2 + sy**2)
        
        # X-rays should have moderate edge content
        edge_mean = np.mean(edges)
        
        # If edge_mean is too low, image is likely a uniform photo
        # If edge_mean is too high, it might be noise
        if edge_mean < 5 or edge_mean > 150:
            return False
        
        # ===== Check 6: High frequency content =====
        # X-rays have moderate high-frequency content (details)
        # Use Laplacian to check for high-frequency components
        laplacian = ndimage.laplace(img_gray)
        high_freq = np.std(laplacian)
        
        if high_freq < 1:
            return False
        
        # ===== Check 7: Quantile analysis =====
        # X-rays use a broad range of intensity values
        p25 = np.percentile(img_gray, 25)
        p75 = np.percentile(img_gray, 75)
        percentile_range = p75 - p25
        
        # Good X-rays have diverse intensity distribution
        if percentile_range < 30:
            return False
        
        # If all checks pass, consider it a valid X-ray
        return True
    
    except Exception as e:
        # If any error occurs during validation, reject the image
        return False

def predict_image(image_bytes):
    """
    Make prediction on image bytes.
    First validates that the image is a chest/lung X-ray before processing.
    """
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # ===== STEP 1: Validate if image is a chest X-ray =====
        is_valid_xray = validate_chest_xray(image)
        if not is_valid_xray:
            return {
                'success': False,
                'error': 'Error: Please upload a valid chest or lung X-ray image only.'
            }
        
        # ===== STEP 2: Preprocess and prepare for model =====
        transform = get_transform()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # ===== STEP 3: Make prediction =====
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        probs = probabilities.cpu().numpy()[0]
        confidence = float(max(probs))
        
        # ===== STEP 4: Apply confidence threshold =====
        # Model must be confident that this is a valid medical image and prediction
        # If confidence is below threshold, the image may not be a proper X-ray
        CONFIDENCE_THRESHOLD = 0.55
        
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                'success': False,
                'error': 'Error: Please upload a valid chest or lung X-ray image only.'
            }
        
        # ===== STEP 5: Return prediction results =====
        return {
            'success': True,
            'prediction': Config.CLASSES[predicted_class],
            'class_index': predicted_class,
            'normal_confidence': float(probs[0]),
            'pneumonia_confidence': float(probs[1]),
            'confidence': confidence
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': 'Error: Please upload a valid chest or lung X-ray image only.'
        }

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         device=str(device))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'success': False, 'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'})
        
        # Read file
        file_content = file.read()
        
        # Make prediction
        result = predict_image(file_content)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(device),
        'classes': Config.CLASSES
    })

@app.route('/model-info')
def model_info():
    """Get model information."""
    return jsonify({
        'model': 'ResNet50 Transfer Learning',
        'classes': Config.CLASSES,
        'input_size': Config.IMG_SIZE,
        'device': str(device),
        'model_loaded': model_loaded,
        'model_path': Config.MODEL_PATH
    })

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("PNEUMONIA DETECTION - WEB APPLICATION")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    load_model()
    
    print("\nStarting Flask server...")
    print(f"Device: {device}")
    print(f"Model Loaded: {model_loaded}")
    print(f"Visit: http://localhost:5000")
    print("="*60)
    
    port = int(os.environ.get('PORT', 5000))
    app.run( host='0.0.0.0', port=port)
