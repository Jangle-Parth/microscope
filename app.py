import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the exact UNET architecture from your notebook
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), padding_mode='circular', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), padding_mode='circular', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), padding_mode='circular', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), padding_mode='circular', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
        # Down Part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature  # pass to next level
    
        # Up Part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
    
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
    
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
    
        x = self.bottleneck(x)
    
        skip_connections = skip_connections[::-1]  # reverse the connection
            
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
                
        return self.final_conv(x)

# Load model
model = UNET(in_channels=3, out_channels=1)
try:
    model.load_state_dict(torch.load('microscopy_circular.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully")
except FileNotFoundError:
    print("Error: 'microscopy_circular.pth' not found in the current directory")
    print("Please ensure the model file is in the same directory as app.py")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    if len(image_array.shape) == 2:
        # Grayscale image
        image_pil = Image.fromarray((image_array * 255).astype(np.uint8), mode='L')
    else:
        # Color image
        image_pil = Image.fromarray(image_array.astype(np.uint8))
    
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_image(image_path):
    """Process the uploaded image and return predictions"""
    # Image transformation
    transform_input = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    input_tensor = transform_input(image).unsqueeze(0).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process output
    output_n = torch.sigmoid(output)
    output_binary = (output_n > 0.5).float()
    
    # Resize to original size
    output_resized = torch.nn.functional.interpolate(
        output_binary, 
        size=(original_size[1], original_size[0]), 
        mode="nearest"
    )
    
    output_np = output_resized.squeeze().cpu().numpy()
    
    # Calculate contours and area
    binary_image = (output_np * 255).astype(np.uint8)
    _, binary_thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create contour visualization
    contour_image = cv2.cvtColor(binary_thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    
    # Calculate total area
    total_area_pixels = sum(cv2.contourArea(contour) for contour in contours)
    
    # Convert to µm² (assuming 10x10 µm² image size)
    image_size_um = 10 * 10
    image_size_pixels = binary_thresh.shape[0] * binary_thresh.shape[1]
    conversion_factor = image_size_um / image_size_pixels
    total_area_um2 = total_area_pixels * conversion_factor
    
    return {
        'original_image': np.array(image),
        'predicted_mask': output_np,
        'contour_image': contour_image,
        'total_area_um2': total_area_um2,
        'num_contours': len(contours)
    }

# Routes
@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Microscopy Analysis Server is running',
        'model_loaded': model is not None
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_loaded': model is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        # Check if file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process the image
                results = process_image(filepath)
                
                # Convert images to base64
                response_data = {
                    'status': 'success',
                    'original_image': image_to_base64(results['original_image']),
                    'predicted_mask': image_to_base64(results['predicted_mask']),
                    'contour_image': image_to_base64(results['contour_image']),
                    'total_area_um2': float(results['total_area_um2']),
                    'num_contours': results['num_contours']
                }
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return jsonify(response_data)
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Invalid file type. Please upload PNG or JPEG images'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/analyze_with_ground_truth', methods=['POST'])
def analyze_with_ground_truth():
    """Endpoint for when ground truth mask is available"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        mask_file = request.files.get('mask', None)
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if image_file and allowed_file(image_file.filename):
            # Save files
            image_filename = secure_filename(image_file.filename)
            image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            image_file.save(image_filepath)
            
            mask_filepath = None
            if mask_file and allowed_file(mask_file.filename):
                mask_filename = secure_filename(mask_file.filename)
                mask_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
                mask_file.save(mask_filepath)
            
            try:
                # Process image
                results = process_image(image_filepath)
                
                response_data = {
                    'status': 'success',
                    'original_image': image_to_base64(results['original_image']),
                    'predicted_mask': image_to_base64(results['predicted_mask']),
                    'contour_image': image_to_base64(results['contour_image']),
                    'total_area_um2': float(results['total_area_um2']),
                    'num_contours': results['num_contours']
                }
                
                # If ground truth mask provided, include it and calculate IoU
                if mask_filepath:
                    mask = Image.open(mask_filepath).convert('L')
                    mask_np = np.array(mask) / 255.0
                    
                    # Resize mask to match predicted mask
                    mask_resized = cv2.resize(mask_np, 
                                            (results['predicted_mask'].shape[1], 
                                             results['predicted_mask'].shape[0]))
                    
                    # Calculate IoU
                    pred_binary = (results['predicted_mask'] > 0.5).astype(float)
                    intersection = np.sum(pred_binary * mask_resized)
                    union = np.sum(pred_binary) + np.sum(mask_resized) - intersection
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    
                    response_data['ground_truth_mask'] = image_to_base64(mask_resized)
                    response_data['iou_score'] = float(iou)
                    
                    # Clean up mask file
                    os.remove(mask_filepath)
                
                # Clean up image file
                os.remove(image_filepath)
                
                return jsonify(response_data)
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(image_filepath):
                    os.remove(image_filepath)
                if mask_filepath and os.path.exists(mask_filepath):
                    os.remove(mask_filepath)
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)