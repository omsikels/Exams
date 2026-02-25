"""
Flask Web App for Video Emotion Detection
Upload videos and get emotion analysis results
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import json
from werkzeug.utils import secure_filename
import threading
from datetime import datetime

# ============================================================================
# Model Architecture (same as your original)
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.conv = nn.Conv2d(in_c, groups, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(groups)
        self.prelu = nn.PReLU(groups)
        self.conv_dw = nn.Conv2d(groups, groups, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn_dw = nn.BatchNorm2d(groups)
        self.prelu_dw = nn.PReLU(groups)
        self.project = nn.Conv2d(groups, out_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn_project = nn.BatchNorm2d(out_c)
        self.residual = residual
    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.prelu(self.bn(self.conv(x)))
        x = self.prelu_dw(self.bn_dw(self.conv_dw(x)))
        x = self.bn_project(self.project(x))
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class MobileFaceNet(nn.Module):
    def __init__(self):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = DepthWise(64, 64, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        return out

class FAN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_landmarks = num_landmarks
        self.backbone = MobileFaceNet()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.heatmap = nn.Conv2d(128, num_landmarks, kernel_size=1)
    def forward(self, x):
        features = self.backbone(x)
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        heatmaps = self.heatmap(x)
        batch_size = heatmaps.size(0)
        h, w = heatmaps.size(2), heatmaps.size(3)
        heatmaps_flat = heatmaps.reshape(batch_size, self.num_landmarks, -1)
        softmax_maps = F.softmax(heatmaps_flat, dim=2)
        grid_x = torch.linspace(0, w-1, w, device=x.device)
        grid_y = torch.linspace(0, h-1, h, device=x.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).reshape(2, -1)
        landmarks = torch.matmul(softmax_maps, grid.T.unsqueeze(0))
        return features, landmarks, heatmaps

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class KeypointFeatureNetwork(nn.Module):
    def __init__(self, num_landmarks=68, feature_dim=2048):
        super(KeypointFeatureNetwork, self).__init__()
        self.num_landmarks = num_landmarks
        self.feature_dim = feature_dim
    def forward(self, feature_map, landmarks):
        B, C, H, W = feature_map.shape
        landmarks_norm = landmarks.clone()
        landmarks_norm[:, :, 0] = 2.0 * landmarks[:, :, 0] / (W - 1) - 1.0
        landmarks_norm[:, :, 1] = 2.0 * landmarks[:, :, 1] / (H - 1) - 1.0
        grid = landmarks_norm.unsqueeze(2)
        keypoint_features = F.grid_sample(feature_map, grid, mode='bilinear', 
                                         padding_mode='border', align_corners=True)
        keypoint_features = keypoint_features.squeeze(-1).transpose(1, 2)
        return keypoint_features

class RKFA(nn.Module):
    def __init__(self, feature_dim=2048):
        super(RKFA, self).__init__()
        self.feature_dim = feature_dim
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )
    def forward(self, keypoint_features, representative_idx=30):
        rep_feature = keypoint_features[:, representative_idx:representative_idx+1, :]
        attention_scores = self.attention(keypoint_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_features = keypoint_features * attention_weights
        return attended_features

class LandmarkPerturbation(nn.Module):
    def __init__(self, perturbation_scale=0.10):
        super(LandmarkPerturbation, self).__init__()
        self.perturbation_scale = perturbation_scale
    def forward(self, landmarks, feature_map_size):
        return landmarks

class NKF(nn.Module):
    def __init__(self, num_classes=7, num_landmarks=68, use_rkfa=True, perturbation_scale=0.10):
        super(NKF, self).__init__()
        self.fan = FAN(num_landmarks=num_landmarks)
        for param in self.fan.parameters():
            param.requires_grad = False
        self.backbone = ResNet50()
        self.kf_net = KeypointFeatureNetwork(num_landmarks=num_landmarks, feature_dim=2048)
        self.use_rkfa = use_rkfa
        if use_rkfa:
            self.rkfa = RKFA(feature_dim=2048)
        self.perturbation = LandmarkPerturbation(perturbation_scale=perturbation_scale)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_kf = nn.Sequential(
            nn.Linear(num_landmarks * 2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.fc_global = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.fc_final = nn.Linear(512 * 2, num_classes)
        self.dropout_global = nn.Dropout(0.4)
        self.dropout_kf = nn.Dropout(0.4)
        self.dropout_final = nn.Dropout(0.6)
    def forward(self, x):
        B = x.size(0)
        with torch.no_grad():
            self.fan.eval()
            _, landmarks, _ = self.fan(x)
        global_features = self.backbone(x)
        H, W = global_features.size(2), global_features.size(3)
        landmarks_perturbed = self.perturbation(landmarks, (H, W))
        keypoint_features = self.kf_net(global_features, landmarks_perturbed)
        if self.use_rkfa:
            keypoint_features = self.rkfa(keypoint_features)
        global_pooled = self.global_pool(global_features).view(B, -1)
        global_pooled = self.fc_global(global_pooled)
        global_pooled = self.dropout_global(global_pooled)
        kf_flattened = keypoint_features.reshape(B, -1)
        kf_reduced = self.fc_kf(kf_flattened)
        kf_reduced = self.dropout_kf(kf_reduced)
        combined = torch.cat([global_pooled, kf_reduced], dim=1)
        combined = self.dropout_final(combined)
        output = self.fc_final(combined)
        return output


# ============================================================================
# Flask App Configuration
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables
MODEL = None
DEVICE = None
TRANSFORM = None
FACE_CASCADE = None
EMOTIONS = {
    0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness",
    4: "Sadness", 5: "Anger", 6: "Neutral"
}

# Processing status tracking
processing_status = {}


# ============================================================================
# Helper Functions
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the emotion detection model"""
    global MODEL, DEVICE, TRANSFORM, FACE_CASCADE
    FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Load model
    MODEL = NKF(num_classes=7, num_landmarks=68, use_rkfa=True)
    
    fan_checkpoint = r'C:\xampp\htdocs\exam\ml_services\checkpoints_rafdb_landmarks\best_fan_rafdb.pth'
    model_path = r'C:\xampp\htdocs\exam\ml_services\checkpoints_fer_IR50_Resnet\best_nkf_rafdb.pth'
    
    fan_ckpt = torch.load(fan_checkpoint, map_location='cpu')
    MODEL.fan.load_state_dict(fan_ckpt['model_state_dict'])
    
    checkpoint = torch.load(model_path, map_location='cpu')
    MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    MODEL.to(DEVICE)
    MODEL.eval()
    
    # Preprocessing
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Face detector
    FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    print("Model loaded successfully!")


def process_video(video_path, result_id):
    """Process video and detect emotions"""
    global processing_status
    
    try:
        processing_status[result_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting processing...'
        }

        # Debug: check if cascade is loaded
        if FACE_CASCADE is None or FACE_CASCADE.empty():
            raise Exception("FACE_CASCADE is not loaded properly!")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        emotion_timeline = []
        emotion_counts = {i: 0 for i in range(7)}
        frames_processed = 0
        
        last_emotion = None
        segment_start_frame = 0
        segment_confidences = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # Update progress
            progress = int((frames_processed / total_frames) * 100)
            processing_status[result_id]['progress'] = progress
            processing_status[result_id]['message'] = f'Processing frame {frames_processed}/{total_frames}'
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                # Preprocess
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img_resized = cv2.resize(face_img_rgb, (112, 112))
                face_pil = Image.fromarray(face_img_resized)
                face_tensor = TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)
                
                # Predict
                with torch.no_grad():
                    outputs = MODEL(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence_val, predicted = torch.max(probabilities, 1)
                    emotion_id = predicted.item()
                    emotion_counts[emotion_id] += 1
                    confidence = confidence_val.item()
                
                # Track emotion changes
                if last_emotion != emotion_id:
                    if last_emotion is not None:
                        start_time = segment_start_frame / fps
                        end_time = frames_processed / fps
                        avg_confidence = sum(segment_confidences) / len(segment_confidences) if segment_confidences else 0.0
                        
                        emotion_timeline.append({
                            "start_time": round(start_time, 2),
                            "end_time": round(end_time, 2),
                            "emotion": EMOTIONS[last_emotion],
                            "confidence": round(avg_confidence, 4)
                        })
                    
                    last_emotion = emotion_id
                    segment_start_frame = frames_processed
                    segment_confidences = [confidence]
                else:
                    segment_confidences.append(confidence)
        
        cap.release()
        
        # Add last segment
        if last_emotion is not None and segment_confidences:
            start_time = segment_start_frame / fps
            end_time = frames_processed / fps
            avg_confidence = sum(segment_confidences) / len(segment_confidences)
            
            emotion_timeline.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "emotion": EMOTIONS[last_emotion],
                "confidence": round(avg_confidence, 4)
            })
        
        # Calculate statistics
        total_detections = sum(emotion_counts.values())
        emotion_percentages = {
            EMOTIONS[k]: round((v / total_detections * 100), 2) if total_detections > 0 else 0
            for k, v in emotion_counts.items()
        }
        
        # Save results
        result_data = {
            'timeline': emotion_timeline,
            'statistics': emotion_percentages,
            'total_frames': total_frames,
            'fps': fps,
            'duration': round(total_frames / fps, 2) if fps > 0 else 0
        }
        
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f'{result_id}.json')
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        processing_status[result_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed!',
            'result_file': f'{result_id}.json'
        }
        
    except Exception as e:
        processing_status[result_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: mp4, avi, mov, mkv, webm'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_id = f'{timestamp}_{filename.rsplit(".", 1)[0]}'
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{result_id}.{filename.rsplit(".", 1)[1]}')
    file.save(video_path)
    
    # Start processing in background
    thread = threading.Thread(target=process_video, args=(video_path, result_id))
    thread.start()
    
    return jsonify({'result_id': result_id})


@app.route('/status/<result_id>')
def get_status(result_id):
    if result_id not in processing_status:
        return jsonify({'error': 'Invalid result ID'}), 404
    
    return jsonify(processing_status[result_id])


@app.route('/results/<result_id>')
def get_results(result_id):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f'{result_id}.json')
    
    if not os.path.exists(result_path):
        return jsonify({'error': 'Results not found'}), 404
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)


@app.route('/results')
def results_page():
    return render_template('results.html')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("FACE_CASCADE LOADED:", FACE_CASCADE)
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
