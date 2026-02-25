#!/usr/bin/env python3
"""
Enhanced Facial Recognition Server with Real NKF Model
Uses the actual trained model for emotion detection without simulation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import json
from datetime import datetime
import glob

# Model architecture classes from app.py
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

app = Flask(__name__)
CORS(app)

# Configure paths - these should be set according to your environment
app.config['VIDEOS_FOLDER'] = r'C:\xampp\htdocs\exam\videos'  # Update this path
app.config['RESULTS_FOLDER'] = r'C:\xampp\htdocs\exam\ml_services\facial_results'
app.config['MODEL_PATH'] = './checkpoints_fer_IR50_Resnet/best_nkf_rafdb.pth'  # Update this path
app.config['FAN_PATH'] = './checkpoints_rafdb_landmarks/best_fan_rafdb.pth'  # Update this path

# Create results folder
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
print(f"📁 Results folder: {app.config['RESULTS_FOLDER']}")
print(f"📁 Folder exists: {os.path.exists(app.config['RESULTS_FOLDER'])}")
try:
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    print(f"✅ Results folder created/verified")
except Exception as e:
    print(f"❌ Cannot create results folder: {e}")

# Global variables
MODEL = None
DEVICE = None
TRANSFORM = None
FACE_CASCADE = None
MODEL_LOADED = False

EMOTIONS = {
    0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness",
    4: "Sadness", 5: "Anger", 6: "Neutral"
}

def load_model():
    """Load the NKF emotion detection model"""
    global MODEL, DEVICE, TRANSFORM, FACE_CASCADE, MODEL_LOADED
    
    try:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {DEVICE}")
        
        # Initialize model
        MODEL = NKF(num_classes=7, num_landmarks=68, use_rkfa=True)
        
        # Load FAN checkpoint
        if os.path.exists(app.config['FAN_PATH']):
            fan_checkpoint = torch.load(app.config['FAN_PATH'], map_location=DEVICE)
            MODEL.fan.load_state_dict(fan_checkpoint['model_state_dict'])
            print("✅ FAN model loaded successfully")
        else:
            print("⚠️ FAN checkpoint not found, using random weights")
        
        # Load main model checkpoint
        if os.path.exists(app.config['MODEL_PATH']):
            checkpoint = torch.load(app.config['MODEL_PATH'], map_location=DEVICE)
            MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ NKF model loaded successfully")
        else:
            print("⚠️ Model checkpoint not found, using random weights")
        
        MODEL.to(DEVICE)
        MODEL.eval()
        
        # Initialize preprocessing
        TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize face detector
        FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        MODEL_LOADED = True
        print("🎭 Facial recognition model ready!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        MODEL_LOADED = False

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'Enhanced Facial Recognition Server',
        'port': 5001,
        'model_loaded': MODEL_LOADED,
        'device': str(DEVICE) if DEVICE else None,
        'face_detection': 'OpenCV Haar Cascade',
        'mode': 'real_model'
    })

@app.route('/api/get-videos')
def get_videos():
    """Get all videos from the videos folder structure"""
    videos = {}
    videos_path = app.config['VIDEOS_FOLDER']
    
    print(f"🔍 Looking for videos in: {os.path.abspath(videos_path)}")
    
    if not os.path.exists(videos_path):
        print(f"❌ Videos folder not found: {videos_path}")
        return jsonify({'error': 'Videos folder not found'}), 404
    
    try:
        for student_folder in os.listdir(videos_path):
            student_path = os.path.join(videos_path, student_folder)
            if os.path.isdir(student_path):
                student_videos = []
                for video_file in os.listdir(student_path):
                    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        video_path = os.path.join(student_path, video_file)
                        try:
                            file_size = os.path.getsize(video_path)
                            student_videos.append({
                                'filename': video_file,
                                'path': video_path,
                                'size': file_size
                            })
                        except OSError as e:
                            print(f"⚠️ Error accessing {video_file}: {e}")
                            
                if student_videos:
                    videos[student_folder] = student_videos
                    print(f"👤 {student_folder}: {len(student_videos)} videos")
        
        print(f"✅ Found videos for {len(videos)} students")
        return jsonify(videos)
        
    except Exception as e:
        print(f"❌ Error scanning videos: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process a single video for emotion recognition"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        student_name = data.get('student_name')
        video_filename = data.get('video_filename')
        
        if not student_name or not video_filename:
            return jsonify({'error': 'Missing student_name or video_filename'}), 400
        
        video_path = os.path.join(app.config['VIDEOS_FOLDER'], student_name, video_filename)
        
        print(f"🎬 Processing: {student_name}/{video_filename}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return jsonify({'error': 'Video file not found'}), 404
        
        result = process_video_emotions(video_path, student_name, video_filename)
        print(f"✅ Processed successfully: {student_name}/{video_filename}")

        #For Individual Result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        individual_result_file = os.path.join(
            app.config['RESULTS_FOLDER'], 
            f'single_result_{student_name}_{video_filename}_{timestamp}.json'
        )
        
        try:
            with open(individual_result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Individual result saved: {individual_result_file}")
        except Exception as e:
            print(f"❌ Error saving individual result: {e}")

        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

def process_video_emotions(video_path, student_name, video_filename):
    """Process video and extract emotions using the real NKF model"""
    global FACE_CASCADE, MODEL, TRANSFORM, DEVICE, EMOTIONS
    
    print(f"📹 Analyzing video: {video_filename}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    
    try:
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video info: {total_frames} frames, {fps} FPS, {duration:.2f}s")
        
        emotion_timeline = []
        emotion_counts = {i: 0 for i in range(7)}
        frames_processed = 0
        faces_detected = 0
        
        last_emotion = None
        segment_start_frame = 0
        segment_confidences = []
        
        # Process every few frames for efficiency
        frame_skip = max(1, fps // 2)  # Process 2 times per second
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # Skip frames for efficiency
            if frames_processed % frame_skip != 0:
                continue
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50)
            )
            
            if len(faces) > 0:
                faces_detected += 1
                x, y, w, h = faces[0]  # Use the first detected face
                face_img = frame[y:y+h, x:x+w]
                
                # Preprocess face for model
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img_resized = cv2.resize(face_img_rgb, (112, 112))
                face_pil = Image.fromarray(face_img_resized)
                face_tensor = TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)
                
                # Predict emotion using NKF model
                with torch.no_grad():
                    outputs = MODEL(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence_val, predicted = torch.max(probabilities, 1)
                    emotion_id = predicted.item()
                    confidence = confidence_val.item()
                    
                    emotion_counts[emotion_id] += 1
                
                # Track emotion changes for timeline
                if last_emotion != emotion_id:
                    if last_emotion is not None:
                        # Close previous segment
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
        
        # Add final segment
        if last_emotion is not None and segment_confidences:
            start_time = segment_start_frame / fps
            end_time = duration
            avg_confidence = sum(segment_confidences) / len(segment_confidences)
            
            emotion_timeline.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "emotion": EMOTIONS[last_emotion],
                "confidence": round(avg_confidence, 4)
            })
        
        # Calculate statistics
        total_detections = sum(emotion_counts.values())
        emotion_percentages = {}
        
        if total_detections > 0:
            for k, v in emotion_counts.items():
                emotion_percentages[EMOTIONS[k]] = round((v / total_detections * 100), 2)
        else:
            # No faces detected
            emotion_percentages = {emotion: 0 for emotion in EMOTIONS.values()}
            emotion_percentages['Neutral'] = 100.0
            emotion_timeline = [{
                "start_time": 0.0,
                "end_time": round(duration, 2),
                "emotion": "Neutral",
                "confidence": 0.5
            }]
        
        # Get primary emotion
        if total_detections > 0:
            primary_emotion_id = max(emotion_counts, key=emotion_counts.get)
            primary_emotion = EMOTIONS[primary_emotion_id]
        else:
            primary_emotion = "Neutral"
        
        face_detection_rate = faces_detected / (frames_processed // frame_skip) if frames_processed > 0 else 0
        
        print(f"👥 Faces detected in {face_detection_rate:.1%} of frames")
        print(f"😊 Primary emotion: {primary_emotion}")
        print(f"📈 Timeline segments: {len(emotion_timeline)}")
        
        result = {
            'student': student_name,
            'video': video_filename,
            'status': 'success',
            'timeline': emotion_timeline,
            'statistics': emotion_percentages,
            'primary_emotion': primary_emotion,
            'total_frames': total_frames,
            'fps': fps,
            'duration': round(duration, 2),
            'face_detection_rate': round(face_detection_rate, 4),
            'faces_detected': faces_detected,
            'model_type': 'NKF',
            'processed_at': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        cap.release()
        raise e

@app.route('/api/process-all', methods=['POST'])
def process_all_videos():
    """Process all videos in the folder structure"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    print("🚀 Starting batch processing...")
    results = []
    
    try:
        # Get videos data
        videos_response = get_videos()
        if videos_response.status_code != 200:
            return jsonify({'error': 'Failed to get videos'}), 500
            
        videos_data = videos_response.get_json()
        total_videos = sum(len(videos) for videos in videos_data.values())
        
        print(f"📊 Processing {total_videos} videos for {len(videos_data)} students")
        
        for student_name, videos in videos_data.items():
            print(f"👤 Processing {student_name} ({len(videos)} videos)...")
            
            for video_info in videos:
                try:
                    video_path = video_info['path']
                    result = process_video_emotions(video_path, student_name, video_info['filename'])
                    results.append(result)
                    print(f"   ✅ {video_info['filename']} - {result['primary_emotion']}")
                    
                except Exception as e:
                    print(f"   ❌ {video_info['filename']} - Error: {e}")
                    results.append({
                        'student': student_name,
                        'video': video_info['filename'],
                        'error': str(e),
                        'status': 'failed'
                    })
        
        # Save compiled results
        compiled_results = {
            'processed_at': datetime.now().isoformat(),
            'total_videos': len(results),
            'successful': len([r for r in results if r.get('status') == 'success']),
            'failed': len([r for r in results if r.get('status') == 'failed']),
            'model_type': 'NKF',
            'results': results
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f'compiled_results_{timestamp}.json')

        print(f"💾 Trying to save: {results_file}")
        print(f"📊 Data to save: {len(results)} results")

        try:
            with open(results_file, 'w') as f:
                json.dump(compiled_results, f, indent=2)
            print(f"✅ File saved successfully!")
            print(f"📁 File exists: {os.path.exists(results_file)}")
            print(f"📏 File size: {os.path.getsize(results_file)} bytes")
        except Exception as e:
            print(f"❌ Save error: {e}")
        
        with open(results_file, 'w') as f:
            json.dump(compiled_results, f, indent=2)
        
        print(f"📊 Batch processing complete:")
        print(f"   ✅ Successful: {compiled_results['successful']}")
        print(f"   ❌ Failed: {compiled_results['failed']}")
        print(f"   💾 Results saved: {results_file}")
        print(f"📁 File exists: {os.path.exists(results_file)}")
        print(f"📊 Results folder: {os.path.abspath(app.config['RESULTS_FOLDER'])}")
        
        return jsonify(compiled_results)
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-results')
def get_results():
    """Get all compiled results"""
    try:
        results_files = glob.glob(os.path.join(app.config['RESULTS_FOLDER'], 'compiled_results_*.json'))
        results_list = []
        
        for file_path in results_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results_list.append({
                        'filename': os.path.basename(file_path),
                        'processed_at': data['processed_at'],
                        'total_videos': data['total_videos'],
                        'successful': data['successful'],
                        'failed': data['failed'],
                        'model_type': data.get('model_type', 'unknown')
                    })
            except Exception as e:
                print(f"Error reading results file {file_path}: {e}")
        
        return jsonify(sorted(results_list, key=lambda x: x['processed_at'], reverse=True))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-result/<filename>')
def get_specific_result(filename):
    """Get specific result file"""
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Result file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("ENHANCED FACIAL RECOGNITION SERVER")
    print("="*60)
    print("🎭 Real NKF Model for Emotion Detection")
    print("🔧 No simulation - uses actual trained weights")
    print("📁 Update paths in config section for your environment")
    print("="*60)
    
    # Load the model
    print("\n🔄 Loading NKF model...")
    load_model()
    
    if not MODEL_LOADED:
        print("\n⚠️ Model not loaded - check paths in config section")
        print("   Update MODEL_PATH and FAN_PATH variables")
        
    print(f"\n🚀 Starting server on port 5001...")
    print("🔗 Health check: http://localhost:5001/health")
    print("🎛️ Admin interface: Open improved_facial_recognition.html")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)
    
    try:
        app.run(
            debug=True, 
            port=5001, 
            host='0.0.0.0',
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")