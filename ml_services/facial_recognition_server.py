from flask import Flask, request, jsonify, send_from_directory
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

# Import your model classes from app.py
from app import NKF, EMOTIONS, TRANSFORM
# Initialize device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize face cascade
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize model
MODEL = NKF(num_classes=7, num_landmarks=68, use_rkfa=True)
checkpoint = torch.load('./checkpoints_fer_IR50_Resnet/best_nkf_rafdb.pth', map_location=DEVICE)
MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
MODEL.to(DEVICE)
MODEL.eval()

app = Flask(__name__)
CORS(app) #Enable CORS for all routes
app.config['VIDEOS_FOLDER'] = r'C:\xampp\htdocs\exam\videos'
app.config['RESULTS_FOLDER'] = 'facial_results'

os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'Facial Recognition Server',
        'port': 5001,
        'model_loaded': MODEL is not None
    })

@app.route('/api/get-videos')
def get_videos():
    """Get all videos from the videos folder structure"""
    videos = {}
    videos_path = app.config['VIDEOS_FOLDER']
    
    if not os.path.exists(videos_path):
        return jsonify({'error': 'Videos folder not found'}), 404
    
    for student_folder in os.listdir(videos_path):
        student_path = os.path.join(videos_path, student_folder)
        if os.path.isdir(student_path):
            student_videos = []
            for video_file in os.listdir(student_path):
                if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    student_videos.append({
                        'filename': video_file,
                        'path': os.path.join(student_path, video_file),
                        'size': os.path.getsize(os.path.join(student_path, video_file))
                    })
            if student_videos:
                videos[student_folder] = student_videos
    
    return jsonify(videos)

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process a single video for emotion recognition"""
    data = request.json
    student_name = data.get('student_name')
    video_filename = data.get('video_filename')
    
    video_path = os.path.join(app.config['VIDEOS_FOLDER'], student_name, video_filename)
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    try:
        result = process_video_emotions(video_path, student_name, video_filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-all', methods=['POST'])
def process_all_videos():
    """Process all videos in the folder structure"""
    results = []
    videos_data = get_videos().json
    
    for student_name, videos in videos_data.items():
        for video_info in videos:
            try:
                video_path = video_info['path']
                result = process_video_emotions(video_path, student_name, video_info['filename'])
                results.append(result)
            except Exception as e:
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
        'results': results
    }
    
    results_file = os.path.join(app.config['RESULTS_FOLDER'], f'compiled_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(compiled_results, f, indent=4)
    
    return jsonify(compiled_results)

def process_video_emotions(video_path, student_name, video_filename):
    """Process video and extract emotions using your model"""
    global FACE_CASCADE, MODEL, TRANSFORM, DEVICE, EMOTIONS

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
    
    # Get primary emotion
    primary_emotion_id = max(emotion_counts, key=emotion_counts.get)
    primary_emotion = EMOTIONS[primary_emotion_id]
    
    result = {
        'student': student_name,
        'video': video_filename,
        'status': 'success',
        'timeline': emotion_timeline,
        'statistics': emotion_percentages,
        'primary_emotion': primary_emotion,
        'total_frames': total_frames,
        'fps': fps,
        'duration': round(total_frames / fps, 2) if fps > 0 else 0,
        'processed_at': datetime.now().isoformat()
    }
    
    return result

@app.route('/api/get-results')
def get_results():
    """Get all compiled results"""
    results_files = glob.glob(os.path.join(app.config['RESULTS_FOLDER'], 'compiled_results_*.json'))
    results_list = []
    
    for file_path in results_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            results_list.append({
                'filename': os.path.basename(file_path),
                'processed_at': data['processed_at'],
                'total_videos': data['total_videos'],
                'successful': data['successful'],
                'failed': data['failed']
            })
    
    return jsonify(sorted(results_list, key=lambda x: x['processed_at'], reverse=True))

@app.route('/api/get-result/<filename>')
def get_specific_result(filename):
    """Get specific result file"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Result file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001, host ='0.0.0.0')