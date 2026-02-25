#!/usr/bin/env python3
"""
Fixed Facial Recognition Server Startup Script
Handles encoding issues and directly runs the server code
"""

import sys
import os
import subprocess
import time

def print_banner():
    print("="*60)
    print("FACIAL RECOGNITION SERVER - STARTUP")
    print("="*60)
    print("🎭 Emotion Detection for Exam Videos")
    print("🔧 Mode: Simulation (No ML dependencies required)")
    print("📡 Server: http://localhost:5001")
    print("="*60)

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['flask', 'flask_cors', 'cv2', 'numpy']
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package} - available")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - missing")
    
    if missing:
        print("\n📦 Installing missing dependencies...")
        pip_packages = {
            'flask': 'Flask',
            'flask_cors': 'Flask-CORS', 
            'cv2': 'opencv-python',
            'numpy': 'numpy'
        }
        
        for package in missing:
            pip_package = pip_packages.get(package, package)
            print(f"   Installing {pip_package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_package])
                print(f"   ✅ {pip_package} installed")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {pip_package}: {e}")
                return False
    
    return True

def check_videos_folder():
    """Check if videos folder exists and has content"""
    videos_path = os.path.join('..', 'videos')  # Look in parent directory
    if not os.path.exists(videos_path):
        # Try current directory
        videos_path = 'videos'
        if not os.path.exists(videos_path):
            print(f"⚠️  Videos folder not found")
            print("   Creating videos folder...")
            os.makedirs(videos_path, exist_ok=True)
            print("   📁 Add your video files to the 'videos' folder")
            print("   📁 Structure: videos/student_name/video.mp4")
            return False, videos_path
    
    # Count videos
    total_videos = 0
    students = 0
    for student in os.listdir(videos_path):
        student_path = os.path.join(videos_path, student)
        if os.path.isdir(student_path):
            videos = [f for f in os.listdir(student_path) 
                     if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
            if videos:
                students += 1
                total_videos += len(videos)
                print(f"📹 {student}: {len(videos)} videos")
    
    if total_videos > 0:
        print(f"✅ Found {total_videos} videos for {students} students")
        return True, videos_path
    else:
        print("⚠️  No videos found in the videos folder")
        print("   Add some video files to test the system")
        return False, videos_path

def start_server(videos_path):
    """Start the facial recognition server directly"""
    print("\n🚀 Starting Facial Recognition Server...")
    print("   Port: 5001")
    print("   Interface: All interfaces (0.0.0.0)")
    print("   Mode: Development with auto-reload")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Import required modules
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        import cv2
        import numpy as np
        import json
        from datetime import datetime
        import glob
        import random
        
        # Create Flask app
        app = Flask(__name__)
        CORS(app)
        
        # Configure paths
        app.config['VIDEOS_FOLDER'] = videos_path
        app.config['RESULTS_FOLDER'] = 'facial_results'
        
        # Create results folder
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        
        # Initialize face detector
        FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotions mapping
        EMOTIONS = {
            0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness",
            4: "Sadness", 5: "Anger", 6: "Neutral"
        }
        
        print("✅ Server components initialized successfully")
        print("🎬 Starting emotion detection server...")
        
        @app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'running',
                'service': 'Facial Recognition Server',
                'port': 5001,
                'model_loaded': True,
                'face_detection': 'OpenCV Haar Cascade',
                'mode': 'simulation'
            })
        
        @app.route('/api/get-videos')
        def get_videos():
            """Get all videos from the videos folder structure"""
            videos = {}
            videos_path_abs = os.path.abspath(app.config['VIDEOS_FOLDER'])
            
            print(f"📁 Looking for videos in: {videos_path_abs}")
            
            if not os.path.exists(videos_path_abs):
                print(f"❌ Videos folder not found: {videos_path_abs}")
                return jsonify({'error': 'Videos folder not found'}), 404
            
            try:
                for student_folder in os.listdir(videos_path_abs):
                    student_path = os.path.join(videos_path_abs, student_folder)
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
                                    print(f"⚠️  Error accessing {video_file}: {e}")
                                    
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
            try:
                data = request.json
                student_name = data.get('student_name')
                video_filename = data.get('video_filename')
                
                if not student_name or not video_filename:
                    return jsonify({'error': 'Missing student_name or video_filename'}), 400
                
                video_path = os.path.join(app.config['VIDEOS_FOLDER'], student_name, video_filename)
                
                print(f"🎬 Processing: {student_name}/{video_filename}")
                print(f"📁 Path: {video_path}")
                
                if not os.path.exists(video_path):
                    print(f"❌ Video file not found: {video_path}")
                    return jsonify({'error': 'Video file not found'}), 404
                
                result = process_video_emotions(video_path, student_name, video_filename)
                print(f"✅ Processed successfully: {student_name}/{video_filename}")
                return jsonify(result)
                
            except Exception as e:
                print(f"❌ Error processing video: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/process-all', methods=['POST'])
        def process_all_videos():
            """Process all videos in the folder structure"""
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
                
                processed_count = 0
                for student_name, videos in videos_data.items():
                    print(f"👤 Processing {student_name} ({len(videos)} videos)...")
                    
                    for video_info in videos:
                        try:
                            video_path = video_info['path']
                            result = process_video_emotions(video_path, student_name, video_info['filename'])
                            results.append(result)
                            processed_count += 1
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
                    'results': results
                }
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = os.path.join(app.config['RESULTS_FOLDER'], f'compiled_results_{timestamp}.json')
                
                with open(results_file, 'w') as f:
                    json.dump(compiled_results, f, indent=2)
                
                print(f"📊 Batch processing complete:")
                print(f"   ✅ Successful: {compiled_results['successful']}")
                print(f"   ❌ Failed: {compiled_results['failed']}")
                print(f"   💾 Results saved: {results_file}")
                
                return jsonify(compiled_results)
                
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                return jsonify({'error': str(e)}), 500
        
        def process_video_emotions(video_path, student_name, video_filename):
            """Process video and extract emotions using simulation"""
            print(f"🔍 Analyzing video: {video_filename}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video: {video_path}")
            
            try:
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                
                print(f"📹 Video info: {total_frames} frames, {fps} FPS, {duration:.2f}s")
                
                # Process video (simulation with actual frame analysis)
                emotion_timeline = []
                emotion_counts = {i: 0 for i in range(7)}
                frames_processed = 0
                faces_detected = 0
                
                last_emotion = None
                segment_start_time = 0
                
                # Sample every 30 frames for faster processing
                frame_skip = max(1, fps // 2)  # Process 2 times per second
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames_processed += 1
                    
                    # Skip frames for efficiency
                    if frames_processed % frame_skip != 0:
                        continue
                    
                    current_time = frames_processed / fps
                    
                    # Detect faces using OpenCV
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = FACE_CASCADE.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(50, 50)
                    )
                    
                    if len(faces) > 0:
                        faces_detected += 1
                        
                        # Simulate emotion detection based on video characteristics
                        emotion_id = simulate_emotion_detection(frame, faces[0], video_filename)
                        emotion_counts[emotion_id] += 1
                        
                        # Create segments when emotion changes
                        if last_emotion != emotion_id:
                            if last_emotion is not None:
                                # Close previous segment
                                emotion_timeline.append({
                                    "start_time": round(segment_start_time, 2),
                                    "end_time": round(current_time, 2),
                                    "emotion": EMOTIONS[last_emotion],
                                    "confidence": round(0.7 + random.random() * 0.25, 4)
                                })
                            
                            last_emotion = emotion_id
                            segment_start_time = current_time
                
                cap.release()
                
                # Add final segment
                if last_emotion is not None:
                    emotion_timeline.append({
                        "start_time": round(segment_start_time, 2),
                        "end_time": round(duration, 2),
                        "emotion": EMOTIONS[last_emotion],
                        "confidence": round(0.7 + random.random() * 0.25, 4)
                    })
                
                # Calculate statistics
                total_detections = sum(emotion_counts.values())
                emotion_percentages = {}
                
                if total_detections > 0:
                    for k, v in emotion_counts.items():
                        emotion_percentages[EMOTIONS[k]] = round((v / total_detections * 100), 2)
                else:
                    # No faces detected - create neutral result
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
                print(f"📈 Segments: {len(emotion_timeline)}")
                
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
                    'simulation_mode': True,
                    'processed_at': datetime.now().isoformat()
                }
                
                return result
                
            except Exception as e:
                cap.release()
                raise e
        
        def simulate_emotion_detection(frame, face_rect, video_filename):
            """Simulate emotion detection based on video characteristics"""
            # Extract some basic features from the frame
            x, y, w, h = face_rect
            face_region = frame[y:y+h, x:x+w]
            
            # Use filename as a hint for demonstration
            if 'wrong' in video_filename.lower():
                # For "wrong" answers, bias towards negative emotions
                emotions_pool = [1, 2, 4, 5, 6]  # Fear, Disgust, Sadness, Anger, Neutral
                weights = [0.15, 0.1, 0.3, 0.2, 0.25]
            elif 'right' in video_filename.lower() or 'correct' in video_filename.lower():
                # For "right/correct" answers, bias towards positive emotions
                emotions_pool = [0, 3, 6]  # Surprise, Happiness, Neutral
                weights = [0.2, 0.5, 0.3]
            else:
                # Default distribution
                emotions_pool = [0, 1, 2, 3, 4, 5, 6]
                weights = [0.1, 0.1, 0.1, 0.2, 0.15, 0.1, 0.25]
            
            # Add some randomness based on image characteristics
            brightness = np.mean(face_region)
            if brightness < 100:
                # Dark image - might indicate confusion or sadness
                if 4 in emotions_pool:  # Sadness
                    idx = emotions_pool.index(4)
                    weights[idx] *= 1.5
            elif brightness > 180:
                # Bright image - might indicate happiness
                if 3 in emotions_pool:  # Happiness
                    idx = emotions_pool.index(3)
                    weights[idx] *= 1.3
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Random selection based on weights
            emotion_id = np.random.choice(emotions_pool, p=weights)
            
            return emotion_id
        
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
                                'failed': data['failed']
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
        
        # Start the server
        print(f"🎯 Videos folder: {os.path.abspath(videos_path)}")
        print(f"💾 Results folder: {os.path.abspath(app.config['RESULTS_FOLDER'])}")
        print("\n🌐 Server URLs:")
        print("   Health Check: http://localhost:5001/health")
        print("   Admin Interface: Open improved_facial_recognition.html in browser")
        print("\n⚡ Server is ready!")
        
        app.run(
            debug=False,  # Disable debug to avoid reloader issues
            port=5001, 
            host='0.0.0.0',
            use_reloader=False
        )
        
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Server error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main startup function"""
    print_banner()
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed")
        return 1
    
    # Check videos folder
    print("\n📁 Checking videos folder...")
    has_videos, videos_path = check_videos_folder()
    
    if not has_videos:
        print("\n⚠️  No videos found, but server will still start")
        print("   You can add videos later and use 'Load Videos' in the interface")
    
    # Ask user if ready
    print(f"\n📋 Setup Summary:")
    print(f"   ✅ Dependencies: Ready")
    print(f"   📁 Videos: {'Ready' if has_videos else 'Empty (can add later)'}")
    print(f"   📁 Videos Path: {os.path.abspath(videos_path)}")
    print(f"   🌐 Interface: improved_facial_recognition.html")
    
    ready = input("\nStart the server? (Y/n): ").lower().strip()
    if ready and ready != 'y':
        print("👋 Startup cancelled")
        return 0
    
    # Start server
    success = start_server(videos_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())