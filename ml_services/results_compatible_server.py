#!/usr/bin/env python3
"""
Updated Facial Recognition Server
Works with existing results folder and displays existing JSON results
"""

import sys
import os
import subprocess
import time

def print_banner():
    print("="*60)
    print("FACIAL RECOGNITION SERVER - UPDATED")
    print("="*60)
    print("🎭 Emotion Detection for Exam Videos")
    print("🔧 Mode: Compatible with existing results")
    print("📡 Server: http://localhost:5001")
    print("📁 Results: C:\\xampp\\htdocs\\exam\\ml_services\\results")
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

def find_folders():
    """Find videos and results folders"""
    # Videos folder locations to check
    videos_locations = [
        r'C:\xampp\htdocs\exam\videos',
        '../videos',
        'videos'
    ]
    
    # Results folder location
    results_path = r'C:\xampp\htdocs\exam\ml_services\results'
    
    # Find videos folder
    videos_path = None
    for location in videos_locations:
        if os.path.exists(location):
            videos_path = location
            break
    
    if not videos_path:
        videos_path = 'videos'
        os.makedirs(videos_path, exist_ok=True)
        print(f"📁 Created videos folder: {os.path.abspath(videos_path)}")
    
    # Create results folder if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    return videos_path, results_path

def check_existing_results(results_path):
    """Check for existing results in the results folder"""
    json_files = []
    if os.path.exists(results_path):
        for file in os.listdir(results_path):
            if file.endswith('.json'):
                json_files.append(file)
                print(f"📄 Found result file: {file}")
    
    if json_files:
        print(f"✅ Found {len(json_files)} existing result files")
        return True
    else:
        print("📋 No existing result files found")
        return False

def start_server(videos_path, results_path):
    """Start the facial recognition server"""
    print(f"\n🚀 Starting Facial Recognition Server...")
    print(f"   Port: 5001")
    print(f"   Videos: {os.path.abspath(videos_path)}")
    print(f"   Results: {os.path.abspath(results_path)}")
    print(f"\n⏹️  Press Ctrl+C to stop the server")
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
        app.config['RESULTS_FOLDER'] = results_path
        
        # Initialize face detector
        try:
            FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("⚠️  Face detection not available, using simulation only")
            FACE_CASCADE = None
        
        # Emotions mapping
        EMOTIONS = {
            0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness",
            4: "Sadness", 5: "Anger", 6: "Neutral"
        }
        
        print("✅ Server components initialized successfully")
        
        @app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'running',
                'service': 'Facial Recognition Server',
                'port': 5001,
                'model_loaded': True,
                'face_detection': 'OpenCV Haar Cascade' if FACE_CASCADE else 'Simulation',
                'mode': 'compatible_with_existing_results',
                'videos_folder': os.path.abspath(app.config['VIDEOS_FOLDER']),
                'results_folder': os.path.abspath(app.config['RESULTS_FOLDER'])
            })
        
        @app.route('/api/get-videos')
        def get_videos():
            """Get all videos from the videos folder structure"""
            videos = {}
            videos_path_abs = os.path.abspath(app.config['VIDEOS_FOLDER'])
            
            print(f"📁 Looking for videos in: {videos_path_abs}")
            
            if not os.path.exists(videos_path_abs):
                print(f"❌ Videos folder not found: {videos_path_abs}")
                return jsonify({'error': 'Videos folder not found', 'path': videos_path_abs}), 404
            
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
        
        @app.route('/api/get-results')
        def get_results():
            """Get all existing results from the results folder"""
            try:
                results_folder = app.config['RESULTS_FOLDER']
                results_list = []
                
                print(f"📁 Checking results in: {os.path.abspath(results_folder)}")
                
                if not os.path.exists(results_folder):
                    print(f"❌ Results folder not found: {results_folder}")
                    return jsonify([])
                
                # Look for JSON files
                json_files = glob.glob(os.path.join(results_folder, '*.json'))
                
                for file_path in json_files:
                    try:
                        filename = os.path.basename(file_path)
                        file_stats = os.stat(file_path)
                        
                        # Try to read the JSON to get basic info
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Create a result entry
                        result_entry = {
                            'filename': filename,
                            'processed_at': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            'file_size': file_stats.st_size,
                            'type': 'unknown'
                        }
                        
                        # Determine file type and extract info
                        if isinstance(data, dict):
                            if 'results' in data and isinstance(data['results'], list):
                                # Compiled results format
                                result_entry['type'] = 'compiled_results'
                                result_entry['total_videos'] = data.get('total_videos', len(data['results']))
                                result_entry['successful'] = data.get('successful', 0)
                                result_entry['failed'] = data.get('failed', 0)
                            elif 'video' in data and 'segments' in data:
                                # Single video segments format
                                result_entry['type'] = 'video_segments'
                                result_entry['video_name'] = data['video']
                                result_entry['duration'] = float(data.get('totalDuration', 0))
                                result_entry['segments'] = len(data.get('segments', []))
                            elif any('.mp4' in key for key in data.keys()):
                                # Multiple video segments format
                                result_entry['type'] = 'multiple_video_segments'
                                videos = [key for key in data.keys() if '.mp4' in key]
                                result_entry['video_count'] = len(videos)
                                result_entry['videos'] = videos[:3]  # Show first 3
                        
                        results_list.append(result_entry)
                        print(f"📄 {filename} - {result_entry['type']}")
                        
                    except Exception as e:
                        print(f"⚠️  Error reading {file_path}: {e}")
                        # Add it anyway with basic info
                        results_list.append({
                            'filename': os.path.basename(file_path),
                            'processed_at': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                            'type': 'error',
                            'error': str(e)
                        })
                
                print(f"✅ Found {len(results_list)} result files")
                return jsonify(sorted(results_list, key=lambda x: x['processed_at'], reverse=True))
                
            except Exception as e:
                print(f"❌ Error getting results: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/get-result/<filename>')
        def get_specific_result(filename):
            """Get specific result file content"""
            try:
                file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
                
                if not os.path.exists(file_path):
                    return jsonify({'error': 'Result file not found', 'path': file_path}), 404
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to display format
                display_data = convert_to_display_format(data, filename)
                
                return jsonify(display_data)
                
            except Exception as e:
                print(f"❌ Error reading result file {filename}: {e}")
                return jsonify({'error': str(e)}), 500
        
        def convert_to_display_format(data, filename):
            """Convert various JSON formats to a standardized display format"""
            
            if isinstance(data, dict):
                # Check if it's already in compiled results format
                if 'results' in data and isinstance(data['results'], list):
                    return data
                
                # Check if it's segments format
                elif 'video' in data and 'segments' in data:
                    # Single video segments format
                    return convert_segments_to_results(data, filename)
                
                # Check if it's multiple video segments
                elif any('.mp4' in key or '.avi' in key for key in data.keys()):
                    # Multiple video segments format
                    return convert_multiple_segments_to_results(data, filename)
            
            # If we can't determine format, return raw data with metadata
            return {
                'processed_at': datetime.now().isoformat(),
                'total_videos': 1,
                'successful': 1,
                'failed': 0,
                'raw_data': data,
                'source_file': filename,
                'results': [{
                    'student': 'unknown',
                    'video': filename,
                    'status': 'success',
                    'raw_data': data
                }]
            }
        
        def convert_segments_to_results(segments_data, filename):
            """Convert segments format to results format"""
            video_name = segments_data.get('video', 'unknown.mp4')
            duration = float(segments_data.get('totalDuration', 0))
            segments = segments_data.get('segments', [])
            
            # Create emotion timeline from segments
            timeline = []
            for i, segment in enumerate(segments):
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', start_time + 1))
                emotion = segment.get('emotion', 'Neutral')
                confidence = segment.get('confidence', 0.8)
                
                timeline.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'emotion': emotion,
                    'confidence': confidence
                })
            
            # Calculate emotion statistics
            emotion_counts = {}
            total_duration = 0
            
            for segment in timeline:
                emotion = segment['emotion']
                segment_duration = segment['end_time'] - segment['start_time']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + segment_duration
                total_duration += segment_duration
            
            statistics = {}
            primary_emotion = 'Neutral'
            max_duration = 0
            
            for emotion, duration in emotion_counts.items():
                percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                statistics[emotion] = round(percentage, 2)
                
                if duration > max_duration:
                    max_duration = duration
                    primary_emotion = emotion
            
            # Extract student name from video filename
            student_name = video_name.split('_')[0] if '_' in video_name else 'unknown'
            
            result = {
                'student': student_name,
                'video': video_name,
                'status': 'success',
                'timeline': timeline,
                'statistics': statistics,
                'primary_emotion': primary_emotion,
                'duration': duration,
                'segments_count': len(segments),
                'processed_at': datetime.now().isoformat()
            }
            
            return {
                'processed_at': datetime.now().isoformat(),
                'total_videos': 1,
                'successful': 1,
                'failed': 0,
                'source_file': filename,
                'results': [result]
            }
        
        def convert_multiple_segments_to_results(data, filename):
            """Convert multiple video segments format to results format"""
            results = []
            
            for video_key, video_data in data.items():
                if not isinstance(video_data, dict):
                    continue
                
                if 'video' in video_data and 'segments' in video_data:
                    converted = convert_segments_to_results(video_data, filename)
                    if 'results' in converted and converted['results']:
                        results.extend(converted['results'])
            
            return {
                'processed_at': datetime.now().isoformat(),
                'total_videos': len(results),
                'successful': len(results),
                'failed': 0,
                'source_file': filename,
                'results': results
            }
        
        @app.route('/api/process-video', methods=['POST'])
        def process_video():
            """Process a single video (simulation for now)"""
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
                
                # Simulate processing
                result = simulate_video_processing(video_path, student_name, video_filename)
                
                # Save individual result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"single_result_{student_name}_{video_filename}_{timestamp}.json"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"✅ Processed and saved: {result_filename}")
                return jsonify(result)
                
            except Exception as e:
                print(f"❌ Error processing video: {e}")
                return jsonify({'error': str(e)}), 500
        
        def simulate_video_processing(video_path, student_name, video_filename):
            """Simulate video processing for demonstration"""
            # Create realistic simulation
            emotions = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Anger', 'Fear']
            duration = 60 + random.randint(-20, 60)  # 40-120 seconds
            
            # Create emotion segments
            timeline = []
            current_time = 0
            
            while current_time < duration:
                segment_duration = random.randint(5, 20)
                end_time = min(current_time + segment_duration, duration)
                
                # Choose emotion based on video name
                if 'wrong' in video_filename.lower():
                    emotion = random.choice(['Sadness', 'Anger', 'Fear', 'Neutral'])
                elif 'right' in video_filename.lower() or 'correct' in video_filename.lower():
                    emotion = random.choice(['Happiness', 'Surprise', 'Neutral'])
                else:
                    emotion = random.choice(emotions)
                
                timeline.append({
                    'start_time': round(current_time, 2),
                    'end_time': round(end_time, 2),
                    'emotion': emotion,
                    'confidence': round(random.uniform(0.7, 0.95), 4)
                })
                
                current_time = end_time
            
            # Calculate statistics
            emotion_durations = {}
            for segment in timeline:
                emotion = segment['emotion']
                segment_duration = segment['end_time'] - segment['start_time']
                emotion_durations[emotion] = emotion_durations.get(emotion, 0) + segment_duration
            
            statistics = {}
            primary_emotion = 'Neutral'
            max_duration = 0
            
            for emotion, total_duration in emotion_durations.items():
                percentage = (total_duration / duration * 100)
                statistics[emotion] = round(percentage, 2)
                
                if total_duration > max_duration:
                    max_duration = total_duration
                    primary_emotion = emotion
            
            return {
                'student': student_name,
                'video': video_filename,
                'status': 'success',
                'timeline': timeline,
                'statistics': statistics,
                'primary_emotion': primary_emotion,
                'duration': duration,
                'face_detection_rate': round(random.uniform(0.6, 0.9), 4),
                'simulation_mode': True,
                'processed_at': datetime.now().isoformat()
            }
        
        @app.route('/api/process-all', methods=['POST'])
        def process_all_videos():
            """Process all videos in batch"""
            print("🚀 Starting batch processing...")
            
            try:
                # Get videos
                videos_response = get_videos()
                if videos_response.status_code != 200:
                    return jsonify({'error': 'Failed to get videos'}), 500
                
                videos_data = videos_response.get_json()
                results = []
                
                for student_name, videos in videos_data.items():
                    for video_info in videos:
                        try:
                            result = simulate_video_processing(
                                video_info['path'], 
                                student_name, 
                                video_info['filename']
                            )
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
                
                # Save batch results
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
                
                print(f"📊 Batch processing complete: {len(results)} videos")
                return jsonify(compiled_results)
                
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Start the server
        print(f"📁 Videos: {os.path.abspath(videos_path)}")
        print(f"📊 Results: {os.path.abspath(results_path)}")
        print("\n🌐 URLs:")
        print("   Health Check: http://localhost:5001/health")
        print("   Get Results: http://localhost:5001/api/get-results")
        print("   Admin Interface: Open improved_facial_recognition.html")
        print("\n⚡ Server is ready!")
        
        app.run(
            debug=False,
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
    
    # Find folders
    print("\n📁 Locating folders...")
    videos_path, results_path = find_folders()
    
    print(f"📹 Videos folder: {os.path.abspath(videos_path)}")
    print(f"📊 Results folder: {os.path.abspath(results_path)}")
    
    # Check existing results
    print("\n📋 Checking existing results...")
    has_results = check_existing_results(results_path)
    
    # Summary
    print(f"\n📋 Setup Summary:")
    print(f"   ✅ Dependencies: Ready")
    print(f"   📁 Videos: {os.path.abspath(videos_path)}")
    print(f"   📊 Results: {os.path.abspath(results_path)} ({'Has files' if has_results else 'Empty'})")
    print(f"   🌐 Interface: improved_facial_recognition.html")
    
    ready = input("\nStart the server? (Y/n): ").lower().strip()
    if ready and ready != 'y':
        print("👋 Startup cancelled")
        return 0
    
    # Start server
    success = start_server(videos_path, results_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())