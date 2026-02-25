#!/usr/bin/env python3
"""
Configuration Helper for Enhanced Facial Recognition Server
"""

import os
import json
from pathlib import Path

def create_config():
    """Create configuration file with paths"""
    
    print("="*60)
    print("FACIAL RECOGNITION SERVER CONFIGURATION")
    print("="*60)
    
    config = {}
    
    # Videos folder
    print("\n📹 VIDEOS FOLDER:")
    print("Where are your student video files located?")
    videos_default = "videos"
    videos_path = input(f"Path [{videos_default}]: ").strip() or videos_default
    
    if not os.path.exists(videos_path):
        create = input(f"Folder '{videos_path}' doesn't exist. Create it? (Y/n): ").strip().lower()
        if create != 'n':
            os.makedirs(videos_path, exist_ok=True)
            print(f"✅ Created folder: {videos_path}")
    
    config['videos_folder'] = os.path.abspath(videos_path)
    
    # Model checkpoint
    print("\n🤖 NKF MODEL CHECKPOINT:")
    print("Path to the main NKF model file (best_nkf_rafdb.pth)")
    model_path = input("Model path: ").strip()
    
    if model_path and os.path.exists(model_path):
        print("✅ Model checkpoint found")
        config['model_path'] = os.path.abspath(model_path)
    else:
        print("⚠️ Model checkpoint not found - will use random weights")
        config['model_path'] = model_path or "checkpoints_fer_IR50_Resnet/best_nkf_rafdb.pth"
    
    # FAN checkpoint
    print("\n👥 FAN LANDMARK MODEL:")
    print("Path to the FAN landmark model file (best_fan_rafdb.pth)")
    fan_path = input("FAN path: ").strip()
    
    if fan_path and os.path.exists(fan_path):
        print("✅ FAN checkpoint found")
        config['fan_path'] = os.path.abspath(fan_path)
    else:
        print("⚠️ FAN checkpoint not found - will use random weights")
        config['fan_path'] = fan_path or "checkpoints_rafdb_landmarks/best_fan_rafdb.pth"
    
    # Results folder
    results_folder = "facial_results"
    config['results_folder'] = os.path.abspath(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    
    # Save configuration
    with open('facial_recognition_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📋 CONFIGURATION SUMMARY:")
    print(f"   Videos: {config['videos_folder']}")
    print(f"   Model: {config['model_path']}")
    print(f"   FAN: {config['fan_path']}")
    print(f"   Results: {config['results_folder']}")
    print(f"\n💾 Configuration saved to: facial_recognition_config.json")
    
    return config

def generate_server_script(config):
    """Generate configured server script"""
    
    server_code = f'''#!/usr/bin/env python3
"""
Auto-configured Enhanced Facial Recognition Server
Generated from configuration
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced server code
from enhanced_facial_recognition_server import *

if __name__ == '__main__':
    # Override configuration with auto-detected paths
    app.config['VIDEOS_FOLDER'] = r'{config['videos_folder']}'
    app.config['RESULTS_FOLDER'] = r'{config['results_folder']}'
    app.config['MODEL_PATH'] = r'{config['model_path']}'
    app.config['FAN_PATH'] = r'{config['fan_path']}'
    
    # Create results folder
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    print("="*60)
    print("AUTO-CONFIGURED FACIAL RECOGNITION SERVER")
    print("="*60)
    print("🎭 Enhanced NKF Model for Real Emotion Detection")
    print(f"📹 Videos: {{app.config['VIDEOS_FOLDER']}}")
    print(f"🤖 Model: {{app.config['MODEL_PATH']}}")
    print(f"👥 FAN: {{app.config['FAN_PATH']}}")
    print(f"📊 Results: {{app.config['RESULTS_FOLDER']}}")
    print("="*60)
    
    # Check paths
    paths_ok = True
    if not os.path.exists(app.config['VIDEOS_FOLDER']):
        print(f"⚠️ Videos folder not found: {{app.config['VIDEOS_FOLDER']}}")
        
    if not os.path.exists(app.config['MODEL_PATH']):
        print(f"⚠️ Model checkpoint not found: {{app.config['MODEL_PATH']}}")
        print("   Server will use random weights (simulation mode)")
        
    if not os.path.exists(app.config['FAN_PATH']):
        print(f"⚠️ FAN checkpoint not found: {{app.config['FAN_PATH']}}")
        print("   Server will use random weights for landmarks")
    
    # Load the model
    print("\\n🔄 Loading NKF model...")
    load_model()
    
    if MODEL_LOADED:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model not loaded - running in simulation mode")
        
    print(f"\\n🚀 Starting server on port 5001...")
    print("🔗 Health check: http://localhost:5001/health")
    print("🎛️ Admin interface: enhanced_facial_recognition.html")
    print("\\nPress Ctrl+C to stop the server")
    print("="*60)
    
    try:
        app.run(debug=False, port=5001, host='0.0.0.0', use_reloader=False)
    except KeyboardInterrupt:
        print("\\n👋 Server stopped")
    except Exception as e:
        print(f"\\n❌ Server error: {{e}}")
'''
    
    with open('run_facial_server.py', 'w') as f:
        f.write(server_code)
    
    print(f"✅ Generated configured server: run_facial_server.py")

def create_batch_files():
    """Create batch files for Windows users"""
    
    # Windows batch file
    batch_content = '''@echo off
echo Starting Enhanced Facial Recognition Server...
python run_facial_server.py
pause
'''
    with open('start_server.bat', 'w') as f:
        f.write(batch_content)
    
    # Shell script for Unix/Linux/Mac
    shell_content = '''#!/bin/bash
echo "Starting Enhanced Facial Recognition Server..."
python3 run_facial_server.py
'''
    with open('start_server.sh', 'w') as f:
        f.write(shell_content)
    
    # Make shell script executable
    try:
        os.chmod('start_server.sh', 0o755)
    except:
        pass
    
    print("✅ Created startup scripts:")
    print("   Windows: start_server.bat")
    print("   Unix/Linux/Mac: start_server.sh")

def main():
    """Main configuration function"""
    
    try:
        # Check if configuration already exists
        if os.path.exists('facial_recognition_config.json'):
            use_existing = input("Configuration file exists. Use existing config? (Y/n): ").strip().lower()
            if use_existing != 'n':
                with open('facial_recognition_config.json', 'r') as f:
                    config = json.load(f)
                print("📋 Using existing configuration")
            else:
                config = create_config()
        else:
            config = create_config()
        
        # Generate server script
        print("\n🔨 Generating configured server script...")
        generate_server_script(config)
        
        # Create batch files
        print("\n📜 Creating startup scripts...")
        create_batch_files()
        
        print("\n✅ Setup complete!")
        print("\n🚀 To start the server:")
        print("   Method 1: python run_facial_server.py")
        print("   Method 2: Double-click start_server.bat (Windows)")
        print("   Method 3: ./start_server.sh (Unix/Linux/Mac)")
        
        # Ask if user wants to start now
        start_now = input("\nStart the server now? (Y/n): ").strip().lower()
        if start_now != 'n':
            print("\n🔄 Starting server...")
            os.system('python run_facial_server.py')
            
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())