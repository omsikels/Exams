#!/usr/bin/env python3
"""
Enhanced Facial Recognition Server Startup Script
Automatically configures paths and starts the server with real NKF model
"""

import os
import sys
import subprocess
from pathlib import Path

def find_project_paths():
    """Find the correct paths for models and videos"""
    current_dir = Path.cwd()
    
    # Common paths to check
    possible_video_paths = [
        current_dir / "videos",
        current_dir / ".." / "videos", 
        Path("C:/xampp/htdocs/exam/videos"),
        current_dir / "exam" / "videos"
    ]
    
    possible_model_paths = [
        current_dir / "checkpoints_fer_IR50_Resnet" / "best_nkf_rafdb.pth",
        current_dir / ".." / "checkpoints_fer_IR50_Resnet" / "best_nkf_rafdb.pth",
        Path("C:/xampp/htdocs/exam/ml_services/checkpoints_fer_IR50_Resnet/best_nkf_rafdb.pth"),
        current_dir / "ml_services" / "checkpoints_fer_IR50_Resnet" / "best_nkf_rafdb.pth"
    ]
    
    possible_fan_paths = [
        current_dir / "checkpoints_rafdb_landmarks" / "best_fan_rafdb.pth",
        current_dir / ".." / "checkpoints_rafdb_landmarks" / "best_fan_rafdb.pth", 
        Path("C:/xampp/htdocs/exam/ml_services/checkpoints_rafdb_landmarks/best_fan_rafdb.pth"),
        current_dir / "ml_services" / "checkpoints_rafdb_landmarks" / "best_fan_rafdb.pth"
    ]
    
    # Find paths
    videos_path = None
    for path in possible_video_paths:
        if path.exists():
            videos_path = str(path)
            break
    
    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = str(path)
            break
            
    fan_path = None
    for path in possible_fan_paths:
        if path.exists():
            fan_path = str(path)
            break
    
    return videos_path, model_path, fan_path

def create_configured_server(videos_path, model_path, fan_path):
    """Create the server script with correct paths"""
    
    server_template = '''#!/usr/bin/env python3'''
