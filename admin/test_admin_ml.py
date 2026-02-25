#!/usr/bin/env python3
"""
Admin Test Integration Script
Test the admin-only ML processing system.
"""

import sys
import time
import requests
import tempfile
import json
from pathlib import Path

class AdminMLTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.admin_authenticated = False
    
    def admin_login(self, username="admin", password="exam3000"):
        """Login as admin to the ML service."""
        print("🔐 Admin login...")
        try:
            response = self.session.post(f"{self.base_url}/admin/login", 
                                       json={"username": username, "password": password})
            
            if response.status_code == 200:
                self.admin_authenticated = True
                print("   ✅ Admin authentication successful")
                return True
            else:
                print(f"   ❌ Admin login failed: {response.json().get('message', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"   ❌ Admin login error: {e}")
            return False
    
    def test_admin_health(self):
        """Test admin health check."""
        print("🔍 Testing admin health check...")
        try:
            response = self.session.get(f"{self.base_url}/admin/health")
            
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Admin health check passed")
                print(f"   📊 Service: {data['service']}")
                print(f"   👤 Admin: {data.get('admin_session', 'Unknown')}")
                print(f"   📹 Processed videos: {data.get('processed_videos', 0)}")
                return True
            else:
                print(f"   ❌ Admin health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Admin health check error: {e}")
            return False
    
    def test_admin_video_processing(self):
        """Test admin video processing."""
        print("🎬 Testing admin video processing...")
        try:
            # Create mock video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as mock_video:
                mock_video.write(b"mock admin video content for testing")
                mock_video.flush()
                
                # Prepare test data
                files = {'video': open(mock_video.name, 'rb')}
                data = {
                    'username': 'test_student_admin',
                    'questionIndex': '1',
                    'result': 'Correct'
                }
                
                response = self.session.post(f"{self.base_url}/admin/process-video", 
                                          files=files, data=data)
                files['video'].close()
                
                if response.status_code == 200:
                    result = response.json()
                    print("   ✅ Admin video processing successful")
                    print(f"   📝 Status: {result['status']}")
                    print(f"   📹 Message: {result['message']}")
                    
                    if 'emotions_detected' in result:
                        emotions = result['emotions_detected']
                        print(f"   🎭 Primary Emotion: {emotions['primary_emotion']}")
                        print(f"   📊 Segments: {emotions['summary']['total_segments']}")
                        print(f"   🔐 Admin processed: {result.get('admin_processed', False)}")
                    
                    return True
                else:
                    print(f"   ❌ Admin video processing failed: {response.status_code}")
                    print(f"   📄 Response: {response.text}")
                    return False
                    
                # Clean up
                Path(mock_video.name).unlink(missing_ok=True)
                
        except Exception as e:
            print(f"   ❌ Admin video processing error: {e}")
            return False
    
    def test_admin_students_list(self):
        """Test admin students listing."""
        print("📚 Testing admin students list...")
        try:
            response = self.session.get(f"{self.base_url}/admin/students")
            
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Admin students list successful")
                print(f"   👥 Total students: {data.get('total_students', 0)}")
                
                if data.get('students'):
                    for student, info in list(data['students'].items())[:3]:  # Show first 3
                        print(f"   📚 {student}: {info['video_count']} videos")
                
                return True
            else:
                print(f"   ❌ Admin students list failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Admin students list error: {e}")
            return False
    
    def test_admin_system_status(self):
        """Test admin system status."""
        print("🔧 Testing admin system status...")
        try:
            response = self.session.get(f"{self.base_url}/admin/system-status")
            
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Admin system status successful")
                
                stats = data.get('statistics', {})
                storage = data.get('storage', {})
                
                print(f"   👥 Students: {stats.get('total_students', 0)}")
                print(f"   📹 Videos: {stats.get('total_videos', 0)}")
                print(f"   ✅ Processed: {stats.get('processed_videos', 0)}")
                print(f"   💾 Storage: {storage.get('video_directory_mb', 0):.1f} MB")
                
                return True
            else:
                print(f"   ❌ Admin system status failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Admin system status error: {e}")
            return False
    
    def admin_logout(self):
        """Logout admin session."""
        print("🔓 Admin logout...")
        try:
            response = self.session.post(f"{self.base_url}/admin/logout")
            
            if response.status_code == 200:
                self.admin_authenticated = False
                print("   ✅ Admin logout successful")
                return True
            else:
                print(f"   ❌ Admin logout failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Admin logout error: {e}")
            return False
    
    def run_full_test(self):
        """Run complete admin ML service test."""
        print("="*60)
        print("ADMIN ML SERVICE TESTING")
        print("="*60)
        
        success = True
        
        # Step 1: Admin Login
        if not self.admin_login():
            print("❌ Cannot continue without admin authentication")
            return False
        
        # Step 2: Health Check
        if not self.test_admin_health():
            success = False
        
        # Step 3: Video Processing
        if not self.test_admin_video_processing():
            success = False
        
        # Step 4: Students List
        if not self.test_admin_students_list():
            success = False
        
        # Step 5: System Status
        if not self.test_admin_system_status():
            success = False
        
        # Step 6: Logout
        if not self.admin_logout():
            success = False
        
        # Final Result
        print("\n" + "="*60)
        if success:
            print("🎉 ALL ADMIN TESTS PASSED!")
            print("The admin ML processing system is working correctly.")
        else:
            print("⚠️  SOME ADMIN TESTS FAILED!")
            print("Please check the error messages above.")
        print("="*60)
        
        return success

def main():
    """Run admin ML service tests."""
    print("Starting Admin ML Processing Service Tests...\n")
    
    print("Make sure the admin ML service is running:")
    print("  python ml_service/admin_video_server.py")
    print()
    
    # Ask user if ready
    ready = input("Is the admin ML service running? (y/N): ").lower().strip()
    if ready != 'y':
        print("Please start the admin ML service first.")
        return 1
    
    # Run tests
    tester = AdminMLTester()
    success = tester.run_full_test()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())