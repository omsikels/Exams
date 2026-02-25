🎓 Exam 3000: AI-Proctored Examination Platform
Welcome to the Exam 3000 platform. This project is a comprehensive, web-based examination system integrated with a state-of-the-art Machine Learning backend. It is designed to administer secure exams while simultaneously recording and analyzing student facial expressions and emotions to monitor testing integrity.

🌟 What Does This Website Do?
This platform is divided into two main experiences:

The Student Experience: Students log in and take a timed, multiple-choice examination. While they answer questions, the system securely records their webcam feed in the background. The video is chunked and saved directly to the server, organized neatly by the student's name.

The Admin Experience: Administrators have access to a secure dashboard where they can manage exam questions, view student directories, and run the Enhanced Facial Recognition engine. The admin panel processes the students' recorded videos to generate statistical timelines of their emotions during the exam.

🤖 Machine Learning Architecture
To monitor student behavior, this platform utilizes a custom-built Neural Keypoint Features (NKF) Deep Learning model. Instead of just looking at the whole face, it specifically tracks the movement of microscopic facial muscles to determine exact emotions.

The ML pipeline operates in four stages:

Face Detection: Uses OpenCV (Haar Cascades) to locate and crop the student's face from the video frames.

Facial Landmark Alignment: Uses a FAN (Face Alignment Network) powered by a lightweight MobileFaceNet backbone. This maps exactly 68 distinct facial landmarks (eyes, nose, mouth corners, jawline) on the student's face.

Feature Extraction & Attention: A powerful ResNet50 backbone extracts global image features, while an RKFA (Representative Keypoint Feature Attention) module focuses specifically on how those 68 landmarks are moving.

Emotion Classification: The network combines these features to classify the student's expression into one of 7 core emotions:

🟢 Happiness

🔵 Neutral

🟡 Surprise

🟠 Sadness

🔴 Anger

🟣 Fear

🟤 Disgust

The system outputs a detailed JSON report and a visual data table showing the primary emotion, face detection rates, confidence scores, and a second-by-second timeline of the student's state.

🚀 Getting Started (How to Run)
We have built a fully automated startup script so that clients and administrators do not need to use the command line or manually install software.

Prerequisites
A Windows computer.

An active internet connection (only required for the very first setup to download dependencies).

A working webcam.

Running the Application
Open the project folder.

Double-click the Start_Exam_System.bat file (or whatever your .bat file is named).

Allow Administrator Privileges: The script will automatically ask for Admin rights. Click "Yes". This is required to install the necessary software.

Wait: The script will automatically:

Check if Node.js is installed (and download/install it silently if missing).

Check if Python 3.11 is installed (and download/install it silently if missing).

Install all required Web dependencies (npm install).

Install all required AI dependencies (PyTorch, OpenCV, Flask, etc.).

Start the Web Server (Port 3000) and the AI Server (Port 5001).

Ready! The script will automatically open your web browser to the correct pages.

⚠️ IMPORTANT: A black console window will remain open in the background. Do not close this window while taking exams or processing videos, as it keeps the servers running. You may minimize it.

🔐 System Access
Once the startup script finishes, the following URLs will be available:

Student Exam Portal: http://localhost:3000/index.html

Admin Login Portal: http://localhost:3000/login.html

Default Admin Credentials:

Username: admin

Password: exam3000

📁 Directory Structure
If you need to manually locate files, the system organizes them as follows:

/videos/ - This is where all student webcam recordings are automatically saved. Each student gets their own sub-folder.

/ml_services/ - Contains the Python Machine Learning server (enhanced_facial_recognition_server.py) and the trained AI models (.pth files).

/facial_results/ - Contains the exported JSON reports generated after the Admin runs the facial recognition processor.

questions.json - The database file storing the exam questions and time limits.
