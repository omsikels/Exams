<p align="center">
<h1 align="center">🎓 Exam 3000: AI-Proctored Examination Platform</h1>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Node.js-20.11-43853D?style=for-the-badge&logo=node.js&logoColor=white" alt="Node.js">
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/OpenCV-4.12-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/Flask-Enabled-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
</p>

Exam 3000 is a comprehensive, web-based examination system integrated with a state-of-the-art Machine Learning backend. It is designed to administer secure exams while simultaneously recording and analyzing student facial expressions and emotions to monitor testing integrity.

🌟 What Does This Platform Do?
This platform is divided into two distinct experiences:

🧑‍🎓 The Student Experience: Students log in and take a timed, multiple-choice examination. While they answer questions, the system securely records their webcam feed in the background. The video is chunked and saved directly to the server, organized neatly by the student's name.

👨‍💻 The Admin Experience: Administrators have access to a secure dashboard where they can manage exam questions, view student directories, and run the Enhanced Facial Recognition engine. The admin panel processes the students' recorded videos to generate statistical timelines of their emotions during the exam.

🤖 Machine Learning Architecture
To monitor student behavior, this platform utilizes a custom-built Neural Keypoint Features (NKF) Deep Learning model. Instead of just looking at the whole face, it specifically tracks the movement of microscopic facial muscles to determine exact emotions.

The ML pipeline operates in four distinct stages:

Face Detection: Uses OpenCV (Haar Cascades) to locate and crop the student's face from the video frames.

Facial Landmark Alignment: Uses a FAN (Face Alignment Network) powered by a lightweight MobileFaceNet backbone. This maps exactly 68 distinct facial landmarks (eyes, nose, mouth corners, jawline) on the student's face.

Feature Extraction & Attention: A powerful ResNet50 backbone extracts global image features, while an RKFA (Representative Keypoint Feature Attention) module focuses specifically on how those 68 landmarks are moving.

Emotion Classification: The network combines these features to classify the student's expression into one of 7 core emotions:
Happiness | Neutral | Surprise | Sadness | Anger | Fear | Disgust

The system outputs a detailed JSON report and a visual data table showing the primary emotion, face detection rates, confidence scores, and a second-by-second timeline of the student's state.

🚀 Getting Started (How to Run)
We have built a fully automated startup script so that clients and administrators do not need to use the command line or manually install software.

Prerequisites
A Windows computer.

An active internet connection (only required for the very first setup to download dependencies).

A working webcam.

Running the Application
Open the main project folder.

Double-click the Start_Exam_System.bat file.

Allow Administrator Privileges: The script will automatically ask for Admin rights. Click "Yes". This is required to install the necessary software.

Wait: The script will automatically:

Check if Node.js & Python are installed (and install them silently if missing).

Install all required Web and AI dependencies (npm install, PyTorch, OpenCV, etc.).

Start the Web Server (Port 3000) and the AI Server (Port 5001).

Ready! The script will automatically open your web browser to the correct pages.

🔐 System Access
Once the startup script finishes, the following URLs will be automatically available:

Student Exam Portal: http://localhost:3000/index.html

Admin Login Portal: http://localhost:3000/login.html

Default Admin Credentials:

Username: admin

Password: exam3000

📁 Advanced Info: Directory Structure
<details>
<summary><b>Click to expand the folder architecture</b></summary>



If you need to manually locate files, the system organizes them as follows:

/videos/ - This is where all student webcam recordings are automatically saved. Each student gets their own sub-folder.

/ml_services/ - Contains the Python Machine Learning server (enhanced_facial_recognition_server.py) and the trained AI models (.pth files).

/facial_results/ - Contains the exported JSON reports generated after the Admin runs the facial recognition processor.

questions.json - The database file storing the exam questions and time limits.

</details>
