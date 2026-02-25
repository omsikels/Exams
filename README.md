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

> **Exam 3000** is a comprehensive, web-based examination system integrated with a state-of-the-art Machine Learning backend. It is designed to administer secure exams while simultaneously recording and analyzing student facial expressions and emotions to monitor testing integrity.

---

## 🌟 What Does This Platform Do?

| Experience | Description |
| --- | --- |
| 🧑‍🎓 **The Student** | Students log in and take a timed, multiple-choice exam. The system securely records their webcam feed in the background. The video is chunked and saved directly to the server, organized neatly by the student's name. |
| 👨‍💻 **The Admin** | Administrators have access to a secure dashboard to manage exam questions, view student directories, and run the **Enhanced Facial Recognition** engine. The admin panel processes the videos to generate statistical emotion timelines. |

---

## 🤖 Machine Learning Architecture

To monitor student behavior, this platform utilizes a custom-built **Neural Keypoint Features (NKF)** Deep Learning model. Instead of just looking at the whole face, it specifically tracks the movement of microscopic facial muscles.

### The 4-Stage AI Pipeline

| Stage | Process | Technology Used |
| --- | --- | --- |
| **1** | **Face Detection:** Locates and crops the student's face from the video frames. | OpenCV (Haar Cascades) |
| **2** | **Landmark Alignment:** Maps exactly **68 distinct facial landmarks** (eyes, nose, mouth corners, jawline) on the student's face. | FAN / MobileFaceNet |
| **3** | **Feature Extraction:** Extracts global image features, focusing specifically on how those 68 landmarks are moving. | ResNet50 & RKFA |
| **4** | **Classification:** Combines features to classify the expression into one of 7 core emotions. | PyTorch Tensors |

**Supported Emotion Classifications:**

> `🟢 Happiness` | `🔵 Neutral` | `🟡 Surprise` | `🟠 Sadness` | `🔴 Anger` | `🟣 Fear` | `🟤 Disgust`

---

## 🚀 Getting Started (How to Run)

We have built a fully automated startup script. You do not need to use the command line or manually install software.

### Running the Application

1. Open the main project folder in File Explorer.
2. Double-click the <kbd>Start_Exam_System.bat</kbd> file.
3. **Allow Administrator Privileges:** The script will ask for Admin rights. Click <kbd>Yes</kbd>.
4. **Wait:** The script will automatically install Node.js, Python, and all required AI dependencies (`PyTorch`, `OpenCV`, etc.).
5. **Ready!** The script will automatically open your web browser to the correct portals.

---

## 🔐 System Access

Once the startup script finishes, the following URLs will be automatically available:

* 📝 **Student Exam Portal:** `http://localhost:3000/index.html`
* ⚙️ **Admin Login Portal:** `http://localhost:3000/login.html`

**Default Admin Credentials:**

> **Username:** `admin` 
> 
> 
> 
> 
> **Password:** `exam3000`

---

## 📁 Advanced Info: Directory Structure

<details>
<summary><b>🖱️ Click to expand the folder architecture</b></summary>




Below is the visual map of how the system organizes data and logic:

```text
📦 Exam3000-Project
 ┣ 📂 public/              # Frontend UI files (HTML/CSS/JS)
 ┣ 📂 videos/              # 🎥 Auto-generated student recording folders
 ┣ 📂 ml_services/         # 🧠 Python AI Backend
 ┃ ┣ 📜 enhanced_facial_recognition_server.py
 ┃ ┗ 📂 checkpoints/       # Trained .pth model weights
 ┣ 📂 facial_results/      # 📊 Exported AI analysis JSON reports
 ┣ 📜 server.js            # ⚙️ Main Node.js Web Server
 ┗ 📜 questions.json       # 📝 Exam database

```

### Folder Details

| Directory / File | Purpose |
| --- | --- |
| 📁 `/videos/` | Where all student webcam recordings are saved. Each student gets their own sub-folder automatically. |
| 📁 `/ml_services/` | Contains the Python ML server and the trained AI models (`.pth` files). |
| 📁 `/facial_results/` | Contains the exported JSON reports generated after the Admin processes the videos. |
| 📄 `questions.json` | The database file storing the exam questions, choices, and time limits. |

</details>
