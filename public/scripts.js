document.addEventListener("DOMContentLoaded", () => {

    function loadHTML(id, file) {
        fetch(file)
            .then(res => res.text())
            .then(data => document.getElementById(id).innerHTML = data)
            .catch(err => console.error(err));
    }

    loadHTML("header", "header.html");
    loadHTML("sidebar", "sidebar.html");
    loadHTML("footer", "footer.html");

    setTimeout(() => {
        const sidebarWrapper = document.getElementById("sidebar-wrapper");

        // Toggle sidebar
        document.addEventListener("click", (e) => {
            if (e.target.id === "menuBtn") {
                sidebarWrapper.classList.toggle("active");
            }
        });

        // Initialize Enhanced Facial Recognition Manager after sidebar is loaded
        if (window.facialManager) {
            window.facialManager.init();
        }
    }, 150);
});

// Enhanced Facial Recognition Manager Class
class EnhancedFacialRecognitionManager {
    constructor() {
        this.baseURL = 'http://localhost:5001';
        this.videos = {};
        this.processing = false;
        this.serverOnline = false;
        this.modelLoaded = false;
        this.expandedStudent = null;
        
        // For backward compatibility with existing folder system
        this.folderData = [
            { name: "Folder 1", videos: ["video1.mp4", "video2.mp4"] },
            { name: "Folder 2", videos: ["video3.mp4", "video4.mp4"] },
            { name: "Folder 3", videos: ["video5.mp4"] }
        ];
    }

    init() {
        this.bindEvents();
        this.checkServerStatus();
        setInterval(() => this.checkServerStatus(), 30000);
    }

    bindEvents() {
        // Enhanced functionality
        this.bindEnhancedEvents();
        
        // Backward compatibility - keep existing behavior as fallback
        this.bindBackwardCompatibilityEvents();
    }

    bindEnhancedEvents() {
        const checkBtn = document.getElementById('checkServerBtn');
        const loadBtn = document.getElementById('loadVideosBtn');
        const processBtn = document.getElementById('processAllBtn');
        const viewBtn = document.getElementById('viewResultsBtn');
        const exportBtn = document.getElementById('exportResultsBtn');

        if (checkBtn) checkBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.checkServerStatus();
        });
        
        if (loadBtn) loadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.loadVideos();
        });
        
        if (processBtn) processBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.processAllVideos();
        });
        
        if (viewBtn) viewBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.viewResults();
        });
        
        if (exportBtn) exportBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.exportResults();
        });
    }

    bindBackwardCompatibilityEvents() {
        // Keep existing folder rendering as fallback
        const loadBtn = document.getElementById('loadVideosBtn');
        if (loadBtn) {
            loadBtn.addEventListener('click', () => {
                // If server is offline, fall back to demo folder display
                if (!this.serverOnline) {
                    this.renderFoldersCompatibility();
                }
            });
        }
    }

    async checkServerStatus() {
        const statusElement = document.getElementById('serverStatus');
        const statusTextElement = document.getElementById('serverStatusText');
        const connectionInfo = document.getElementById('connectionInfo');
        const modelInfo = document.getElementById('modelInfo');

        this.log('🔍 Checking enhanced server status...');

        try {
            const response = await fetch(`${this.baseURL}/health`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' },
                signal: AbortSignal.timeout(10000)
            });

            if (response.ok) {
                const data = await response.json();
                this.serverOnline = true;
                this.modelLoaded = data.model_loaded;

                if (statusElement) {
                    statusElement.className = 'server-status status-online';
                    statusElement.innerHTML = '✅ Online';
                }
                
                if (statusTextElement) {
                    statusTextElement.innerHTML = this.modelLoaded ? '✅ Ready' : '⚠️ No Model';
                }

                if (document.getElementById('modelType')) {
                    document.getElementById('modelType').textContent = data.mode || 'Enhanced';
                }

                if (connectionInfo) {
                    connectionInfo.innerHTML = `
                        <h4>🔗 Server Connection - Online</h4>
                        <p>✅ Connected to ${data.service}</p>
                        <p>🤖 Model Status: ${this.modelLoaded ? 'Loaded ✅' : 'Not Loaded ⚠️'}</p>
                        <p>🎭 Detection: ${data.face_detection || 'OpenCV'}</p>
                        <p>💻 Device: ${data.device || 'CPU'}</p>
                    `;
                    connectionInfo.className = 'connection-info';
                    connectionInfo.style.background = 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)';
                    connectionInfo.style.borderColor = '#28a745';
                }

                if (modelInfo && this.modelLoaded) {
                    modelInfo.innerHTML = `
                        <h4>🤖 NKF Model Information</h4>
                        <p>✅ Enhanced Neural Keypoint Features (NKF) model is loaded</p>
                        <p>🎯 Real emotion detection with landmark-based features</p>
                        <p>📊 7 emotion classes: Happiness, Sadness, Anger, Fear, Surprise, Disgust, Neutral</p>
                        <p>💻 Running on: ${data.device || 'CPU'}</p>
                    `;
                    modelInfo.classList.remove('hidden');
                }

                const loadVideosBtn = document.getElementById('loadVideosBtn');
                if (loadVideosBtn) loadVideosBtn.disabled = false;

                this.log('✅ Enhanced server is online and ready');
                return true;
            } else {
                throw new Error(`Server responded with status ${response.status}`);
            }
        } catch (error) {
            this.serverOnline = false;
            this.modelLoaded = false;

            if (statusElement) {
                statusElement.className = 'server-status status-offline';
                statusElement.innerHTML = '❌ Offline';
            }
            
            if (statusTextElement) {
                statusTextElement.innerHTML = '❌ Offline';
            }

            if (connectionInfo) {
                connectionInfo.innerHTML = `
                    <h4>🔗 Server Connection - Offline</h4>
                    <p>❌ Cannot connect to enhanced facial recognition server</p>
                    <p>🔧 Start the server with one of these commands:</p>
                    <pre style="background: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 8px; font-size: 14px;">
python enhanced_facial_recognition_server.py
# OR use the auto-configurator:
python start_facial_server.py</pre>
                    <p>⚠️ Error: ${error.message}</p>
                `;
                connectionInfo.className = 'error-message';
            }

            if (modelInfo) {
                modelInfo.classList.add('hidden');
            }

            const loadVideosBtn = document.getElementById('loadVideosBtn');
            const processAllBtn = document.getElementById('processAllBtn');
            if (loadVideosBtn) loadVideosBtn.disabled = true;
            if (processAllBtn) processAllBtn.disabled = true;

            this.log(`❌ Server connection failed: ${error.message}`);
            return false;
        }
    }

    async loadVideos() {
        if (!this.serverOnline) {
            // Fall back to demo folders if server is offline
            this.log('⚠️ Server offline. Showing demo folders...');
            this.renderFoldersCompatibility();
            return;
        }

        this.log('📁 Loading videos from server...');

        try {
            const response = await fetch(`${this.baseURL}/api/get-videos`);

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const videos = await response.json();

            this.videos = videos;
            this.displayStudentsList(videos);
            this.updateStats();

            const studentCount = Object.keys(videos).length;
            const videoCount = Object.values(videos).reduce((sum, studentVideos) => sum + studentVideos.length, 0);

            this.log(`✅ Loaded ${videoCount} videos for ${studentCount} students`);

            if (videoCount > 0 && this.modelLoaded) {
                const processAllBtn = document.getElementById('processAllBtn');
                if (processAllBtn) processAllBtn.disabled = false;
            }

        } catch (error) {
            this.log(`❌ Error loading videos: ${error.message}`);
            // Fall back to demo folders
            this.renderFoldersCompatibility();
        }
    }

    displayStudentsList(videos) {
        const studentsContainer = document.getElementById('studentsContainer');
        const videosSection = document.getElementById('videosSection');

        if (!studentsContainer) return;

        studentsContainer.innerHTML = '';

        if (Object.keys(videos).length === 0) {
            if (videosSection) videosSection.classList.add('hidden');
            return;
        }

        Object.entries(videos).forEach(([studentName, studentVideos]) => {
            const totalSize = studentVideos.reduce((sum, v) => sum + v.size, 0);
            const avgSizeMB = (totalSize / (1024 * 1024) / studentVideos.length).toFixed(1);

            const studentItem = document.createElement('div');
            studentItem.className = 'student-item';
            studentItem.dataset.student = studentName;

            studentItem.innerHTML = `
                <div class="student-header" onclick="facialManager.toggleStudent('${studentName}')">
                    <div class="student-info">
                        <div class="student-avatar">${studentName.charAt(0).toUpperCase()}</div>
                        <div class="student-details">
                            <h4>👤 ${studentName}</h4>
                            <div class="student-meta">
                                <span><strong>${studentVideos.length}</strong> videos</span>
                                <span>Total: <strong>${this.formatFileSize(totalSize)}</strong></span>
                                <span>Avg: <strong>${avgSizeMB} MB</strong></span>
                            </div>
                        </div>
                    </div>
                    <div class="expand-icon">▼</div>
                </div>
                <div class="student-videos">
                    <div class="video-grid">
                        ${studentVideos.map(video => `
                            <div class="video-card">
                                <div class="video-info">
                                    <div class="video-details">
                                        <h6>📄 ${video.filename}</h6>
                                        <div class="video-size">${this.formatFileSize(video.size)}</div>
                                    </div>
                                </div>
                                <div class="video-actions">
                                    <button class="btn btn-small" 
                                            onclick="facialManager.processSingleVideo('${studentName}', '${video.filename}')"
                                            ${!this.modelLoaded ? 'disabled title="Model not loaded"' : ''}>
                                        🎭 Process
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;

            studentsContainer.appendChild(studentItem);
        });

        if (videosSection) videosSection.classList.remove('hidden');
    }

    // Backward compatibility function - renders demo folders in the old style
    renderFoldersCompatibility() {
        const studentsContainer = document.getElementById('studentsContainer');
        if (!studentsContainer) return;

        studentsContainer.innerHTML = '';

        this.folderData.forEach((folder, index) => {
            const folderDiv = document.createElement('div');
            folderDiv.classList.add('folder');

            const title = document.createElement('div');
            title.classList.add('folder-title');
            title.textContent = folder.name;

            const content = document.createElement('div');
            content.classList.add('folder-content');

            folder.videos.forEach(video => {
                const videoLink = document.createElement('a');
                videoLink.href = '#';
                videoLink.textContent = video;
                videoLink.style.display = 'block';
                videoLink.style.marginBottom = '5px';

                videoLink.addEventListener('click', e => {
                    e.preventDefault();
                    this.log(`📹 Demo video clicked: ${video}`);
                    alert("Demo video: " + video + "\n\nConnect to server for real functionality.");
                });

                content.appendChild(videoLink);
            });

            folderDiv.appendChild(title);
            folderDiv.appendChild(content);
            studentsContainer.appendChild(folderDiv);

            // Accordion click
            title.addEventListener("click", () => {
                // Close other folders
                document.querySelectorAll(".folder").forEach(f => {
                    if (f !== folderDiv) f.classList.remove("active");
                });
                // Toggle current folder
                folderDiv.classList.toggle("active");
            });
        });

        this.log('📁 Loaded demo folders (server offline)');
    }

    toggleStudent(studentName) {
    const targetStudent = document.querySelector(`[data-student="${studentName}"]`);
    if (!targetStudent) return;

    const isCurrentlyExpanded = targetStudent.classList.contains('expanded');
    
    // Close ALL students first
    document.querySelectorAll('.student-item').forEach(student => {
        student.classList.remove('expanded');
    });
    
    // If the clicked student wasn't expanded, expand it
    if (!isCurrentlyExpanded) {
        targetStudent.classList.add('expanded');
        this.expandedStudent = studentName;
        this.log(`📂 Expanded ${studentName}'s videos`);
    } else {
        this.expandedStudent = null;
        this.log(`📁 Collapsed ${studentName}'s videos`);
    }
}

    updateStats() {
        const totalStudents = Object.keys(this.videos).length;
        const totalVideos = Object.values(this.videos).reduce((sum, videos) => sum + videos.length, 0);

        const totalStudentsElement = document.getElementById('totalStudents');
        const totalVideosElement = document.getElementById('totalVideos');
        
        if (totalStudentsElement) totalStudentsElement.textContent = totalStudents;
        if (totalVideosElement) totalVideosElement.textContent = totalVideos;
    }

    async processSingleVideo(studentName, videoFilename) {
        if (!this.serverOnline || !this.modelLoaded) {
            alert('Server offline or model not loaded. Please check server status.');
            return;
        }

        this.log(`🎬 Processing ${studentName}/${videoFilename} with NKF model...`);

        try {
            const response = await fetch(`${this.baseURL}/api/process-video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ student_name: studentName, video_filename: videoFilename })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const result = await response.json();

            if (result.status === 'success') {
                const detectionRate = result.face_detection_rate ? (result.face_detection_rate * 100).toFixed(1) + '%' : 'N/A';
                this.log(`✅ ${studentName}/${videoFilename}: ${result.primary_emotion} (${detectionRate} face detection)`);

                if (result.timeline && result.timeline.length > 0) {
                    this.log(`   📈 ${result.timeline.length} emotion segments detected`);
                    this.log(`   ⏱️ Duration: ${result.duration}s`);
                }
            } else {
                this.log(`❌ ${studentName}/${videoFilename}: ${result.error || 'Processing failed'}`);
            }

        } catch (error) {
            this.log(`❌ Error processing ${studentName}/${videoFilename}: ${error.message}`);
            alert(`Processing failed: ${error.message}`);
        }
    }

    async processAllVideos() {
        if (this.processing) return;
        if (!this.serverOnline || !this.modelLoaded) {
            alert('Server offline or model not loaded. Please check server status.');
            return;
        }

        this.processing = true;
        const processAllBtn = document.getElementById('processAllBtn');
        if (processAllBtn) processAllBtn.disabled = true;

        this.log('🚀 Starting batch processing with NKF model...');
        this.updateProgress(0, 'Initializing batch processing...');

        try {
            const response = await fetch(`${this.baseURL}/api/process-all`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const results = await response.json();

            this.updateProgress(100, 'Processing complete!', `Processed ${results.total_videos} videos: ${results.successful} successful, ${results.failed} failed`);

            this.log(`✅ Batch processing complete!`);
            this.log(`📊 Total: ${results.total_videos}, Success: ${results.successful}, Failed: ${results.failed}`);

            if (results.model_type) this.log(`🤖 Model used: ${results.model_type}`);

            this.displayBatchResults(results);

        } catch (error) {
            this.log(`❌ Batch processing error: ${error.message}`);
            this.updateProgress(0, 'Error occurred', error.message);
            alert(`Batch processing failed: ${error.message}`);
        } finally {
            this.processing = false;
            if (processAllBtn) processAllBtn.disabled = false;
        }
    }

    displayBatchResults(results) {
        const resultsContent = document.getElementById('resultsContent');
        const resultsSection = document.getElementById('resultsSection');

        if (!resultsContent || !resultsSection) return;

        // header / stats html
        let html = `
            <div style="margin-bottom: 25px;">
                <h4>📊 Enhanced NKF Processing Results</h4>
                <div class="video-stats">
                    <div class="stat-item">
                        <div class="stat-value">${results.total_videos}</div>
                        <div class="stat-label">Total Videos</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="color: #27ae60;">${results.successful}</div>
                        <div class="stat-label">Successful</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="color: #e74c3c;">${results.failed}</div>
                        <div class="stat-label">Failed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${results.model_type || 'NKF'}</div>
                        <div class="stat-label">Model Used</div>
                    </div>
                </div>
                <p><small>Processed: ${new Date(results.processed_at).toLocaleString()}</small></p>
            </div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Student</th>
                        <th>Video</th>
                        <th>Status</th>
                        <th>Primary Emotion</th>
                        <th>Duration</th>
                        <th>Face Detection</th>
                        <th>Confidence</th>
                        <th>Segments</th>
                    </tr>
                </thead>
                <tbody>
        `;

        // GROUP BY student (frontend only)
        const grouped = {};
        results.results.forEach(r => {
            if (!grouped[r.student]) {
                grouped[r.student] = {
                    student: r.student,
                    entries: [],
                    success_videos: [],
                    failed_videos: [],
                    primary_emotion_counts: {},
                    total_duration: 0,
                    face_detection_values: [],
                    timeline_segments: 0,
                    per_video_avg_conf: []
                };
            }

            const g = grouped[r.student];
            g.entries.push(r);

            // categorize video
            if (r.status === 'success') g.success_videos.push(r.video);
            else { g.failed_videos.push(r.video); }

            // count primary emotion occurrences
            if (r.primary_emotion) {
                g.primary_emotion_counts[r.primary_emotion] = (g.primary_emotion_counts[r.primary_emotion] || 0) + 1;
            }

            // sum duration if present
            if (typeof r.duration === 'number') g.total_duration += r.duration;

            // face detection rates (collect)
            if (typeof r.face_detection_rate === 'number') g.face_detection_values.push(r.face_detection_rate);

            // segments count
            if (Array.isArray(r.timeline)) g.timeline_segments += r.timeline.length;

            // compute per-video average confidence (if timeline present)
            if (Array.isArray(r.timeline) && r.timeline.length > 0) {
                const avgConf = r.timeline.reduce((s, seg) => s + (seg.confidence || 0), 0) / r.timeline.length;
                g.per_video_avg_conf.push(avgConf);
            }
        });

        // Build merged rows
        const mergedResults = Object.values(grouped);

        mergedResults.forEach(g => {
            // Videos column logic:
            let videosDisplay = '-';
            if (g.failed_videos.length > 0) {
                videosDisplay = 'Error: ' + g.failed_videos.join(', ');
            } else if (g.success_videos.length > 1) {
                videosDisplay = 'All Videos';
            } else if (g.success_videos.length === 1) {
                videosDisplay = g.success_videos[0];
            } else {
                // fallback: if no success but entries exist (unlikely), show entries filenames
                videosDisplay = g.entries.map(e => e.video).join(', ');
            }

            // Status: if any failure, folder = failed else success
            const status = g.failed_videos.length > 0 ? 'failed' : 'success';
            const statusColor = status === 'success' ? '#27ae60' : '#e74c3c';

            // Primary emotion: most frequent among primary_emotion_counts
            let primaryEmotion = '-';
            const emotionEntries = Object.entries(g.primary_emotion_counts);
            if (emotionEntries.length > 0) {
                emotionEntries.sort((a, b) => b[1] - a[1]);
                primaryEmotion = emotionEntries[0][0];
            }

            // Duration: total duration (sum of durations)
            const durationText = g.total_duration ? parseFloat(g.total_duration.toFixed(2)) + 's' : '-';

            // Face detection: average across videos (if any)
            let faceDetectText = '-';
            if (g.face_detection_values.length > 0) {
                const avgFace = g.face_detection_values.reduce((a,b) => a + b, 0) / g.face_detection_values.length;
                faceDetectText = (avgFace * 100).toFixed(1) + '%';
            }

            // Confidence: average of per-video average confidences
            let confidenceText = '-';
            if (g.per_video_avg_conf.length > 0) {
                const avgConfAcrossVideos = g.per_video_avg_conf.reduce((a,b) => a + b, 0) / g.per_video_avg_conf.length;
                confidenceText = avgConfAcrossVideos.toFixed(3);
            }

            // Segments: total timeline segments across videos
            const segmentsText = g.timeline_segments || '-';

            // Emotion tag display (if available)
            const emotionHTML = primaryEmotion !== '-' ? `<span class="emotion-tag emotion-${primaryEmotion.toLowerCase()}">${primaryEmotion}</span>` : '-';

            html += `
                <tr>
                    <td><strong>${g.student}</strong></td>
                    <td>${videosDisplay}</td>
                    <td style="color: ${statusColor}; font-weight: bold;">${status}</td>
                    <td>${emotionHTML}</td>
                    <td>${durationText}</td>
                    <td>${faceDetectText}</td>
                    <td>${confidenceText}</td>
                    <td>${segmentsText}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';

        resultsContent.innerHTML = html;
        resultsSection.classList.remove('hidden');
    }

    async viewResults() {
        if (!this.serverOnline) {
            alert('Server is offline. Please check server connection first.');
            return;
        }

        try {
            const response = await fetch(`${this.baseURL}/api/get-results`);
            const resultsList = await response.json();

            if (resultsList.length === 0) {
                this.log('📋 No results found. Process some videos first.');
                return;
            }

            const latestResult = resultsList[0];
            const detailResponse = await fetch(`${this.baseURL}/api/get-result/${latestResult.filename}`);
            const detailData = await detailResponse.json();

            this.displayBatchResults(detailData);
            this.log(`📋 Loaded latest results: ${latestResult.filename}`);
            this.log(`   Model: ${latestResult.model_type || 'Unknown'}`);
            this.log(`   Processed: ${new Date(latestResult.processed_at).toLocaleString()}`);

        } catch (error) {
            this.log(`❌ Error loading results: ${error.message}`);
            alert(`Failed to load results: ${error.message}`);
        }
    }

    async exportResults() {
        if (!this.serverOnline) {
            alert('Server is offline. Please check server connection first.');
            return;
        }

        try {
            const response = await fetch(`${this.baseURL}/api/get-results`);
            const resultsList = await response.json();

            if (resultsList.length === 0) {
                this.log('📤 No results to export');
                alert('No results to export. Process some videos first.');
                return;
            }

            const latestResult = resultsList[0];
            const detailResponse = await fetch(`${this.baseURL}/api/get-result/${latestResult.filename}`);
            const detailData = await detailResponse.json();

            const blob = new Blob([JSON.stringify(detailData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `enhanced_facial_results_${new Date().toISOString().slice(0,10)}.json`;
            a.click();
            URL.revokeObjectURL(url);

            this.log('📤 Enhanced results exported successfully');

        } catch (error) {
            this.log(`❌ Export error: ${error.message}`);
            alert(`Export failed: ${error.message}`);
        }
    }

    updateProgress(percentage, text, details = '') {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const progressDetails = document.getElementById('progressDetails');
        
        if (progressFill) progressFill.style.width = `${percentage}%`;
        if (progressText) progressText.textContent = text;
        if (progressDetails) progressDetails.textContent = details;
    }

    log(message) {
        const logOutput = document.getElementById('logOutput');
        const logSection = document.getElementById('logSection');
        const timestamp = new Date().toLocaleTimeString();

        if (logOutput) {
            logOutput.innerHTML += `<span style="color: #3498db;">[${timestamp}]</span> ${message}<br>`;
            logOutput.scrollTop = logOutput.scrollHeight;
        }
        
        if (logSection) {
            logSection.classList.remove('hidden');
        }

        console.log(`[${timestamp}] ${message}`);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the enhanced facial recognition manager
const facialManager = new EnhancedFacialRecognitionManager();
window.facialManager = facialManager;

// Backward compatibility functions - these maintain the original behavior
function setProgress(percent, statusText) {
    facialManager.updateProgress(percent, statusText);
}

function addLog(message) {
    facialManager.log(message);
}

function renderFolders() {
    facialManager.renderFoldersCompatibility();
}

// Initialize demo behavior
setTimeout(() => {
    // Initial status
    setProgress(0, "Ready To Process");
    addLog("Enhanced Facial Recognition System initialized");
    addLog("Ready to connect to server or use demo mode");
    
    // Demo progress (only if server is offline)
    setTimeout(() => {
        if (!facialManager.serverOnline) {
            setProgress(100, "Demo Mode Active");
            addLog("Running in demo mode - connect to server for full functionality");
        }
    }, 3000);
}, 1000);

// Example folder/video data
const folderData = [
    { name: "Folder 1", videos: ["video1.mp4", "video2.mp4"] },
    { name: "Folder 2", videos: ["video3.mp4", "video4.mp4"] },
    { name: "Folder 3", videos: ["video5.mp4"] }
];

// Function to render folders inside #studentsContainer
function renderVideos() {
    const container = document.getElementById("studentsContainer");
    container.innerHTML = ""; // clear previous

    folderData.forEach(folder => {
        const folderDiv = document.createElement("div");
        folderDiv.classList.add("folder");

        const title = document.createElement("div");
        title.classList.add("folder-title");
        title.textContent = folder.name;

        const content = document.createElement("div");
        content.classList.add("folder-content");

        folder.videos.forEach(video => {
            const videoLink = document.createElement("a");
            videoLink.href = "#";
            videoLink.textContent = video;

            videoLink.addEventListener("click", e => {
                e.preventDefault();
                alert("Clicked video: " + video);
            });

            content.appendChild(videoLink);
        });

        folderDiv.appendChild(title);
        folderDiv.appendChild(content);
        container.appendChild(folderDiv);

        // Accordion behavior
        title.addEventListener("click", () => {
            document.querySelectorAll(".folder").forEach(f => {
                if (f !== folderDiv) f.classList.remove("active");
            });
            folderDiv.classList.toggle("active");
        });
    });
}

// Load Videos button in sidebar
const loadBtn = document.getElementById("loadVideosBtn");
if (loadBtn) {
    loadBtn.addEventListener("click", () => {
        renderVideos();
    });
}

