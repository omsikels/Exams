const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const session = require('express-session');
// Note: Add these dependencies to package.json:
// const FormData = require('form-data');
// const fetch = require('node-fetch');

const app = express();
const PORT = process.env.PORT || 3000;
const VIDEOS_DIR = path.join(__dirname, 'videos');
const EXTRACTED_DIR = path.join(__dirname, 'extracted');
const QUESTIONS_PATH = path.join(process.cwd(), 'questions.json');

// Admin ML service configuration
const ADMIN_ML_SERVICE_URL = process.env.ADMIN_ML_SERVICE_URL || 'http://localhost:5001';

// Auto-create directories if missing
if (!fs.existsSync(QUESTIONS_PATH)) {
  fs.writeFileSync(QUESTIONS_PATH, JSON.stringify({ totalSeconds: 60, questions: [] }, null, 2));
  console.log("Created questions.json at:", QUESTIONS_PATH);
}

if (!fs.existsSync(EXTRACTED_DIR)) {
  fs.mkdirSync(EXTRACTED_DIR, { recursive: true });
  console.log("Created extracted directory at:", EXTRACTED_DIR);
}

if (!fs.existsSync(VIDEOS_DIR)) {
  fs.mkdirSync(VIDEOS_DIR, { recursive: true });
  console.log("Created videos directory at:", VIDEOS_DIR);
}

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));

// Try to use favicon, but don't crash if it doesn't exist
try {
  const favicon = require('serve-favicon');
  app.use(favicon(path.join(__dirname, "public", "favicon.ico")));
} catch (err) {
  console.warn("Favicon middleware not available or favicon not found");
}

// Simple session-based login
app.use(
  session({
    secret: 'supersecretkey',
    resave: false,
    saveUninitialized: true,
  })
);

// Helper functions
function sanitize(name) {
  return name.replace(/[^a-z0-9\- _\.]/gi, '_');
}

// Authentication routes
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  if (username === 'admin' && password === 'exam3000') {
    req.session.loggedIn = true;
    req.session.adminUser = username;
    return res.json({ ok: true });
  }
  res.status(401).json({ ok: false, message: 'Invalid credentials' });
});

app.post('/logout', (req, res) => {
  req.session.destroy(() => res.json({ ok: true }));
});

// Protect /admin.html
app.get('/admin.html', (req, res, next) => {
  if (!req.session.loggedIn) {
    return res.redirect('/login.html');
  }
  next();
});

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// API routes
app.use(express.json());

//Serve Facial Recognition Page
app.use(express.static(path.join(__dirname, 'public')));

// Get questions
app.get('/api/questions', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(QUESTIONS_PATH, 'utf8'));
    res.json(data);
  } catch (err) {
    res.json({ totalSeconds: 60, questions: [] });
  }
});

// Add Facial Recognition Route
app.get('/improved_facial_recognition.html', (req, res, next) => {
  if (!req.session.loggedIn) {
    return res.redirect('/login.html');
  }
  next();
});

// add facial recognition proxy endpoints (optional)
app.get('/api/facial/*', async (req, res) => {
  if (!req.session.loggedIn) {
    return res.status(401).json({ error: 'Admin Access Required' });
  }

  try {
    const facialURL = `http://localhost:5001${req.path.replace('/api/facial', '/api')}`;
    const response = await fetch(facialURL);
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'Facial recognition service unavailable' });
  }
});

// Save questions
app.post('/api/questions', (req, res) => {
  try {
    fs.writeFileSync(QUESTIONS_PATH, JSON.stringify(req.body, null, 2));
    res.json({ ok: true });
  } catch (err) {
    console.error('Error saving questions:', err);
    res.status(500).json({ ok: false });
  }
});

// File Upload Handling
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const { username } = req.body;
    if (!username) return cb(new Error('Missing username field'));

    const userDir = path.join(VIDEOS_DIR, sanitize(username));
    if (!fs.existsSync(userDir)) fs.mkdirSync(userDir, { recursive: true });

    cb(null, userDir);
  },
  filename: function (req, file, cb) {
    const { questionIndex, result } = req.body;
    if (!questionIndex || !result) return cb(new Error('Missing fields'));

    const label = result === 'Correct' ? 'right' : 'wrong';
    const filename = `${label}${questionIndex}.mp4`;
    cb(null, filename);
  }
});

const upload = multer({ storage });

// Video upload - stores locally and notifies about admin ML processing
app.post('/api/upload-video', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ ok: false, message: 'No File Uploaded' });
  }

  const videoPath = req.file.path;
  const { username, questionIndex, result } = req.body;

  console.log(`\n${'='.repeat(60)}`);
  console.log(`VIDEO UPLOAD RECEIVED`);
  console.log(`${'='.repeat(60)}`);
  console.log(`Student: ${username}`);
  console.log(`Question: ${questionIndex}`);
  console.log(`Result: ${result}`);
  console.log(`File: ${req.file.filename}`);
  console.log(`Size: ${(req.file.size / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Stored: ${videoPath}`);
  console.log(`💡 Note: Admin can process this via ML service`);
  console.log(`${'='.repeat(60)}\n`);

  // Prepare response for the client
  const clientResponse = {
    ok: true,
    path: `/videos/${sanitize(username)}/${req.file.filename}`,
    message: 'Video uploaded successfully',
    ml_processing: 'Available for admin processing'
  };

  // Send response to client immediately
  res.json(clientResponse);

  // Log upload for admin ML processing queue
  try {
    const uploadLog = {
      timestamp: new Date().toISOString(),
      student: username,
      question: questionIndex,
      result: result,
      filename: req.file.filename,
      filepath: videoPath,
      size_mb: (req.file.size / 1024 / 1024).toFixed(2),
      status: 'ready_for_admin_processing'
    };

    const logPath = path.join(EXTRACTED_DIR, 'upload_queue.json');
    let uploadQueue = [];

    if (fs.existsSync(logPath)) {
      try {
        uploadQueue = JSON.parse(fs.readFileSync(logPath, 'utf8'));
      } catch (err) {
        console.warn('Could not read upload queue, starting fresh');
      }
    }

    uploadQueue.push(uploadLog);

    // Keep only last 1000 entries to prevent file from growing too large
    if (uploadQueue.length > 1000) {
      uploadQueue = uploadQueue.slice(-1000);
    }

    fs.writeFileSync(logPath, JSON.stringify(uploadQueue, null, 2));
    console.log(`📝 Upload logged for admin ML processing: ${req.file.filename}`);

  } catch (error) {
    console.error('Error logging upload for ML processing:', error.message);
  }
});

// Videos list for admin
app.get("/api/videos", (req, res) => {
  const videosRoot = path.join(__dirname, "videos");

  fs.readdir(videosRoot, { withFileTypes: true }, (err, users) => {
    if (err) return res.status(500).json({ error: "Failed to read videos directory" });

    const result = {};

    users.forEach(userDir => {
      if (userDir.isDirectory()) {
        const userPath = path.join(videosRoot, userDir.name);
        try {
          const files = fs.readdirSync(userPath)
            .filter(f => f.endsWith(".mp4") || f.endsWith(".webm"));
          result[userDir.name] = files;
        } catch (err) {
          console.warn(`Error reading user directory ${userDir.name}:`, err.message);
          result[userDir.name] = [];
        }
      }
    });

    res.json(result);
  });
});

// Enhanced Emotions Save API
app.post("/api/save-emotions", (req, res) => {
  try {
    const newData = req.body;
    if (!newData.folder || !newData.video || !newData.segments) {
      return res.status(400).json({ ok: false, message: "Invalid data structure" });
    }

    const emotionsPath = path.join(EXTRACTED_DIR, "emotions.json");
    let existingData = {};

    if (fs.existsSync(emotionsPath)) {
      try {
        const fileContent = fs.readFileSync(emotionsPath, 'utf8');
        existingData = JSON.parse(fileContent);
      } catch (err) {
        console.warn("Could not parse existing emotions.json, starting fresh:", err.message);
        existingData = {};
      }
    }

    const videoKey = `${newData.folder}_${newData.video}`;
    existingData[videoKey] = {
      video: newData.video,
      folder: newData.folder,
      totalDuration: newData.totalDuration,
      segments: newData.segments,
      manual_annotation: true,
      annotated_at: new Date().toISOString(),
      annotated_by: req.session.adminUser || 'admin'
    };

    fs.writeFileSync(emotionsPath, JSON.stringify(existingData, null, 2));

    console.log(`Manual emotion annotation saved: ${videoKey}`);
    res.json({ ok: true });
  } catch (err) {
    console.error("Error saving emotions.json:", err);
    res.status(500).json({ ok: false, message: "Server error while saving emotions data" });
  }
});

// API to get extracted emotions data
app.get("/api/extracted-emotions", (req, res) => {
  try {
    const emotionsPath = path.join(EXTRACTED_DIR, "emotions.json");
    
    if (!fs.existsSync(emotionsPath)) {
      return res.json({});
    }

    const data = JSON.parse(fs.readFileSync(emotionsPath, 'utf8'));
    res.json(data);
  } catch (err) {
    console.error("Error reading extracted emotions:", err);
    res.status(500).json({ error: "Failed to read extracted emotions data" });
  }
});

// API to get upload queue for admin ML processing
app.get("/api/upload-queue", (req, res) => {
  // Only allow admin access
  if (!req.session.loggedIn) {
    return res.status(401).json({ error: "Admin access required" });
  }

  try {
    const queuePath = path.join(EXTRACTED_DIR, 'upload_queue.json');
    
    if (!fs.existsSync(queuePath)) {
      return res.json([]);
    }

    const queue = JSON.parse(fs.readFileSync(queuePath, 'utf8'));
    
    // Filter for videos ready for processing
    const readyForProcessing = queue.filter(entry => 
      entry.status === 'ready_for_admin_processing'
    );

    res.json(readyForProcessing);
  } catch (err) {
    console.error("Error reading upload queue:", err);
    res.status(500).json({ error: "Failed to read upload queue" });
  }
});

// API to check admin ML service status (admin only)
app.get("/api/admin-ml-status", async (req, res) => {
  // Only allow admin access
  if (!req.session.loggedIn) {
    return res.status(401).json({ error: "Admin access required" });
  }

  try {
    // Try to fetch from admin ML service (no auth needed for health endpoint)
    const response = await fetch(`${ADMIN_ML_SERVICE_URL}/health`, { timeout: 5000 });
    
    if (response.ok) {
      const data = await response.json();
      res.json({ 
        available: true, 
        service: data,
        admin_endpoint: `${ADMIN_ML_SERVICE_URL}/admin/login`,
        management_interface: "/admin/ml_management.html"
      });
    } else {
      res.json({ available: false, error: "Service not responding properly" });
    }
  } catch (error) {
    res.json({ 
      available: false, 
      error: error.message,
      note: "Start with: ./admin/start_ml_service.sh"
    });
  }
});

// Add this route before the "Serve static files" line
app.get("/admin/real_emotion_ml_management.html", (req, res) => {
  // Only allow admin access
  if (!req.session.loggedIn) {
    return res.redirect('/login.html');
  }

  const managementPath = path.join(__dirname, 'public', 'admin', 'real_emotion_ml_management.html');
  
  if (fs.existsSync(managementPath)) {
    res.sendFile(managementPath);
  } else {
    res.status(404).send(`
      <html>
        <body>
          <h1>Real Emotion ML Management</h1>
          <p>The real emotion ML management interface file is not found.</p>
          <p>Expected location: ${managementPath}</p>
          <p><a href="/admin.html">← Back to Admin Panel</a></p>
        </body>
      </html>
    `);
  }
});

// Serve admin ML management interface (admin only)
app.get("/admin/ml_management.html", (req, res) => {
  // Only allow admin access
  if (!req.session.loggedIn) {
    return res.redirect('/login.html');
  }

  const managementPath = path.join(__dirname, 'admin', 'ml_management.html');
  
  if (fs.existsSync(managementPath)) {
    res.sendFile(managementPath);
  } else {
    res.status(404).send(`
      <html>
        <body>
          <h1>ML Management Interface</h1>
          <p>The ML management interface file is not found.</p>
          <p>Expected location: ${managementPath}</p>
          <p><a href="/admin.html">← Back to Admin Panel</a></p>
        </body>
      </html>
    `);
  }
});

// Serve videos and extracted files
app.use('/videos', express.static(VIDEOS_DIR));
app.use('/extracted', express.static(EXTRACTED_DIR));

// Start server
console.log('\n' + '='.repeat(80));
console.log('EXAM WEBSITE SERVER WITH ADMIN ML INTEGRATION');
console.log('='.repeat(80));
console.log(`🌐 Server running on: http://localhost:${PORT}`);
console.log(`📁 Videos directory: ${VIDEOS_DIR}`);
console.log(`📊 Extracted directory: ${EXTRACTED_DIR}`);
console.log(`🤖 Admin ML service URL: ${ADMIN_ML_SERVICE_URL}`);
console.log(`📝 Questions file: ${QUESTIONS_PATH}`);
console.log('='.repeat(80));
console.log('🔐 Admin Access:');
console.log(`   Main Admin Panel: http://localhost:${PORT}/admin.html`);
console.log(`   ML Management: http://localhost:${PORT}/admin/ml_management.html`);
console.log('   Username: admin');
console.log('   Password: exam3000');
console.log('='.repeat(80));

app.listen(PORT, async () => {
  console.log(`\n✅ Server is ready and listening on port ${PORT}`);
  
  // Check admin ML service on startup
  setTimeout(async () => {
    try {
      const fetch = (await import('node-fetch')).default;
      const response = await fetch(`${ADMIN_ML_SERVICE_URL}/health`);
      if (response.ok) {
        console.log(`✅ Admin ML service is available at ${ADMIN_ML_SERVICE_URL}`);
        console.log(`🔐 Start it with: ./admin/start_ml_service.sh`);
      }
    } catch (error) {
      console.log(`⚠️  Admin ML service not available at ${ADMIN_ML_SERVICE_URL}`);
      console.log('   Start it with: ./admin/start_ml_service.sh');
      console.log('   Videos will be stored locally and available for admin ML processing');
    }
  }, 1000);
});

module.exports = app;