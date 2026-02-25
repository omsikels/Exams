#!/bin/bash

# Fixed Admin ML Service Startup Script for Windows/Git Bash

echo "=============================================="
echo "ADMIN ML SERVICE STARTUP - WINDOWS/GIT BASH"
echo "=============================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use (Windows compatible)
port_in_use() {
    netstat -an 2>/dev/null | grep ":$1 " >/dev/null 2>&1
}

# Check dependencies
echo "🔍 Checking dependencies..."

if ! command_exists python; then
    if ! command_exists python3; then
        echo "❌ Python is not installed. Please install Python first."
        exit 1
    else
        PYTHON_CMD="python3"
    fi
else
    PYTHON_CMD="python"
fi

echo "✅ Python is available ($PYTHON_CMD)"

if ! command_exists pip3 && ! command_exists pip; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip is available"

# Install Python dependencies
if [ -f "ml_service/requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r ml_service/requirements.txt
else
    echo "📦 Installing Flask directly..."
    pip install Flask==2.3.3 Werkzeug==2.3.7
fi

# Check ports (Windows compatible)
echo "🔍 Checking ports..."
if port_in_use 5000; then
    echo "⚠️  Port 5000 might be in use"
    echo "Current connections:"
    netstat -an 2>/dev/null | grep ":5000"
    echo ""
fi

if port_in_use 3000; then
    echo "⚠️  Port 3000 might be in use"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p videos extracted ml_service/processed_output ml_service/admin_videos

echo "✅ Directories created"

echo "=============================================="
echo "STARTING SERVICES"
echo "=============================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$ML_PID" ]; then
        kill $ML_PID 2>/dev/null
    fi
    if [ ! -z "$NODE_PID" ]; then
        kill $NODE_PID 2>/dev/null
    fi
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Start ML service
echo "🚀 Starting Admin ML Processing Service on port 5000..."
echo "🔐 Admin access only - authentication required"
echo ""

cd ml_service 2>/dev/null || mkdir -p ml_service
$PYTHON_CMD "/c/Users/user/AppData/Local/Programs/Python/Python314/python.exe" admin_video_server.py &
ML_PID=$!

# Wait for ML service to start
echo "⏳ Waiting for ML service to start..."
sleep 5

# Check if ML service is running (Windows compatible way)
ML_RUNNING=false
for i in {1..10}; do
    if netstat -an 2>/dev/null | grep ":5000" | grep "LISTENING" >/dev/null; then
        ML_RUNNING=true
        break
    fi
    echo "   Checking attempt $i/10..."
    sleep 2
done

if [ "$ML_RUNNING" = true ]; then
    echo "✅ Admin ML service started successfully (PID: $ML_PID)"
else
    echo "⚠️  ML service may still be starting up..."
    echo "   Check manually: http://localhost:5000/health"
fi

# Start Node.js server if available
cd ..
if [ -f "server.js" ] || [ -f "updated_server.js" ]; then
    if command_exists node; then
        echo ""
        echo "🚀 Starting Node.js server on port 3000..."
        
        # Use updated_server.js if available, otherwise server.js
        if [ -f "updated_server.js" ]; then
            node updated_server.js &
        else
            node server.js &
        fi
        NODE_PID=$!
        
        # Wait for Node.js service
        sleep 3
        
        if netstat -an 2>/dev/null | grep ":3000" | grep "LISTENING" >/dev/null; then
            echo "✅ Node.js server started successfully (PID: $NODE_PID)"
        else
            echo "⚠️  Node.js server may still be starting up..."
        fi
    else
        echo "⚠️  Node.js not found, skipping web server startup"
    fi
else
    echo "⚠️  server.js not found, skipping web server startup"
fi

echo ""
echo "=============================================="
echo "SERVICES STATUS"
echo "=============================================="
echo "🤖 ML Service: http://localhost:5000"
if [ ! -z "$NODE_PID" ]; then
    echo "🌐 Web Server: http://localhost:3000"
    echo "🔐 Admin Panel: http://localhost:3000/admin.html"
    echo "🎛️  ML Management: http://localhost:3000/admin/ml_management.html"
fi
echo ""
echo "🔑 Admin credentials:"
echo "   Username: admin"
echo "   Password: exam3000"
echo ""
echo "🧪 Test commands:"
echo "   curl http://localhost:5000/health"
if [ ! -z "$NODE_PID" ]; then
    echo "   curl http://localhost:3000"
fi
echo ""
echo "Press Ctrl+C to stop all services"
echo "=============================================="

# Keep services running
echo "✅ Services are running. Monitoring..."

while true; do
    # Check if ML service is still running
    if [ ! -z "$ML_PID" ] && ! kill -0 $ML_PID 2>/dev/null; then
        echo "❌ ML service stopped unexpectedly"
        break
    fi
    
    # Check if Node.js service is still running (if it was started)
    if [ ! -z "$NODE_PID" ] && ! kill -0 $NODE_PID 2>/dev/null; then
        echo "❌ Node.js server stopped unexpectedly"
        break
    fi
    
    sleep 10
done

cleanup