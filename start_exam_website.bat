@echo off
:: ------------------------------------------------------------
:: AUTO-RELAUNCH WITH ADMIN RIGHTS
:: ------------------------------------------------------------
if not defined RUN_IN_CMD (
    set RUN_IN_CMD=1
    :: Check for admin rights
    net session >nul 2>&1
    if errorlevel 1 (
        echo Requesting administrator privileges...
        powershell -Command "Start-Process '%~f0' -Verb RunAs"
        exit /b
    )
    start "" cmd /k "%~f0"
    exit /b
)

:: Enable delayed expansion for variable handling
setlocal enabledelayedexpansion

:: Change to the directory where the script is located
cd /d "%~dp0"

:: From here below, the window WILL NOT close automatically.

echo ===============================================
echo          EXAM WEBSITE STARTUP SCRIPT
echo ===============================================
echo.
echo Starting automatic setup...
echo Please wait, this may take several minutes...
echo.
echo Current folder: %CD%
echo.

:: ============================================================
:: 1 — AUTO-INSTALL NODE.JS
:: ============================================================
:CHECK_NODE
echo [1/5] Checking Node.js...
node --version >nul 2>&1

if errorlevel 1 (
    echo Node.js not found. Installing automatically...
    echo.
    
    :: Download Node.js installer
    echo Downloading Node.js installer...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi' -OutFile '%TEMP%\nodejs.msi'}"
    
    if exist "%TEMP%\nodejs.msi" (
        echo Installing Node.js silently...
        msiexec /i "%TEMP%\nodejs.msi" /qn /norestart
        
        :: Wait for installation
        timeout /t 10 >nul
        
        :: Refresh environment variables
        call :RefreshEnv
        
        :: Clean up
        del "%TEMP%\nodejs.msi"
        
        echo Node.js installed successfully!
    ) else (
        echo ERROR: Failed to download Node.js installer.
        echo Please check your internet connection.
        pause
        goto END
    )
) else (
    echo Node.js already installed!
)
echo.


:: ============================================================
:: 2 — AUTO-INSTALL PYTHON
:: ============================================================
:CHECK_PYTHON
echo [2/5] Checking Python...
python --version >nul 2>&1

if errorlevel 1 (
    echo Python not found. Installing automatically...
    echo.
    
    :: Download Python installer
    echo Downloading Python installer...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe' -OutFile '%TEMP%\python_installer.exe'}"
    
    if exist "%TEMP%\python_installer.exe" (
        echo Installing Python silently...
        "%TEMP%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        
        :: Wait for installation
        timeout /t 15 >nul
        
        :: Refresh environment variables
        call :RefreshEnv
        
        :: Clean up
        del "%TEMP%\python_installer.exe"
        
        echo Python installed successfully!
    ) else (
        echo ERROR: Failed to download Python installer.
        echo Please check your internet connection.
        pause
        goto END
    )
) else (
    echo Python already installed!
)
echo.


:: ============================================================
:: 3 — CHECK package.json
:: ============================================================
echo [3/5] Checking project files...
echo Current folder: %CD%
echo.
echo Listing all files in current folder:
dir /b
echo.
echo Checking for package.json...

if exist "package.json" (
    echo SUCCESS: package.json found!
    goto CONTINUE_SETUP
) else (
    echo.
    echo ERROR: package.json not found in current folder.
    echo.
    echo Please make sure:
    echo 1. This .bat file is in the same folder as package.json
    echo 2. The file is named exactly "package.json" (not package.json.txt)
    echo.
    timeout /t 30
    goto END
)

:CONTINUE_SETUP
echo.


:: ============================================================
:: 4 — INSTALL NODE MODULES
:: ============================================================
echo [4/5] Installing Node.js packages...
echo This may take a few minutes...
if not exist "node_modules" (
    echo Installing all packages...
    call npm install
) else (
    echo Checking for missing packages...
    call npm list --depth=0 body-parser cors express express-session form-data multer node-fetch serve-favicon >nul 2>&1
    if errorlevel 1 (
        echo Installing missing packages...
        call npm install body-parser cors express express-session form-data multer node-fetch serve-favicon
    ) else (
        echo All packages ready!
    )
)
echo.


:: ============================================================
:: 5 — INSTALL PYTHON DEPENDENCIES (SYSTEM-WIDE)
:: ============================================================
echo [5/5] Installing Python packages to system...

:: Find Python executable
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please restart your computer and try again.
    timeout /t 30
    goto END
)

:: Get Python location
for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i
echo Using Python at: %PYTHON_PATH%

echo Checking Python dependencies...
python -c "import flask; from flask_cors import CORS; import numpy, cv2, PIL, torch, torchvision, werkzeug" >nul 2>&1

if errorlevel 1 (
    echo Installing required Python packages to system...
    echo This may take several minutes, please be patient...
    echo.
    python -m pip install --upgrade pip --quiet
    pip install Flask flask-cors numpy opencv-python==4.12.0.88 Pillow==12.0.0 torch==2.9.1 torchvision==0.24.1 Werkzeug==3.1.4
    echo.
    echo Python packages installed to system!
) else (
    echo Python packages ready!
)
echo.


:: ============================================================
:: 6 — START SERVICES
:: ============================================================
echo Starting exam website...
echo.

echo Starting web server...
start "Exam Web Server" cmd /k "npm start"

timeout /t 3 >nul

if exist "ml_services\enhanced_facial_recognition_server.py" (
    echo Starting facial recognition service...
    start "Facial Recognition" cmd /k "cd ml_services && python enhanced_facial_recognition_server.py"
) else (
    echo Note: Facial recognition service not found - continuing without it.
)
echo.

timeout /t 3 >nul

echo Opening exam website in browser...
start "" "http://localhost:3000/index.html"
start "" "http://localhost:3000/login.html"
echo.


echo ===============================================
echo          WEBSITE IS RUNNING!
echo ===============================================
echo.
echo The exam website is now open in your browser.
echo.
echo IMPORTANT: Keep this window open while using the exam website.
echo            Close this window only when done with exams.
echo.
timeout /t 60

goto END


:: ============================================================
:: HELPER FUNCTION: REFRESH ENVIRONMENT VARIABLES
:: ============================================================
:RefreshEnv
:: Refresh PATH from registry
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%b"
set "PATH=%SYS_PATH%;%USER_PATH%"
goto :eof


:END
echo.
echo Window will stay open. You can minimize it but don't close it.
echo.
:KEEPOPEN
timeout /t 3600 >nul
goto KEEPOPEN