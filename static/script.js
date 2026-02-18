// Global variables
let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let stream = null;
let detectionInterval = null;
let timeoutId = null;
let startBtn = document.getElementById('startBtn');
let statusCheckInterval = null;  // For polling kiosk status

// Load face-api models (no auto-start)
document.addEventListener('DOMContentLoaded', async () => {
    if (window.location.pathname === '/student_panel') {
        try {
            console.log('Loading face-api models from local...');
            await faceapi.nets.tinyFaceDetector.loadFromUri('/static/weights/');
            console.log('Models loaded successfully.');
            showStatus('Models loaded. Click "Start Camera" to begin.', 'info');
            
            // Start polling to check if kiosk is still active
            statusCheckInterval = setInterval(checkKioskStatus, 10000);  // Every 10 seconds
        } catch (err) {
            console.error('Failed to load models:', err);
            showStatus('Error loading models. Ensure weights are in static/weights/ and refresh.', 'danger');
        }
    }
});

// Check if kiosk is still active (lock file exists)
async function checkKioskStatus() {
    try {
        const response = await fetch('/kiosk_status');
        const data = await response.json();
        if (!data.active) {
            console.log('Kiosk ended by admin. Stopping...');
            stopCamera();
            clearInterval(statusCheckInterval);
            showStatus('Student Panel has been ended by the admin. Please close this tab.', 'warning');
        }
    } catch (err) {
        console.error('Error checking status:', err);
    }
}

// Start kiosk: Manual trigger via button
async function startKiosk() {
    console.log('Starting kiosk...');
    
    startBtn.style.display = 'none';
    video.style.display = 'block';
    
    if (detectionInterval) clearInterval(detectionInterval);
    if (timeoutId) clearTimeout(timeoutId);
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            console.log('Requesting camera access...');
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            console.log('Camera started.');
            showStatus('Camera active. Detecting faces...', 'info');
            
            detectionInterval = setInterval(async () => {
                if (video.readyState === 4) {
                    console.log('Detecting faces...');
                    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions());
                    if (detections.length > 0) {
                        console.log('Face detected, capturing...');
                        clearInterval(detectionInterval);
                        clearTimeout(timeoutId);
                        showStatus('Face detected. Marking attendance...', 'success');
                        await captureAndSend();
                    }
                }
            }, 1000);
            
            timeoutId = setTimeout(() => {
                console.log('No face detected in 4 seconds. Stopping camera.');
                stopCamera();
                showStatus('No face detected in 4 seconds. Click "Start Camera" to retry.', 'warning');
            }, 4000);
        } catch (err) {
            console.error('Camera error:', err);
            stopCamera();
            if (err.name === 'NotAllowedError') {
                showStatus('Camera access denied. Please allow camera permissions and click "Start Camera".', 'danger');
            } else {
                showStatus('Camera error. Check permissions and click "Start Camera".', 'danger');
            }
        }
    } else {
        console.error('Camera not supported.');
        stopCamera();
        showStatus('Camera not supported on this device.', 'danger');
    }
}

// Stop camera and reset UI
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (detectionInterval) clearInterval(detectionInterval);
    if (timeoutId) clearTimeout(timeoutId);
    video.style.display = 'none';
    startBtn.style.display = 'block';
}

// Capture image and send to server
async function captureAndSend() {
    console.log('Capturing image...');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 640, 480);
    const image = canvas.toDataURL('image/jpeg');
    
    const formData = new FormData();
    formData.append('image', image);
    
    try {
        console.log('Sending image to server...');
        const response = await fetch('/mark_attendance_student', { 
            method: 'POST', 
            body: formData,
            headers: { 'X-Requested-With': 'XMLHttpRequest' }
        });
        const data = await response.json();
        console.log('Server response:', data);
        showStatus(data.message, data.status === 'success' ? 'success' : 'warning');
        if (data.status === 'success') {
            setTimeout(() => stopCamera(), 3000);
        } else {
            setTimeout(() => stopCamera(), 2000);
        }
    } catch (err) {
        console.error('Send error:', err);
        stopCamera();
        showStatus('Error sending data. Click "Start Camera" to retry.', 'danger');
    }
}

// Status display
function showStatus(text, type = 'info') {
    const statusEl = document.getElementById('status');
    statusEl.textContent = text;
    statusEl.className = `kiosk-message text-${type}`;
}

// Stop camera on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
    if (statusCheckInterval) clearInterval(statusCheckInterval);
});