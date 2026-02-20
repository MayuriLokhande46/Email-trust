"""
Single command to run both Frontend and Backend together.
"""
import subprocess
import sys
import os
import time

def main():
    # Set PYTHONPATH to include the src directory
    env = os.environ.copy()
    src_path = os.path.join(os.path.dirname(__file__), 'spam_detector', 'src')
    env['PYTHONPATH'] = src_path
    
    print("="*60)
    print("  Starting Email Spam Detector")
    print("="*60)
    print()
    
    # Start backend
    print("[1/2] Starting Backend API on port 8000...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "api:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ]
    
    backend = subprocess.Popen(
        backend_cmd,
        cwd=src_path,
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )
    
    # Wait for backend to start
    print("   Waiting for backend to initialize...")
    time.sleep(5)
    
    # Start frontend
    print("[2/2] Starting Frontend on port 8501...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run",
        os.path.join(os.path.dirname(__file__), "spam_detector", "src", "app.py")
    ]
    
    frontend = subprocess.Popen(
        frontend_cmd,
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )
    
    print()
    print("="*60)
    print("  ✅ Application Started Successfully!")
    print("="*60)
    print()
    print("  📧 Frontend: http://localhost:8501")
    print("  🔧 Backend:  http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop both services...")
    print()
    
    try:
        # Keep the script running
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("Services stopped.")

if __name__ == "__main__":
    main()
