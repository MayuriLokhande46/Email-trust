import subprocess
import sys
import time
import webbrowser
import platform

def start_services():
    """
    Starts the backend and frontend services concurrently.
    """
    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "spam_detector.src.api:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    frontend_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "spam_detector/src/app.py",
        "--server.port", "8501"
    ]

    # Use a specific creation flag for Windows to avoid inheriting console handles
    creationflags = 0
    if platform.system() == "Windows":
        # Using getattr to be safe, though this is standard on Windows
        creationflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)

    print("Starting backend server (FastAPI)...")
    try:
        backend_process = subprocess.Popen(backend_command, creationflags=creationflags)
        print(f"Backend server started with PID: {backend_process.pid}")
    except Exception as e:
        print(f"Failed to start backend: {e}")
        return

    # Give the backend a moment to start up before the frontend tries to connect
    print("Waiting for backend to initialize...")
    time.sleep(5)

    print("\nStarting frontend server (Streamlit)...")
    try:
        frontend_process = subprocess.Popen(frontend_command, creationflags=creationflags)
        print(f"Frontend server started with PID: {frontend_process.pid}")
    except Exception as e:
        print(f"Failed to start frontend: {e}")
        backend_process.kill()
        return

    frontend_url = "http://localhost:8501"
    print(f"\nYour application should be available at: {frontend_url}")
    print("Opening the application in your default web browser...")

    # Wait a bit for the frontend server to be ready before opening the browser
    time.sleep(2)
    try:
        webbrowser.open(frontend_url)
    except Exception as e:
        print(f"Could not automatically open browser: {e}")

    print("\n=======================================================")
    print("  Both servers are running. Press Ctrl+C to stop them. ")
    print("=======================================================")

    try:
        # Wait for the frontend process to complete. 
        # Since Streamlit runs until stopped, this will keep the script alive.
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n\nCtrl+C received. Shutting down both servers...")
        if platform.system() == "Windows":
            # On Windows, sending CTRL_BREAK_EVENT to the process group is a reliable way
            # to terminate the child and its own subprocesses.
            backend_process.send_signal(subprocess.CTRL_BREAK_EVENT)
            frontend_process.send_signal(subprocess.CTRL_BREAK_EVENT)
        else:
            # On Unix-like systems, SIGTERM is standard for graceful shutdown.
            backend_process.terminate()
            frontend_process.terminate()
        
        # Wait for processes to actually terminate to avoid orphaned processes
        try:
            backend_process.wait(timeout=5)
            frontend_process.wait(timeout=5)
            print("Servers have been shut down successfully.")
        except subprocess.TimeoutExpired:
            print("Servers did not shut down gracefully. Forcing termination.")
            backend_process.kill()
            frontend_process.kill()
            print("Servers have been killed.")
            
    finally:
        # Ensure processes are killed if they are still alive
        if backend_process.poll() is None:
            backend_process.kill()
        if frontend_process.poll() is None:
            frontend_process.kill()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    start_services()