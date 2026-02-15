# main.py
import sys
import signal
import subprocess
from multiprocessing import Process
from core import Model

from core.watchdog.watch_images_other import start_watch as start_watch_images
from core.watchdog.watch_pdfs import start_watch as start_watch_pdfs
from core.watchdog.watch_audio_other import start_watch as start_watch_audio
from core.watchdog.sync_manager import run_initial_sync


watchdog_processes = []
streamlit_process = None


# ============================
# Graceful shutdown
# ============================
def shutdown(sig, frame):
    print("\n🛑 Shutting down system...")

    for p in watchdog_processes:
        if p.is_alive():
            p.terminate()

    if streamlit_process:
        streamlit_process.terminate()

    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ============================
# MAIN
# ============================
if __name__ == "__main__":

    model = Model()
    model.download_model();

    print("=======================================")
    print("🔄 Running initial filesystem sync...")
    print("=======================================")
    run_initial_sync()

    print("\n🚀 Starting watchdog services...")

    watchdog_processes = [
        Process(target=start_watch_images, name="Watchdog-Images"),
        Process(target=start_watch_pdfs,   name="Watchdog-PDFs"),
        Process(target=start_watch_audio,  name="Watchdog-Audio"),
    ]

    for p in watchdog_processes:
        p.start()
        print(f"✅ {p.name} started (PID {p.pid})")

    print("\n🚀 Starting Streamlit UI...")

    streamlit_process = subprocess.Popen(
        [
            "/opt/miniconda3/envs/content-search-ai/bin/streamlit",
            "run",
            "app.py",
            "--server.port", "8501",
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"✅ Streamlit started (PID {streamlit_process.pid})")

    print("\n👀 System running. Press Ctrl+C to stop.\n")

    # Block main process
    for p in watchdog_processes:
        p.join()
