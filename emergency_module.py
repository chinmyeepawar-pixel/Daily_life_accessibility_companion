import threading
import time
import winsound

running = False

def start_emergency():
    global running
    running = True

    def alert_loop():
        while running:
            try:
                # 🔊 Loud beep
                winsound.Beep(1000, 700)
            except:
                pass
            time.sleep(0.3)

    thread = threading.Thread(target=alert_loop, daemon=True)
    thread.start()


def stop_emergency():
    global running
    running = False