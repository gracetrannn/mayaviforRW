import win32gui
import win32process
import psutil


def get_foreground_window_processes():
    window = win32gui.GetForegroundWindow()
    _, pid = win32process.GetWindowThreadProcessId(window)
    return psutil.Process(pid)


foreground_process = get_foreground_window_processes()
# Add your screen-capturing app names here
screen_capturing_apps = ["zoom.exe",
                         "skype.exe", "obs-studio.exe", "NetKey.exe"]
while true
   if foreground_process.name().lower() in screen_capturing_apps:
        print(f"Screen capturing detected by {foreground_process.name()}")
    else:
        print("No screen capturing detected.")
