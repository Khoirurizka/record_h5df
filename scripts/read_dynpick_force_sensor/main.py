import sys
import time
import select
import termios
import tty

sys.path.append("build")
import wacoh_sensor

def kbhit():
    """Returns True if a key has been pressed (non-blocking stdin)."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr != []

def getch():
    """Read a single character from stdin."""
    return sys.stdin.read(1)

# Save terminal settings
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

try:
    print("Detecting ports...")
    wacoh_sensor.detect_serialPort()

    print("Getting port list...")
    ports = wacoh_sensor.get_serial_ports()
    print("Ports:", ports)

    if ports:
        print("Connecting to port:", ports[0])
        ret = wacoh_sensor.serial_connect(ports[0])
        print("serial_connect returned:", ret)

        if ret == 1:
            print("Reading force sensor... (press 'q' to quit)")

            while True:
                if kbhit():
                    key = getch()
                    if key.lower() == 'q':
                        print("\n[INFO] 'q' pressed. Exiting...")
                        break

                data = wacoh_sensor.WacohRead()
                print("Force:", data)
                time.sleep(0.1)
        else:
            print("ERROR: Failed to connect")
    else:
        print("ERROR: No ports found")
finally:
    # Restore terminal settings
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    wacoh_sensor.serial_close()
    print("Serial connection closed.")
