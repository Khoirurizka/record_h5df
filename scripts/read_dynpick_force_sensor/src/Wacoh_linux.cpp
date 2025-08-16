#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <iostream>
#include <dirent.h>
#include <vector>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

int COM = -1;
bool wacoh_isConnected = false;
string serialPortList[10];

void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void WacohRead(float force_tmp[6]) {
    if (COM < 0) {
        fprintf(stderr, "[ERROR] COM port not open\n");
        return;
    }

    char str[128] = {0};
    int n = write(COM, "R", 1);
    if (n <= 0) {
        perror("write failed");
        return;
    }

    sleep_ms(1);
    n = read(COM, str, 27);
    if (n <= 0 || n > 127) {
        perror("read failed or invalid length");
        return;
    }
    str[n] = '\0';

    int tmp;
    unsigned short forceLoad[6] = {0};
    int forceNoLoad[6] = {8342, 8230, 8055, 8134, 8192, 8436};
    float forceSenstv[6] = {32, 32, 29.455, 1635, 1635, 1635};

    int parsed = sscanf(str, "%1d%4hx%4hx%4hx%4hx%4hx%4hx", &tmp,
                        &forceLoad[0], &forceLoad[1], &forceLoad[2],
                        &forceLoad[3], &forceLoad[4], &forceLoad[5]);

    if (parsed < 7) {
        fprintf(stderr, "[ERROR] Failed to parse all force values\n");
        return;
    }

    for (int i = 0; i < 6; ++i) {
        force_tmp[i] = (static_cast<int>(forceLoad[i]) - forceNoLoad[i]) / forceSenstv[i];
    }
}

int serial_connect(string com_path) {
    COM = open(com_path.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (COM < 0) {
        perror("> Unable to open port");
        return -1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(COM, &tty) != 0) {
        perror("> Error from tcgetattr");
        close(COM);
        return -1;
    }

    cfsetospeed(&tty, B921600);
    cfsetispeed(&tty, B921600);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 5;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(COM, TCSANOW, &tty) != 0) {
        perror("> Error from tcsetattr");
        close(COM);
        return -1;
    }

    wacoh_isConnected = true;
    printf("> Serial Communication success... \n");
    return 1;
}

void serial_close() {
    if (COM >= 0) {
        close(COM);
        COM = -1;
    }
    wacoh_isConnected = false;
}

void detect_serialPort() {
    int num = 0;
    const string prefixUSB = "/dev/ttyUSB";
    const string prefixS = "/dev/ttyS";

    for (int i = 0; i < 256 && num < 10; i++) {
        string port1 = prefixUSB + to_string(i);
        if (access(port1.c_str(), F_OK) == 0) {
            serialPortList[num++] = port1;
            cout << port1 << " found" << endl;
        }

        string port2 = prefixS + to_string(i);
        if (access(port2.c_str(), F_OK) == 0) {
            serialPortList[num++] = port2;
            cout << port2 << " found" << endl;
        }
    }
}
