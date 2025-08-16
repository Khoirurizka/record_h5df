#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Wacoh_linux.h"

namespace py = pybind11;

PYBIND11_MODULE(wacoh_sensor, m) {
    m.def("detect_serialPort", &detect_serialPort);
    m.def("get_serial_ports", []() {
        std::vector<std::string> ports;
        for (int i = 0; i < 10; ++i) {
            if (!serialPortList[i].empty())
                ports.push_back(serialPortList[i]);
        }
        return ports;
    });

    m.def("serial_connect", &serial_connect);
    m.def("serial_close", &serial_close);

    m.def("WacohRead", []() {
        float tmp[6];
        WacohRead(tmp);
        return std::vector<float>(tmp, tmp + 6);
    });
}
