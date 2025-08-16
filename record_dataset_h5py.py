import os
import cv2
import h5py
import numpy as np
import torch
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox, QLineEdit, QSpinBox, QCheckBox, QComboBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, Qt
from transformers import AutoTokenizer, AutoModel
import pyqtgraph as pg
from pyqtgraph import LegendItem
import wacoh_sensor



# Custom LegendItem class to support three-column layout
class LegendObj(pg.GraphicsWidget):
    def __init__(self, curves_dict, labels, colors, column_count=5):
        super().__init__()
        self.column_count = column_count
        self.items = []
        self.curves_dict = curves_dict
        self.labels = labels
        self.colors = colors

        self.staticLayoutItems()

    def staticLayoutItems(self):
        # Clear previous items
        for item in self.items:
            item[0].setParentItem(None)
            item[1].setParentItem(None)
        self.items.clear()

        items_per_column = max(1, (len(self.labels) + self.column_count - 1) // self.column_count)
        spacing = 5
        marker_size = 10

        max_label_width = 0
        item_height = 0

        # First pass to calculate layout sizes
        label_items = []
        for label in self.labels:
            text_item = pg.TextItem(text=label, anchor=(0, 0))
            text_item.setParentItem(self)
            label_items.append(text_item)
            br = text_item.boundingRect()
            max_label_width = max(max_label_width, br.width())
            item_height = max(item_height, br.height())

        col_width = marker_size + spacing + max_label_width + spacing

        # Layout
        for i, label in enumerate(self.labels):
            col = i // items_per_column
            row = i % items_per_column
            x = col * col_width
            y = row * (item_height + spacing)

            # Create color box based on curve pen
            pen = self.curves_dict[label].opts['pen']
            color = pen.color()

            marker = pg.QtWidgets.QGraphicsRectItem(0, 0, marker_size, marker_size)
            marker.setBrush(color)
            marker.setPen(pg.mkPen(color))
            marker.setParentItem(self)
            marker.setPos(x, y + 2)

            label_item = label_items[i]
            label_item.setPos(x + marker_size + spacing, y)

            self.items.append((marker, label_item))

        total_width = self.column_count * col_width
        total_height = items_per_column * (item_height + spacing)
        self.setMinimumSize(total_width, total_height)
        self.setMaximumSize(total_width, total_height)
        self._bounding_rect = pg.QtCore.QRectF(0, 0, total_width, total_height)

    def boundingRect(self):
        return self._bounding_rect

# ------------------ Load LLaVA-Pythia tokenizer & model -------------------
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model = AutoModel.from_pretrained("EleutherAI/pythia-410m").to(device)  # Move model to CUDA
model.eval()

def embed_text(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)  # Move inputs to CUDA
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # Move back to CPU for NumPy

# ------------------ PyQt6 Data Recorder App -------------------
class DataRecorder(QWidget):

    ### BEV cam button
    def toggle_BEV_cam(self):
        if self.run_cam_1_btn.isChecked():
            try:
                self.cap_main = cv2.VideoCapture(0)  # Reinitialize the camera
                ret, _ = self.cap_main.read()
                if ret:
                    self.run_cam_1_btn.setText("BEV cam on")
                    self.run_cam_1_btn.setStyleSheet("background-color: lightgreen;")
                    self.BEV_cam_on=True
                    print(f"[INFO] BEV cam is on")
                else:
                    raise RuntimeError("Failed to turn on BEV cam")
            except Exception as e:
                self.run_cam_1_btn.setChecked(False)
                self.run_cam_1_btn.setText("BEV cam off")
                self.run_cam_1_btn.setStyleSheet("background-color: lightcoral;")
                self.BEV_cam_on=False
                print(f"[ERROR] Failed to turn on BEV cam: {e}")
        else:
            try:
                self.cap_main.release()
                self.run_cam_1_btn.setText("BEV cam off")
                self.run_cam_1_btn.setStyleSheet("background-color: lightcoral;")
                self.BEV_cam_on=False
                print(f"[INFO] BEV cam is off")
            except Exception as e:
                print(f"[WARNING] Error while turning off BEV cam: {e}")

    ### Wrist cam button
    def toggle_wrist_cam(self):
        if self.run_cam_2_btn.isChecked():
            try:
                self.cap_wrist = cv2.VideoCapture(0)  # Reinitialize the second camera
                ret, _ = self.cap_wrist.read()
                if ret:
                    self.run_cam_2_btn.setText("Wrist cam on")
                    self.run_cam_2_btn.setStyleSheet("background-color: lightgreen;")
                    self.Wrist_cam_on=True
                    print(f"[INFO] Wrist cam is on")
                else:
                    raise RuntimeError("Failed to turn on Wrist cam")
            except Exception as e:
                self.run_cam_2_btn.setChecked(False)
                self.run_cam_2_btn.setText("Wrist cam off")
                self.run_cam_2_btn.setStyleSheet("background-color: lightcoral;")
                self.Wrist_cam_on=False
                print(f"[ERROR] Failed to turn on Wrist cam: {e}")
        else:
            try:
                self.cap_wrist.release()
                self.run_cam_2_btn.setText("Wrist cam off")
                self.run_cam_2_btn.setStyleSheet("background-color: lightcoral;")
                self.Wrist_cam_on=False
                print(f"[INFO] Wrist cam is off")
            except Exception as e:
                print(f"[WARNING] Error while turning off Wrist cam: {e}")


    ### DynPick Force Sensor
    def toggle_dynpick_connection(self):
        selected_port = self.usb_port_combo.currentText()

        if self.DynPick_connect_btn.isChecked():
            try:
                ret = wacoh_sensor.serial_connect(selected_port)
                if ret == 1:
                    self.DynPick_connect_btn.setText("Connected")
                    self.DynPick_connect_btn.setStyleSheet("background-color: lightgreen;")
                    print(f"[INFO] DynPick sensor connected on {selected_port}")
                else:
                    raise RuntimeError("Failed to connect to port.")
            except Exception as e:
                wacoh_sensor.serial_close()
                self.DynPick_connect_btn.setChecked(False)
                self.DynPick_connect_btn.setText("Disconnected")
                self.DynPick_connect_btn.setStyleSheet("background-color: lightcoral;")
                QMessageBox.critical(self, "Connection Error", f"Could not connect to {selected_port}\n\n{e}")
                print(f"[ERROR] DynPick connection failed: {e}")
        else:
            try:
                wacoh_sensor.serial_close()
                self.DynPick_connect_btn.setText("Disconnected")
                self.DynPick_connect_btn.setStyleSheet("background-color: lightcoral;")
                print(f"[INFO] DynPick sensor disconnected from {selected_port}")
            except Exception as e:
                print(f"[WARNING] Error while disconnecting: {e}")
            
    def update_force_displays(self):
        if not self.DynPick_connect_btn.isChecked():
            return  # Only read if connected

        try:
            data = wacoh_sensor.WacohRead()
            if len(data) != 6:
                print("[WARNING] Unexpected DynPick data:", data)
                return

            fx, fy, fz, mx, my, mz = [f"{val:.2f}" for val in data]
            force_values = [fx, fy, fz, mx, my, mz]
            for i in range(6):
                self.qpos_inputs[7 + i].setText(force_values[i])  

        except Exception as e:
            print(f"[ERROR] Reading DynPick sensor failed: {e}")

    def populate_usb_ports(self):
        self.usb_port_combo.clear()
        wacoh_sensor.detect_serialPort()
        ports = wacoh_sensor.get_serial_ports()  # From your script
        for port in ports:
            self.usb_port_combo.addItem(port)
        if not ports:
            self.usb_port_combo.addItem("No USB ports found")


    ### Initialize Py Qt
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLA Dataset Recorder")
        self.resize(900, 1100)

        # Set up cameras
        self.BEV_cam_on=True
        self.Wrist_cam_on=True
        self.cap_main = None #cv2.VideoCapture(0)
        self.cap_wrist = None #cv2.VideoCapture(0) 
        # if not self.cap_main.isOpened():
        #     raise RuntimeError("Main camera not detected")
        # if not self.cap_wrist.isOpened():
        #     print("Warning: Wrist camera not detected, using main camera for both.")
        #     self.cap_wrist = self.cap_main  # Fallback to main camera

        self.run_cam_1_btn = QPushButton("BEV cam off")
        self.run_cam_1_btn.setCheckable(True)
        self.run_cam_1_btn.setChecked(False)
        self.run_cam_1_btn.setStyleSheet("background-color: lightcoral;")
        self.run_cam_1_btn.clicked.connect(self.toggle_BEV_cam)

        self.run_cam_2_btn = QPushButton("Wrist cam off")
        self.run_cam_2_btn.setCheckable(True)
        self.run_cam_2_btn.setChecked(False)
        self.run_cam_2_btn.setStyleSheet("background-color: lightcoral;")
        self.run_cam_2_btn.clicked.connect(self.toggle_wrist_cam)

        # Recording buffers
        self.max_frames = 100
        self.frame_count = 0
        self.episode_count = 0
        self.instruction = """prediction_action": ["lock"], "predicted_wire": "red","predicted_terminal": "terminal_0"""  # Initialize empty
        self.instruction_emb = embed_text("default instruction")  # Default embedding for empty case
        self.output_dir = "hucenrotia_TM5_700_screwing_psu_dataset"  
        self.list_image = []
        self.list_wrist_image = []
        self.list_action = []
        self.list_is_edited = []
        self.list_qpos = []
        self.list_qvel = []
        self.list_language_embedding = []

        # Graph data
        self.action_data = {i: np.array([]) for i in range(7)}  # x, y, z, rx, ry, rz, state
        self.qpos_data = {i: np.array([]) for i in range(7)}
        self.F_data = {i: np.array([]) for i in range(7)}
        self.qvel_data = {i: np.array([]) for i in range(7)}
        self.dF_data = {i: np.array([]) for i in range(7)}

        self.time_data = np.array([])

        # UI Elements
        self.main_cam = QLabel("Main Cam")
        self.main_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrist_cam = QLabel("Wrist Cam")
        self.wrist_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel(f"Frames: 0 / {self.max_frames}")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Directory and file name inputs
        self.dir_label = QLabel("Output Directory:")
        self.dir_input = QLineEdit(self.output_dir)
        self.dir_input.setPlaceholderText("Enter output directory (e.g., dataset_folder)")  # Placeholder text
        self.file_label = QLabel("File Name:")
        self.file_input = QLineEdit(f"episode_{self.episode_count}.h5")

        # Max frames input
        self.max_frames_label = QLabel("Max Frames:")
        self.max_frames_input = QSpinBox()
        self.max_frames_input.setRange(10, 1000)
        self.max_frames_input.setValue(self.max_frames)
        self.max_frames_input.setSingleStep(10)

        # DynPick Force
        self.DynPick_label = QLabel("DynPick Force:")
        self.DynPick_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.DynPick_USB_port_label = QLabel("USB Port:")
        self.usb_port_combo = QComboBox()
        self.populate_usb_ports()

        self.DynPick_connect_btn = QPushButton("Disconnected")
        self.DynPick_connect_btn.setCheckable(True)
        self.DynPick_connect_btn.setStyleSheet("background-color: lightcoral;")
        self.DynPick_connect_btn.clicked.connect(self.toggle_dynpick_connection)

        # Bold Observation label
        self.set_label_header = QLabel("Set data")
        self.set_label_header.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Instruction input
        self.instruction_label = QLabel("Instruction:")
        self.instruction_input = QLineEdit(self.instruction)
        self.instruction_input.setPlaceholderText("Enter short instruction (e.g., tighten screw)")  # Placeholder text
        
        # Input fields for action, qpos, qvel
        self.action_label = QLabel("Action (x,y,z,rx,ry,rz,state):")
        self.action_inputs = [QLineEdit("0.0") for _ in range(7)]
        
        
        # Bold Observation label
        self.observation_header = QLabel("Observation")
        self.observation_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.qpos_label = QLabel("Qpos (x, y, z, rx, ry, rz, EE_F_state, Fx, Fy, Fz, Mx, My, Mz):")
        self.qpos_inputs = [QLineEdit("0.0") for _ in range(13)]
        self.qvel_label = QLabel("QVel (x, y, z, rx, ry, rz, EE_F_state, Fx, Fy, Fz, Mx, My, Mz):")
        self.qvel_inputs = [QLineEdit("0.0") for _ in range(13)]
        
        # Bold Observation label
        self.graph_header = QLabel("Graph")
        self.graph_header.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Graph setup
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.showGrid(x=True, y=True)
    

        # Create curves for each component
        self.curves = {}

        # Define unique colors for all 21 curves
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFA500', '#800080',  # Action (7)
            '#FF4500', '#2E8B57', '#4682B4', '#DAA520', '#20B2AA', '#BA55D3', '#CD5C5C',  # Qpos (7)
            '#9ACD32', '#4169E1', '#FF69B4', '#ADFF2F', '#7B68EE', '#FFD700',             # Force (6)
            '#98FB98', '#8B0000', '#00CED1', '#C71585', '#8B4513', '#6A5ACD', '#B0C4DE',  # Qvel (7)
            '#DC143C', '#00FA9A', '#1E90FF', '#FFDAB9', '#9932CC', '#708090'              # dForce (6)
        ]
        
        self.labels = ['Action_x', 'Action_y', 'Action_z', 'Action_rx', 'Action_ry', 'Action_rz', 'Action_state',
                    'Qpos_x', 'Qpos_y', 'Qpos_z', 'Qpos_rx', 'Qpos_ry', 'Qpos_rz', 'Qpos_state',
                    "Fx", "Fy", "Fz", "Mx", "My", "Mz",                      # Force
                    'Qvel_x', 'Qvel_y', 'Qvel_z', 'Qvel_rx', 'Qvel_ry', 'Qvel_rz', 'Qvel_state',
                    "dFx", "dFy", "dFz", "dMx", "dMy", "dMz"                 # Derivative of Force
                ]
                  
        for i, label in enumerate(self.labels):
            self.curves[label] = self.plot_widget.plot(pen=pg.mkPen(self.colors[i], width=2), name=label)

        # Create static legend
        self.legend = LegendObj(curves_dict=self.curves, labels=self.labels, colors=self.colors)
        self.plot_widget.scene().addItem(self.legend)

        def position_legend():
            view_rect = self.plot_widget.getViewBox().sceneBoundingRect()
            legend_rect = self.legend.boundingRect()
            margin = 10
            x = view_rect.right() - legend_rect.width() - margin
            y = view_rect.top() + margin
            self.legend.setPos(x, y)

        # Call on view change or resize
        self.plot_widget.getViewBox().sigResized.connect(position_legend)
        self.plot_widget.sigRangeChanged.connect(lambda _, __: position_legend())
                
        # Bold Filter label
        self.filter_graph_header = QLabel("Filter: ")
        self.filter_graph_header.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.action_check = QCheckBox("Show Action")
        self.action_check.setChecked(True)
        self.qpos_check = QCheckBox("Show Qpos")
        self.qpos_check.setChecked(True)
        self.F_check = QCheckBox("Show F_check")
        self.F_check.setChecked(True)
        
        self.qvel_check = QCheckBox("Show Qvel")
        self.qvel_check.setChecked(True)
        self.dF_check = QCheckBox("Show dF_check")
        self.dF_check.setChecked(True)

        self.capture_btn = QPushButton("Capture Frame")
        self.save_btn = QPushButton("Save to HDF5")
        self.clear_btn = QPushButton("Clear")
        
        self.next_episode_btn = QPushButton("Save and Next Episode")

        # Layout
        layout = QVBoxLayout()
        
        # Camera frames with centered padding
        h_cam = QHBoxLayout()
        h_cam.setContentsMargins(100, 10, 100, 10)
        h_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_cam.setSpacing(50) 

        v_main = QVBoxLayout()
        v_main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_main.addWidget(self.main_cam)
        v_main.addWidget(self.run_cam_1_btn)

        v_wrist = QVBoxLayout()
        v_wrist.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_wrist.addWidget(self.wrist_cam)
        v_wrist.addWidget(self.run_cam_2_btn)

        h_cam.addLayout(v_main)
        h_cam.addLayout(v_wrist)

        layout.addLayout(h_cam)
        layout.addWidget(self.status_label)

        # Directory and file name layout with left padding
        h_dir = QHBoxLayout()
        h_dir.setContentsMargins(10, 10, 0, 0) 
        h_dir.addWidget(self.dir_label)
        h_dir.addWidget(self.dir_input)
        h_dir.addWidget(self.file_label)
        h_dir.addWidget(self.file_input)
        layout.addLayout(h_dir)

        # Max frames layout with left padding
        h_config_widget = QHBoxLayout()
        h_config_widget.setAlignment(Qt.AlignmentFlag.AlignLeft)  
        h_config_widget.setContentsMargins(10, 10, 0, 0) 
        h_config_widget.addWidget(self.max_frames_label)
        h_config_widget.addWidget(self.max_frames_input)
        h_config_widget.addWidget(self.DynPick_label)
        h_config_widget.addWidget(self.DynPick_USB_port_label)
        h_config_widget.addWidget(self.usb_port_combo)       
        h_config_widget.addWidget(self.DynPick_connect_btn)

        layout.addLayout(h_config_widget)

        # set_label section with left padding
        layout.addWidget(self.set_label_header)
        self.set_label_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Instruction layout with left padding
        h_instruction = QHBoxLayout()
        h_instruction.setContentsMargins(10, 10, 0, 0) 
        h_instruction.addWidget(self.instruction_label)
        h_instruction.addWidget(self.instruction_input)
        layout.addLayout(h_instruction)

        # Input layouts with left padding
        h_action = QHBoxLayout()
        h_action.setContentsMargins(10, 10, 0, 0)  
        h_action.addWidget(self.action_label)
        for i, inp in enumerate(self.action_inputs):
            inp.setPlaceholderText(["x", "y", "z", "rx", "ry", "rz", "state"][i])
            inp.setFixedWidth(80)
            h_action.addWidget(inp)
        layout.addLayout(h_action)

        # Observation section with left padding
        layout.addWidget(self.observation_header)
        self.observation_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        h_qpos = QHBoxLayout()
        h_qpos.setContentsMargins(10, 0, 0, 0)  # Left padding
        h_qpos.addWidget(self.qpos_label)
        for i, inp in enumerate(self.qpos_inputs):
            inp.setPlaceholderText(["x", "y", "z", "rx", "ry", "rz", "state","Fx", "Fy", "Fz", "Mx", "My", "Mz"][i])
            inp.setFixedWidth(80)
            inp.setReadOnly(True)
            h_qpos.addWidget(inp)
        layout.addLayout(h_qpos)

        h_qvel = QHBoxLayout()
        h_qvel.setContentsMargins(10, 10, 10, 0)  # Left padding
        h_qvel.addWidget(self.qvel_label)
        for i, inp in enumerate(self.qvel_inputs):
            inp.setPlaceholderText(["x", "y", "z", "rx", "ry", "rz", "state","Fx", "Fy", "Fz", "Mx", "My", "Mz"][i])
            inp.setFixedWidth(80)
            inp.setReadOnly(True)
            h_qvel.addWidget(inp)
        layout.addLayout(h_qvel)

        # Graph section with left padding
        layout.addWidget(self.graph_header)
        self.graph_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Graph layout with left and right padding
        h_graph = QHBoxLayout()
        h_graph.setContentsMargins(10, 0, 10, 0) 
        h_graph.addWidget(self.plot_widget)
        layout.addLayout(h_graph)

        # Filter checkboxes with left padding
        h_filters = QHBoxLayout()
        h_filters.setContentsMargins(10, 10, 0, 0)  
        h_filters.addWidget(self.filter_graph_header)
        h_filters.addWidget(self.action_check)
        h_filters.addWidget(self.qpos_check)
        h_filters.addWidget(self.F_check)
        h_filters.addWidget(self.qvel_check)
        h_filters.addWidget(self.dF_check)
        layout.addLayout(h_filters)

        # Buttons layout with left padding
        h_buttons = QHBoxLayout()
        h_buttons.setContentsMargins(20, 10, 0, 0)  # Left padding
        h_buttons.addWidget(self.capture_btn)
        h_buttons.addWidget(self.save_btn)
        h_buttons.addWidget(self.clear_btn)
        h_buttons.addWidget(self.next_episode_btn)
        layout.addLayout(h_buttons)
        
        self.setLayout(layout)

        # Timer
        self.timer_frame = QTimer()
        if self.BEV_cam_on:
            self.timer_frame.timeout.connect(self.update_BEV_frames)
        if self.Wrist_cam_on:
            self.timer_frame.timeout.connect(self.update_Wrist_frames)
        self.timer_frame.start(30)
        self.force_timer = QTimer()
        self.force_timer.timeout.connect(self.update_force_displays)
        self.force_timer.start(30) 

        # Signals
        self.capture_btn.clicked.connect(self.capture_frame)
        self.save_btn.clicked.connect(self.save_hdf5)
        self.clear_btn.clicked.connect(self.clear_buffer)
        self.next_episode_btn.clicked.connect(self.save_and_next_episode)
        self.instruction_input.textChanged.connect(self.update_instruction)
        self.max_frames_input.valueChanged.connect(self.update_max_frames)
        self.action_check.stateChanged.connect(self.update_graph)
        self.qpos_check.stateChanged.connect(self.update_graph)
        self.F_check.stateChanged.connect(self.update_graph)
        self.qvel_check.stateChanged.connect(self.update_graph)
        self.dF_check.stateChanged.connect(self.update_graph)

    def update_max_frames(self):
        self.max_frames = self.max_frames_input.value()
        self.status_label.setText(f"Frames: {self.frame_count} / {self.max_frames}")

    def update_instruction(self):
        self.instruction = self.instruction_input.text()
        if self.instruction.strip():
            self.instruction_emb = embed_text(self.instruction)
        else:
            self.instruction = "default instruction" 
            self.instruction_emb = embed_text(self.instruction)

    def update_BEV_frames(self):
        if self.cap_main is None:
            return
        ret1, frame1 = self.cap_main.read()
        if ret1:
            self.display_frame(frame1, self.main_cam)

    def update_Wrist_frames(self):
        if self.cap_wrist is None:
            return
        ret2, frame2 = self.cap_wrist.read() 
        if ret2 :
            self.display_frame(frame2 , self.wrist_cam)

    def display_frame(self, frame, label):
        frame_resized = cv2.resize(frame, (320, 240))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def capture_frame(self):
        if self.frame_count >= self.max_frames:
            QMessageBox.warning(self, "Limit", "Maximum frames reached.")
            return

        ret_main, img_main = self.cap_main.read()
        ret_wrist, img_wrist = self.cap_wrist.read() if self.cap_wrist != self.cap_main else (False, img_main)
        if not ret_main:
            QMessageBox.critical(self, "Error", "Failed to capture main camera")
            return
        if not ret_wrist and self.cap_wrist != self.cap_main:
            img_wrist = img_main.copy()

        img_main = cv2.resize(img_main, (320, 240))
        img_wrist = cv2.resize(img_wrist, (320, 240))

        # Validate and get input values
        try:
            action_vals = [float(inp.text()) for inp in self.action_inputs]
            qpos_vals = [float(inp.text()) for inp in self.qpos_inputs[:7]]
            F_vals     = [float(inp.text()) for inp in self.qpos_inputs[7:13]]
            qvel_vals  = [float(inp.text()) for inp in self.qvel_inputs[:7]]
            dF_vals    = [float(inp.text()) for inp in self.qvel_inputs[7:13]]

            # Ensure state (last value) is between 0 and 1
            if not (0 <= action_vals[6] <= 1 and 0 <= qpos_vals[6] <= 1 and 0 <= qvel_vals[6] <= 1):
                raise ValueError("State values must be between 0 and 1")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input: {str(e)}")
            return

        self.list_image.append(img_main)
        self.list_wrist_image.append(img_wrist)
        self.list_action.append(np.array(action_vals, dtype=np.float32))
        self.list_is_edited.append(0)
        self.list_qpos.append(np.array(qpos_vals + F_vals, dtype=np.float32))
        self.list_qvel.append(np.array(qvel_vals + dF_vals, dtype=np.float32))
        self.list_language_embedding.append(self.instruction_emb.copy())

        for i in range(7):
            self.action_data[i] = np.append(self.action_data[i], action_vals[i])
            self.qpos_data[i] = np.append(self.qpos_data[i], qpos_vals[i])
            self.qvel_data[i] = np.append(self.qvel_data[i], qvel_vals[i])
        for i in range(6):
            self.F_data[i] = np.append(self.F_data[i], F_vals[i])
            self.dF_data[i] = np.append(self.dF_data[i], dF_vals[i])

            
        self.time_data = np.append(self.time_data, self.frame_count)

        self.frame_count += 1
        self.status_label.setText(f"Frames: {self.frame_count} / {self.max_frames}")
        print(f"Frame {self.frame_count} captured.")
        self.update_graph()

    def save_hdf5(self):
        if self.frame_count == 0:
            QMessageBox.warning(self, "Empty", "No data to save.")
            return

        output_dir = self.dir_input.text().strip()
        file_name = self.file_input.text().strip()

        # Validate inputs
        if not output_dir:
            QMessageBox.critical(self, "Error", "Output directory is required.")
            return
        if not file_name:
            file_name = f"episode_{self.episode_count}.h5"
        elif not file_name.endswith(".h5"):
            file_name += ".h5"

        os.makedirs(output_dir, exist_ok=True)
        h5_path = os.path.join(output_dir, file_name)

        image_sequence = np.stack(self.list_image, axis=0)
        wrist_sequence = np.stack(self.list_wrist_image, axis=0)
        arr_action = np.stack(self.list_action, axis=0)
        arr_is_edited = np.stack(self.list_is_edited, axis=0)
        arr_qpos = np.stack(self.list_qpos, axis=0)
        arr_qvel = np.stack(self.list_qvel, axis=0)
        arr_lang_emb = np.stack(self.list_language_embedding, axis=0)

        with h5py.File(h5_path, 'w') as hf:
            obs = hf.create_group('observations')
            img_grp = obs.create_group('images')
            img_grp.create_dataset('image', data=image_sequence)
            img_grp.create_dataset('wrist_image', data=wrist_sequence)
            hf.create_dataset('action', data=arr_action)
            hf.create_dataset('is_edited', data=arr_is_edited)
            hf.create_dataset('language_embedding', data=arr_lang_emb)
            hf.create_dataset('language_raw', data=np.string_(self.instruction))
            obs.create_dataset('qpos', data=arr_qpos)
            obs.create_dataset('qvel', data=arr_qvel)

        QMessageBox.information(self, "Success", f"Saved to {h5_path}")
        print(f"Saved {self.frame_count} frames.")

    def save_and_next_episode(self):
        self.save_hdf5()
        self.clear_buffer()
        self.episode_count += 1
        self.file_input.setText(f"episode_{self.episode_count}.h5")
        print(f"Starting new episode: episode_{self.episode_count}.h5")

    def clear_buffer(self):
        self.list_image.clear()
        self.list_wrist_image.clear()
        self.list_action.clear()
        self.list_is_edited.clear()
        self.list_qpos.clear()
        self.list_qvel.clear()
        self.list_language_embedding.clear()
        for i in range(7):
            self.action_data[i] = np.array([])
            self.qpos_data[i] = np.array([])
            self.qvel_data[i] = np.array([])
        for i in range(6):
            self.F_data[i] = np.array([])
            self.dF_data[i] = np.array([])

        self.time_data = np.array([])
        self.frame_count = 0
        self.status_label.setText(f"Frames: 0 / {self.max_frames}")
        self.update_graph()
        print("Cleared buffer.")

    def update_graph(self):
        visible_labels = []

        # Count how many groups are active to decide column count
        active_groups = sum([
            self.action_check.isChecked(),
            self.qpos_check.isChecked(),
            self.F_check.isChecked(),
            self.qvel_check.isChecked(),
            self.dF_check.isChecked()

        ])
        self.legend.column_count = max(1, active_groups)

        # Action group
        if self.action_check.isChecked():
            action_labels = ['Action_x', 'Action_y', 'Action_z', 'Action_rx', 'Action_ry', 'Action_rz', 'Action_state']
            for i, label in enumerate(action_labels):
                if len(self.time_data) == len(self.action_data[i]):
                    self.curves[label].setData(self.time_data, self.action_data[i])
                self.curves[label].setVisible(True)
                visible_labels.append(label)
        else:
            for label in ['Action_x', 'Action_y', 'Action_z', 'Action_rx', 'Action_ry', 'Action_rz', 'Action_state']:
                self.curves[label].setVisible(False)

        # Qpos group
        if self.qpos_check.isChecked():
            qpos_labels = ['Qpos_x', 'Qpos_y', 'Qpos_z', 'Qpos_rx', 'Qpos_ry', 'Qpos_rz', 'Qpos_state']
            for i, label in enumerate(qpos_labels):
                if len(self.time_data) == len(self.qpos_data[i]):
                    self.curves[label].setData(self.time_data, self.qpos_data[i])
                self.curves[label].setVisible(True)
                visible_labels.append(label)
        else:
            for label in ['Qpos_x', 'Qpos_y', 'Qpos_z', 'Qpos_rx', 'Qpos_ry', 'Qpos_rz', 'Qpos_state']:
                self.curves[label].setVisible(False)
        # F group
        if self.F_check.isChecked():
            F_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            for i, label in enumerate(F_labels):
                if len(self.time_data) == len(self.F_data[i]):
                    self.curves[label].setData(self.time_data, self.F_data[i])
                self.curves[label].setVisible(True)
                visible_labels.append(label)
        else:
            for label in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
                self.curves[label].setVisible(False)

        # Qvel group
        if self.qvel_check.isChecked():
            qvel_labels = ['Qvel_x', 'Qvel_y', 'Qvel_z', 'Qvel_rx', 'Qvel_ry', 'Qvel_rz', 'Qvel_state']
            for i, label in enumerate(qvel_labels):
                if len(self.time_data) == len(self.qvel_data[i]):
                    self.curves[label].setData(self.time_data, self.qvel_data[i])
                self.curves[label].setVisible(True)
                visible_labels.append(label)
        else:
            for label in ['Qvel_x', 'Qvel_y', 'Qvel_z', 'Qvel_rx', 'Qvel_ry', 'Qvel_rz', 'Qvel_state']:
                self.curves[label].setVisible(False)
        
        # F group
        if self.dF_check.isChecked():
            dF_labels = ["dFx", "dFy", "dFz", "dMx", "dMy", "dMz"]
            for i, label in enumerate(dF_labels):
                if len(self.time_data) == len(self.dF_data[i]):
                    self.curves[label].setData(self.time_data, self.dF_data[i])
                self.curves[label].setVisible(True)
                visible_labels.append(label)
        else:
            for label in ["dFx", "dFy", "dFz", "dMx", "dMy", "dMz"]:
                self.curves[label].setVisible(False)

        # Update legend items and layout
        self.legend.labels = visible_labels
        self.legend.staticLayoutItems()

        # Reposition legend to right-10
        view_rect = self.plot_widget.getViewBox().sceneBoundingRect()
        legend_rect = self.legend.boundingRect()
        margin = 10
        margin_filter = ((3-active_groups)*self.legend.boundingRect().width())/3
        x = view_rect.right() - legend_rect.width() - margin- margin_filter
        y = view_rect.top()+ margin
        self.legend.setPos(x, y)

    def closeEvent(self, event):
        self.cap_main.release()
        if self.cap_wrist != self.cap_main and self.cap_wrist.isOpened():
            self.cap_wrist.release()
        super().closeEvent(event)

# ------------------ Run Application -------------------
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = DataRecorder()
    window.show()
    sys.exit(app.exec())