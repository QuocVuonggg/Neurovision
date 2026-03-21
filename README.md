# 🧠 NeuroVision: AI-Assisted MRI Diagnostic Workspace

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![PyQt6](https://img.shields.io/badge/PyQt6-Desktop%20GUI-41cd52?logo=qt&logoColor=white)
![Hackathon](https://img.shields.io/badge/Track-Social%20%26%20Mobility%20(TASCO)-ff69b4)

## 📖 Overview
**NeuroVision** is an end-to-end medical desktop application designed to tackle a pressing social challenge: **healthcare accessibility**. Built for the TASCO "Social & Mobility" track, this software democratizes advanced neurological diagnostics by equipping frontline clinics and radiologists with a state-of-the-art AI assistant.

By leveraging a custom **UNet architecture** for high-precision brain abnormality localization and **Grad-CAM** for explainable heatmaps, NeuroVision ensures that AI remains transparent. Combined with a "Human-in-the-Loop" workflow, an offline patient database, and automated clinical reporting, the system significantly reduces cognitive fatigue for doctors while keeping them in full control of the final diagnosis.

## ✨ Core Features
* 🚀 **Zero-Latency Batch Processing:** Drag and drop multiple MRI scans (DICOM, PNG, JPG). The AI runs asynchronously in the background.
* 🧠 **UNet & Explainable AI:** Accurately segments brain lesions and provides Grad-CAM heatmaps to visualize the AI's focal points (Red/Green bounding boxes).
* 👨‍⚕️ **Human-in-the-Loop (Doctor Override):** A built-in clinical text editor allows physicians to manually correct AI findings and append clinical notes.
* 📂 **Local Patient Database (Local PACS):** Automatically organizes patient demographics, scans, and histories into a searchable, sortable, and secure offline JSON database.
* 📄 **Automated PDF Reporting:** One-click generation of professional medical reports containing patient demographics, annotated MRI scans, and physician signatures.

## 🗂️ Project Structure
The project is organized in a modular structure to separate the UI layer from the Deep Learning engine.

```text
.HACKATHON/
│
├── app.py                 # Main Application GUI & Workflow Logic (PyQt6)
├── mri_analyzer.py        # AI Inference Engine & Grad-CAM Generator
├── unet_model.py          # UNet Deep Learning Architecture Definition
├── unet_dataset.py        # Dataset Loader & Preprocessing scripts
├── train_unet.py          # Script used for training the UNet model
│
├── models/                
│   └── unet_best.pth      # Pre-trained UNet weights (Core Model)
│
├── patient_database/      # Auto-generated offline database for patient records
│   └── PID-XXXXX/         # Patient-specific folders containing JSON data & images
│
├── data/                  # Directory for sample/input MRI scans
├── requirements.txt       # Python dependencies
└── brain.ico              # Application Icon