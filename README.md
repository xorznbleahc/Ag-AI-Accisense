# 🚦 Accisense

**An AI that watches the road, so responders don't have to wait for someone to call.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c)
![YOLO](https://img.shields.io/badge/detector-Ultralytics%20YOLO-00b8d4)
![Status](https://img.shields.io/badge/status-hackathon%20prototype-orange)

Accisense analyzes CCTV video, tracks every vehicle on the road, and flags the moment two of them appear to collide. It is built as a **decision-support tool**: the AI raises the flag, and a trained person confirms it.

---

## 🧭 The problem

Road accidents need a fast response, but today the first alert often depends on luck.

- Authorities rely on witnesses, manual emergency calls, or staff watching CCTV feeds. Nobody can watch every camera at once.
- Some bystanders record an accident instead of reporting it.
- Reports arrive late, incomplete, or not at all.

The scale of the problem (figures from our project paper, to be verified against the original sources):

- **~1.16 million** people die in road crashes every year worldwide (WHO).
- **20 to 50 million** more suffer non-fatal injuries each year (WHO).
- The Philippines recorded **31,258 road crashes and 2,747 deaths in 2024** (PNP Highway Patrol Group, as reported by Philstar).

Cities already have the cameras. What they lack is something that watches all of them, all the time.

## 💡 Our solution

Accisense turns existing CCTV into an automatic accident spotter. It:

1. **Detects and tracks** every vehicle in the video.
2. **Recognizes collisions** from how vehicles interact and move over time, not from a single frame.
3. **Flags** the incident with a confidence score, the time, and the vehicles involved.
4. Leaves the **final decision to a human reviewer**. AI detections are treated as leads, never as confirmed accidents.

## ⚙️ How it works

```mermaid
flowchart LR
    A[CCTV / video feed] --> B[YOLO vehicle detection<br/>GPU, FP16]
    B --> C[ByteTrack<br/>vehicle tracking]
    C --> D[Accident engine<br/>collision recognition]
    D --> E[Alert + evidence<br/>time, vehicles, confidence]
    E -.-> F[Human review<br/>planned dashboard]
    F -.-> G[Notify responders<br/>planned]
```

*Solid arrows are built today. Dotted arrows are on the roadmap.*

### The accident engine

A collision is flagged when **all three** of these happen together and persist for several frames:

| Signal | What it means |
|---|---|
| **Contact** | Two vehicle boxes overlap |
| **Sudden change** | At least one vehicle abruptly stops or changes direction |
| **Closing speed** | They were approaching each other fast, so slow queueing traffic doesn't trigger it |

Details that make it more robust:

- Speeds are measured in **vehicle-lengths per second**, so the logic is less sensitive to camera zoom and perspective.
- **ID re-linking:** trackers often lose and re-number vehicles at the moment of impact. Accisense recognizes a new ID that appears where a touching vehicle just vanished, and keeps its motion history.
- **Time-based, not frame-based:** results don't depend on how fast your computer is.
- **Cool-down:** vehicles in a reported crash are muted for 20 s, so one pile-up doesn't create a flood of alerts.

## ✨ Features

**Working today**

- ✅ GPU-accelerated vehicle detection and tracking (YOLO + ByteTrack)
- ✅ Collision recognition from vehicle interactions over time
- ✅ Real-time playback of video files and RTSP camera streams, with on-screen alerts
- ✅ Debug mode that shows *why* each vehicle pair was or wasn't flagged (scores plus a CSV log)
- ✅ Dataset tools: frame extraction, auto-labeling, dataset inspection, class remapping, and a training helper for building a custom detector (vehicles, damaged vehicles, debris)

**Planned**

- 🛠 Incident records with snapshot and short video clip
- 🛠 Web dashboard for review, history, and multiple cameras
- 🛠 Emergency notification after human confirmation
- 🛠 Learned accident classifier trained on tracked vehicle data

## 🧰 Tech stack

Python, PyTorch (CUDA), Ultralytics YOLO, ByteTrack, OpenCV, NumPy.

## 🚀 Getting started

**Requirements:** Python 3.10+, an NVIDIA GPU is recommended (it also runs on CPU, more slowly).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install the CUDA build of PyTorch FIRST (command from https://pytorch.org/get-started/locally/)
#    Skipping this can leave you with a CPU-only PyTorch.

# 3. Install the rest
pip install ultralytics opencv-python numpy

# 4. Check the GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Run it:**

```bash
python detect_accidents.py path/to/video.mp4
```

Press **Q** to quit. A red **ACCIDENT DETECTED** banner appears for 5 seconds after a detection.

| Option | What it does |
|---|---|
| `--speed 0.5` | Playback speed for video files (`1` = real time, `0` = as fast as possible) |
| `--weights yolo26s.pt` | Choose the YOLO model, or point to your own trained `best.pt` |
| `--conf 0.15` | Detection confidence (low values help the tracker keep vehicles through occlusion) |
| `--classes 0,4` | Class ids to track (default `2,3,5,7` = car, motorcycle, bus, truck for COCO models) |
| `--debug` | Show per-pair scores on screen and write `debug_pairs.csv` |
| `--save out.mp4` | Save the annotated video |

A camera stream works too: `python detect_accidents.py "rtsp://user:password@camera-ip:554/stream"`. **Never commit camera URLs or credentials to the repository.**

## 🏋️ Train your own detector

Off-the-shelf models only know everyday objects, so a detached car door can be mistaken for a car. `dataset_tools.py` helps you build a dataset with extra classes such as `damaged_vehicle` and `debris`:

```bash
python dataset_tools.py frames --videos clips --out raw       # pull sharp, non-duplicate frames
python dataset_tools.py autolabel --images raw                # pre-label normal vehicles
#   ...review and add damaged_vehicle / debris labels in a labeling tool...
python dataset_tools.py split --images raw --val 0.2          # split by whole clip, writes data.yaml
python dataset_tools.py train                                 # train on your GPU
python dataset_tools.py inspect --dataset some_dataset        # vet a downloaded dataset
python dataset_tools.py remap --dataset some_dataset --map "0=car" --out converted --dedupe
```

## 🔬 Status and honest limitations

Accisense is a **hackathon prototype**.

- The detection thresholds are **starting values that have not yet been tuned or measured** on a large set of real accident and normal-traffic clips. We have **no accuracy figures yet**, and we would rather say so than invent them.
- It works best with a **fixed camera** (CCTV style). A moving or dashcam view breaks the speed logic.
- The collision rules were checked on synthetic scenarios and a small number of real clips. Motorcycle collisions, where the tracker often loses the vehicle at impact, are an area we are actively improving.
- Expect false alarms and misses until it is evaluated on more footage. That is why a human always reviews a flag.

## 🗺️ Roadmap

1. Build an evaluation set (accident clips plus many hours of normal traffic) and measure detection rate and false alarms per hour.
2. Tune the thresholds on one part of the clips and report results on clips the system has never seen.
3. Train a detector with `damaged_vehicle` and `debris` classes and use them as extra evidence.
4. Add incident records (snapshot, clip), the review dashboard, and confirmed-incident notifications.
5. Replace hand-set rules with a learned classifier if the data supports it.

## 🔒 Privacy, data, and licenses

- **Privacy.** CCTV footage can identify people and vehicles. Any real deployment must follow the Philippine Data Privacy Act (RA 10173) and National Privacy Commission rules on CCTV and data sharing, and should restrict access to footage and incident records.
- **Datasets.** Public accident datasets we looked at are generally research-only or have unclear image sources, so we use them for **prototyping only**. A production model must be trained on data we own or are licensed to use.
- **Model license.** Accisense uses Ultralytics YOLO, which is licensed under AGPL-3.0 (an Enterprise license is available for commercial use). Check the terms before any commercial deployment.
- Our code's license: see the `LICENSE` file.

## 👥 Team

| Name | Role |
|---|---|
| _Add team member_ | _Role_ |
| _Add team member_ | _Role_ |

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection and tracking
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the tracking approach
- Public datasets used for prototyping and research, including TUM Traffic Accid3nD (non-commercial license)
- WHO and the PNP Highway Patrol Group for the road safety statistics cited above
