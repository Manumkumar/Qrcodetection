# QR Code Detection & API Integration

A Python application for real-time QR code detection using a camera feed, with automatic zoom adjustment, deduplication, and integration with a remote API. Designed for high-resolution cameras and reliable scanning in challenging environments.

## Features

- **Real-Time QR Detection:** Uses OpenCV and Pyzbar for fast, accurate QR code scanning from a camera.
- **Auto Zoom:** Dynamically adjusts digital zoom to keep detected QR codes at optimal size.
- **Noise & Duplicate Handling:** Multiple-scan verification and deduplication to avoid false positives or repeated scans.
- **API Integration:** Sends verified QR code data to a configurable remote API endpoint.
- **Snapshot Capture:** Periodically saves frames with detected QR codes for audit or debugging.
- **Robust Camera Recovery:** Automatically reconnects if the camera disconnects.
- **Logging:** Console logging for scan results, API status, and errors.

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- Pyzbar
- Requests
- NumPy

Install dependencies with:

```bash
pip install opencv-python pyzbar requests numpy
```

## Usage

1. Clone the repository.
2. Connect your camera.
3. Run the main script:

```bash
python main.py
```

- Press `q` or `Q` to quit.
- Snapshots are saved in the `snapshots` folder.

## Configuration

Edit the following constants at the top of `main.py` as needed:

- `CAMERA_INDEX`: Index of the camera to use.
- `API_URL`: Remote API endpoint for QR code data.
- `CAMERA_NAME`: Identifier for this scanner (sent with each scan).
- `FRAME_WIDTH` / `FRAME_HEIGHT`: Capture resolution.
- Other parameters control zoom, deduplication, and snapshot interval.

## How It Works

1. **Camera Initialization:** Sets resolution and autofocus.
2. **Frame Loop:** 
    - Reads a frame from the camera.
    - Attempts QR detection (with preprocessing if needed).
    - If a QR code is found and verified:
        - Draws rectangle and label on screen.
        - Sends data to API with timestamp and scanner ID.
        - Saves a snapshot at set intervals.
    - Adjusts zoom automatically to maximize detection.
    - Displays FPS and zoom status on the preview window.

## API Format

On each scan, the following JSON is POSTed to the API:

```json
{
  "qr_content": "detected-qr-code",
  "scanner_id": "CAMERA_NAME",
  "timestamp": "YYYY-MM-DDTHH:MM:SS"
}
```

## Customization

- Change detection logic, snapshot frequency, or API payload in `main.py`.
- Integrate with other services (databases, notifications) as needed.

## Troubleshooting

- Make sure your camera is detected by OpenCV.
- If API requests fail, check your endpoint and network connection.
- For high CPU usage, lower the resolution or frame rate.

## License

(Not specified. Please add a license if desired.)

---

**Author:** [Manumkumar](https://github.com/Manumkumar)
