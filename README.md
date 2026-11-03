# ASD Video Classification — Flask Backend

An academic backend that accepts a video and returns an experimental
ASD/TD classification using VGG16 feature extraction and a trained LSTM.

This project is not a clinically validated diagnostic or screening tool.
Model scores are not clinical probabilities.

## Requirements

- Python 3.10
- Packages listed in requirements.txt
- Saved model: models/ASD_VGG16_LSTM_model.keras
- Internet access for the first download of VGG16 ImageNet weights
  (approximately 528 MB)

The original training dataset is not required for inference.

## Project files

- app.py — Flask routes, upload validation, and error handling
- ml_service.py — frame extraction and model inference
- test_prediction.py — standalone test with a local video path
- requirements.txt — pinned main dependencies
- models/ — saved model files
- uploads/ — uploaded files and temporary prediction files

The PCA + SVM model is not integrated into the current API.

## Setup on another computer

Open PowerShell in the project folder.

Create a Python 3.10 virtual environment:

```powershell
py -3.10 -m venv .venv
```

Install dependencies:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Place ASD_VGG16_LSTM_model.keras inside the models folder.
VGG16 ImageNet weights are downloaded automatically on first use.

## Start the backend

With the virtual environment above:

```powershell
& ".\.venv\Scripts\python.exe" app.py
```

On the original development laptop:

```powershell
& "C:\Users\gayat\anaconda3\envs\asd_backend\python.exe" app.py
```

Expected address:

http://127.0.0.1:5000

Keep the terminal running while using the API.
Press Ctrl+C to stop it.

The current configuration uses Flask's development server with debug
enabled. It is intended for local development. Production deployment
requires debug to be disabled and a production server configured.

## API endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | / | Backend welcome message |
| GET | /health | Service availability check |
| POST | /upload | Save a video without prediction |
| POST | /predict | Process a video and return model output |

The health endpoint checks service availability; it does not verify
that model loading or prediction succeeds.

## Prediction request

Send POST to:

http://127.0.0.1:5000/predict

In Postman:

1. Select Body → form-data.
2. Add a key named exactly video.
3. Change the row type to File.
4. Select a video and keep the row checkbox checked.
5. Click Send.

Let Postman generate the multipart Content-Type header automatically.

Allowed filename extensions:
.mp4, .avi, .mov, .mkv, .mpeg, .mpg

Maximum total request size: 100 × 1024 × 1024 bytes.
Multipart overhead counts toward this limit.

A supported extension alone does not guarantee a readable video.

## Example successful response

The following scores are an example from one test video.
Other videos can produce different results.

```json
{
  "experimental": true,
  "frames_extracted": 5,
  "model_scores": {
    "ASD": 0.5143538117408752,
    "TD": 0.48564621806144714
  },
  "notice": "Academic experiment only. Not a clinically validated diagnostic or screening tool. Scores are not clinical probabilities.",
  "predicted_class": "ASD"
}
```

The class with the larger model score is returned.
These scores do not establish a medical diagnosis.

## Error responses

| Status | Meaning |
|---|---|
| 400 | Missing video, no selected file, unsupported extension, invalid filename, or empty file |
| 413 | Total request exceeds the upload-size limit |
| 422 | Video cannot be opened or sampled frames cannot be read |
| 500 | Upload saving, model loading, prediction, or another server operation failed |

Detailed failures are logged in the backend terminal.
Internal tracebacks are not included in the API's JSON error response.

## Processing pipeline

1. Sample five evenly spaced frame indices across the video.
2. Resize successfully read frames to 224 × 224.
3. Apply the preprocessing and color handling used in proper_train.py.
4. Extract 1,000 VGG16 features per frame.
5. Pad missing sampled frames with zero features.
6. Pass input shaped (1, 5, 1000) to the LSTM.
7. Interpret output index 0 as ASD and index 1 as TD.

Models are loaded lazily and reused within each server process.
The first prediction can take longer than later requests.

## File handling

/predict attempts to delete its temporary uploaded video after either
success or failure. Cleanup failures are logged.

/upload retains saved files. It validates the filename extension but
does not run video decoding or prediction.

## Frontend integration

The frontend should:

- Send the actual file using multipart FormData with key video.
- Call POST /predict.
- Show a loading indicator while processing.
- Display predicted_class, model_scores, and notice.
- Display the API error message when a request fails.
- Avoid manually setting Content-Type when sending browser FormData.

127.0.0.1 refers to the computer opening the address.
A frontend running on another computer needs a reachable backend address.

CORS is currently enabled broadly for development.
Configure allowed frontend origins before deployment.

## Checks completed during development

- Valid video prediction
- Missing-video rejection
- Unsupported-extension rejection
- Unreadable-video rejection
- Empty-file rejection
- Oversized-request rejection
- Temporary-file cleanup observed after successful and failed processing
- Missing-model error handling
- Simulated model-loading ValueError returned as HTTP 500
- Dependency check: no broken requirements found
- Absolute upload-folder path verified

These checks verify backend behavior, not clinical validity or model accuracy.

## Remaining work

- Verify installation and startup on another computer
- Connect and test the frontend when available
- Configure hosting and production settings if online access is required