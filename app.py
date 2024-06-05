from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
# from pose_detector import PoseDetector  # Import your PoseDetector class
import cv2
import time
import numpy as np
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.85, trackCon=0.85):
        # Initialization code for PoseDetector (same as in your original script)
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def findPose(self, img, draw=True):
        # Method to find pose landmarks in an image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        # Method to find position of pose landmarks
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (125,251,5), cv2.FILLED)
        return self.lmList
        
def overlay_skeletal_model(benchmark_video):
    benchmark_cam = cv2.VideoCapture(benchmark_video)
    user_cam = cv2.VideoCapture(0)  # Accessing webcam, change index if needed

    detector = PoseDetector()

    fps_time = 0
    frame_counter = 0
    correct_frames = 0

    while True:
        ret_val_benchmark, image_benchmark = benchmark_cam.read()
        ret_val_user, image_user = user_cam.read()

        if not ret_val_benchmark or not ret_val_user:
            break

        # Resize images
        image_user = cv2.resize(image_user, (720, 640))

        # Find poses for user's video
        image_user_with_pose = detector.findPose(image_user)
        lmList_user = detector.findPosition(image_user_with_pose)
        del lmList_user[1:11]

        if lmList_user:
            if ret_val_benchmark:  # Only compute error if there's a benchmark frame
                # Resize and process benchmark frame
                image_benchmark = cv2.resize(image_benchmark, (720, 640))
                image_benchmark_with_pose = detector.findPose(image_benchmark)
                lmList_benchmark = detector.findPosition(image_benchmark_with_pose)
                del lmList_benchmark[1:11]

                if lmList_benchmark:
                    error, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)
                    # Displaying the error percentage
                    cv2.putText(image_user_with_pose, 'Error: {}%'.format(str(round(100 * float(error), 2))), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # If the similarity is > 90%, take it as correct step. Otherwise incorrect step.
                    if error < 0.3:
                        cv2.putText(image_user_with_pose, "CORRECT POSTURE", (40, 600),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        correct_frames += 1
                    else:
                        cv2.putText(image_user_with_pose, "INCORRECT POSTURE", (40, 600),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    frame_counter += 1

                    # Draw landmarks and connections
                    if error > 0.3:
                        for lm in lmList_benchmark:
                            cv2.circle(image_user_with_pose, (lm[1], lm[2]), 5, (0, 0, 255), cv2.FILLED)
                        # Specify the color of the points
                        landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                                       circle_radius=2)

                        # Draw landmarks with specified color
                        detector.mpDraw.draw_landmarks(image_user, detector.results.pose_landmarks,
                                                       detector.mpPose.POSE_CONNECTIONS, landmark_drawing_spec)

                    else:
                        for lm in lmList_benchmark:
                            cv2.circle(image_user_with_pose, (lm[1], lm[2]), 5, (125, 251, 5), cv2.FILLED)
                        # Specify the color of the points
                        landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(125, 251, 5), thickness=2,
                                                                                       circle_radius=2)

                        # Draw landmarks with specified color
                        detector.mpDraw.draw_landmarks(image_user, detector.results.pose_landmarks,
                                                       detector.mpPose.POSE_CONNECTIONS, landmark_drawing_spec)

            # Display user's video with overlay
            cv2.putText(image_user_with_pose, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if frame_counter != 0:  # Ensure frame_counter is not zero to avoid division by zero
                cv2.putText(image_user_with_pose, "Dance Steps Accurately Done: {}%".format(
                    str(round(100 * correct_frames / frame_counter, 2))),
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # cv2.imshow('User Video', image_user_with_pose)
            # Display the video in fullscreen mode
            cv2.namedWindow('User Video', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('User Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('User Video', image_user_with_pose)


            fps_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    benchmark_cam.release()
    user_cam.release()
    cv2.destroyAllWindows()



app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pose Estimation</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            height: 100vh;
            margin: 0;
            background: linear-gradient(to bottom, #3498db, #2c3e50);
        }

        #sidebar {
            width: 150px;
            background: #34495e;
            padding-top: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
        }

        #app {
            text-align: center;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        button, input, label {
            padding: 12px 20px;
            font-size: 18px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            margin: 10px 0;
            transition: background-color 0.3s ease, color 0.3s ease, filter 0.3s ease;
            width: 100%;
            box-sizing: border-box;
        }

        #cameraBtn {
            background-color: #2ecc71;
            color: #fff;
        }

        #recordBtn, #uploadBtn, #removeBtn {
            background-color: #e67e22;
            color: #fff;
        }

        #closeBtn {
            background-color: #c0392b;
            color: #fff;
        }

        button:hover, input:hover, label:hover {
            filter: brightness(90%);
        }

        video {
            width: 100%;
            max-width: 600px;
            border: 2px solid #2ecc71;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            margin-top: 20px;
        }

        input[type="file"] {
            display: none;
        }

        .upload-section {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .upload-btn-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }

        .btn {
            border: 2px solid gray;
            color: gray;
            background-color: white;
            padding: 8px 20px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
        }

        .upload-btn-wrapper input[type=file] {
            font-size: 100px;
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
        }

        .file-name {
            margin-left: 8px;
        }
    </style>
</head>
<body>
  
    <div id="app">
            <h1>Motion Detection Web App</h1>
            <form method="post" enctype="multipart/form-data" action="/detect-pose">
                <input type="file" name="video" accept="video/mp4" style="display: block;"placeholder = "Upload">
                <button type="submit">Upload Video</button>
            </form>
        <video id="videoElement" autoplay playsinline></video>
    </div>

    <script>
        let videoStream;
        let mediaRecorder;
        let recordedChunks = [];

        function startCamera() {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function (stream) {
                        videoStream = stream;
                        var videoElement = document.getElementById('videoElement');
                        videoElement.srcObject = stream;
                        document.getElementById('recordBtn').disabled = false;
                        document.getElementById('uploadBtn').disabled = false;
                    })
                    .catch(function (error) {
                        console.error('Error accessing camera: ', error);
                    });
            } else {
                alert('Your browser does not support the WebRTC API. Please use a modern browser.');
            }
        }

        function startRecording() {
            if (videoStream) {
                mediaRecorder = new MediaRecorder(videoStream);

                mediaRecorder.ondataavailable = function (event) {
                    if (event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = function () {
                    document.getElementById('uploadBtn').disabled = false;
                    document.getElementById('removeBtn').disabled = false;
                };

                mediaRecorder.start();
                document.getElementById('recordBtn').disabled = true;
            }
        }

        function handleFileUpload() {
            const fileInput = document.getElementById('fileInput');
            const selectedFile = fileInput.files[0];

            if (selectedFile) {
                const videoElement = document.getElementById('videoElement');
                videoElement.src = URL.createObjectURL(selectedFile);
                document.getElementById('recordBtn').disabled = true;
                document.getElementById('uploadBtn').disabled = false;
                document.getElementById('removeBtn').disabled = false;
            }
        }

        # function uploadVideo() {
        #     const fileInput = document.getElementById('fileInput');
        #     const selectedFile = fileInput.files[0];

        #     if (selectedFile) {
        #         // Perform the upload action, for example, send the file to a server.
        #         alert('Uploading: ' + selectedFile.name);
        #     }
        # }

        function removeVideo() {
            const videoElement = document.getElementById('videoElement');
            videoElement.src = '';
            document.getElementById('recordBtn').disabled = false;
            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('removeBtn').disabled = true;
        }

        function closeCamera() {
            if (videoStream) {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                }
                videoStream.getTracks().forEach(track => track.stop());
                var videoElement = document.getElementById('videoElement');
                videoElement.srcObject = null;
                document.getElementById('recordBtn').disabled = true;
                document.getElementById('uploadBtn').disabled = true;
                document.getElementById('removeBtn').disabled = true;
            }
        }
    </script>
</body>
</html>
    """
@app.post("/detect-pose")
async def detect_pose(video: UploadFile = File(...)):
    # Save uploaded video temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(await video.read())
    
    # Perform pose detection on the uploaded video
    overlay_skeletal_model("temp_video.mp4")  # Call your pose detection function

    # Return the result (e.g., display a processed video or send JSON data)
    # For demonstration, return a simple message
    return {"message": "Pose detection completed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
