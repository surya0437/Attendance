from flask import (
    Flask,
    Response,
    render_template_string,
    jsonify,
    redirect,
    url_for,
)
import cv2
import os
from PIL import Image
import numpy as np
from flask_mysqldb import MySQL
import MySQLdb.cursors
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta


app = Flask(__name__)

# Use a generated secret key
app.secret_key = os.urandom(24)

# Ensure the 'data' directory exists in the same directory as the script
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Counter for images
image_counter = 0
max_images = 100
user_recognized = False
classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

app.config['MYSQL_HOST'] = 'localhost'  # Your MySQL host
app.config['MYSQL_USER'] = 'root'       # Your MySQL username
app.config['MYSQL_PASSWORD'] = ''  # Your MySQL password
app.config['MYSQL_DB'] = 'mycollege'
mysql = MySQL(app)


# ======================================================== Check Face Existence =============================================


# def gen_check_face_frames():
#     global image_counter
#     userId = None

#     # Load face recognizer and the trained classifier
#     clf = cv2.face.LBPHFaceRecognizer_create()
#     try:
#         clf.read("trainedClassifier.xml")  # Make sure this file exists
#     except cv2.error as e:
#         print(f"Error loading classifier: {e}")
#         return

#     # Load Haar Cascade for face detection
#     classifier = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, clf):
#         nonlocal userId
#         gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         features = classifier.detectMultiScale(
#             gray_image, scaleFactor, minNeighbours)
#         coords = []

#         for x, y, w, h in features:
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             id, predict = clf.predict(gray_image[y: y + h, x: x + w])
#             userId = id
#             confidence = int((100 * (1 - predict / 300)))

#             if confidence > 77:
#                 coords.append((x, y, w, h))
#         return coords

#     def recognize(img, clf, classifier):
#         coords = draw_boundary(img, classifier, 1.1, 10, (0, 255, 0), clf)
#         return bool(coords)

#     video_capture = cv2.VideoCapture(0)
#     i = 0
#     while True:
#         i += 1
#         success, frame = video_capture.read()

#         if not success:
#             print("Failed to grab frame")
#             break
#         else:
#             user = recognize(frame, clf, classifier)
#             if cv2.waitKey(1) == 13 or i == 50:
#                 break

#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode(".jpg", frame)
#             if not ret:
#                 print("Failed to encode frame")
#                 break
#             frame = buffer.tobytes()

#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

#     video_capture.release()

#     if user:
#         print("Login")
#         print(userId)
#         return redirect(url_for('error'))
#     else:
#         print("Logout")


# @app.route("/error", methods=["GET"])
# def error():
#     return "<h1>Face already exists. Redirected to error page!</h1>"


# @app.route("/check_face_video_feed")
# def check_face_video_feed():
#     frames_generator = gen_check_face_frames()
#     if frames_generator is None:
#         return redirect(url_for("checkFaceExistence"))
#     return Response(frames_generator, mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/checkFaceExistence")
# def check_face_existence():
#     return render_template_string(
#         """
#         <!doctype html>
#         <html lang="en">
#         <head>
#             <meta charset="UTF-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#             <title>Video Streaming</title>
#         </head>
#         <body>
#             <h1>Mark Attendance</h1>
#             <img src="{{ url_for('check_face_video_feed') }}" width="640" height="480">
#             <script>
#                 const maxImages = {{ max_images }};

#                 const checkImageCount = setInterval(() => {
#                     fetch('/image_count')
#                         .then(response => response.json())
#                         .then(data => {
#                             if (data.count >= maxImages) {
#                                 clearInterval(checkImageCount);
#                                 window.location.href = "http://127.0.0.1:8000/admin/addFace";
#                             }
#                         }).catch(error => console.error('Error:', error));
#                 }, 1000); // Check every second

#                 document.addEventListener('keydown', function(event) {
#                     if (event.keyCode === 13) {
#                         window.location.reload();
#                     }
#                 });
#             </script>
#         </body>
#         </html>
#     """
#     )


# def gen_check_face_frames():
#     global image_counter
#     userId = None

#     # Load face recognizer and the trained classifier
#     clf = cv2.face.LBPHFaceRecognizer_create()
#     try:
#         clf.read("trainedClassifier.xml")  # Make sure this file exists
#     except cv2.error as e:
#         print(f"Error loading classifier: {e}")
#         return

#     # Load Haar Cascade for face detection
#     classifier = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, clf):
#         nonlocal userId
#         gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         features = classifier.detectMultiScale(
#             gray_image, scaleFactor, minNeighbours)
#         coords = []

#         for x, y, w, h in features:
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             id, predict = clf.predict(gray_image[y: y + h, x: x + w])
#             userId = id
#             confidence = int((100 * (1 - predict / 300)))

#             if confidence > 77:
#                 coords.append((x, y, w, h))
#         return coords

#     def recognize(img, clf, classifier):
#         coords = draw_boundary(img, classifier, 1.1, 10, (0, 255, 0), clf)
#         return bool(coords)

#     video_capture = cv2.VideoCapture(0)
#     user_found = False
#     i = 0
#     while i < 50:  # Limit the number of frames to 50
#         i += 1
#         success, frame = video_capture.read()

#         if not success:
#             print("Failed to grab frame")
#             break
#         else:
#             user = recognize(frame, clf, classifier)
#             if user:
#                 user_found = True
#                 break  # Stop capturing more frames once a user is recognized

#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode(".jpg", frame)
#             if not ret:
#                 print("Failed to encode frame")
#                 break
#             frame = buffer.tobytes()

#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

#     video_capture.release()

#     if user_found:
#         print("User recognized, stopping stream")
#         return redirect(url_for("error"))  # Indicate that streaming should stop


# @app.route("/error", methods=["GET"])
# def error():
#     return "<h1>Face already exists. Redirected to error page!</h1>"


# @app.route("/check_face_video_feed")
# def check_face_video_feed():
#     frames_generator = gen_check_face_frames()

#     if frames_generator is None:
#         # Redirect to the error route if user is recognized
#         return redirect(url_for("error"))

#     return Response(frames_generator, mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/checkFaceExistence")
# def check_face_existence():
#     return render_template_string(
#         """
#         <!doctype html>
#         <html lang="en">
#         <head>
#             <meta charset="UTF-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#             <title>Video Streaming</title>
#         </head>
#         <body>
#             <h1>Mark Attendance</h1>
#             <img src="{{ url_for('check_face_video_feed') }}" width="640" height="480">

#            <script>
#                 const videoFeed = document.querySelector('img');
#                 let userRecognized = false;

#                 const checkFaceRecognition = setInterval(() => {
#                     if (!userRecognized) {
#                         fetch("{{ url_for('check_face_video_feed') }}")
#                             .then(response => response.text())
#                             .then(data => {
#                                 if (data.includes("RECOGNIZED")) {
#                                     userRecognized = true;  // Stop further requests
#                                     clearInterval(checkFaceRecognition);  // Stop polling
#                                     window.location.href = "{{ url_for('error') }}";  // Redirect to error page
#                                 }
#                             });
#                     }
#                 }, 1000);  // Check every second
#     </script>
#         </body>
#         </html>
#     """
#     )

# ================================================== Check Face Existence End ==============================================


# ======================================================== Add New Face Start ==============================================
def gen_frames(user_id):
    global image_counter
    image_counter = 0

    video_capture = cv2.VideoCapture(0)

    while image_counter < max_images:
        success, frame = video_capture.read()

        if not success:
            print("Failed to grab frame")  # Log instead of flashing
            break
        else:

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # If faces are detected, save the region of interest (the face)
            for (x, y, w, h) in faces:
                # Crop the face from the grayscale frame
                face_roi = gray_frame[y:y+h, x:x+w]
                # Resize face for uniformity
                face_resized = cv2.resize(face_roi, (200, 200))

                # Save the cropped face in grayscale to the 'data' folder
                image_path = os.path.join(data_folder, f"user_{user_id}_face_{
                                          image_counter:02d}.jpg")
                cv2.imwrite(image_path, face_resized)
                image_counter += 1

                # Encode the frame with the face rectangle in JPEG format (for display)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Encode the frame in JPEG format (for display)
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                print("Failed to encode frame")  # Log instead of flashing
                break
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    video_capture.release()


@app.route("/video_feed/<user_id>")
def video_feed(user_id):
    frames_generator = gen_frames(user_id)
    if frames_generator is None:
        return redirect(url_for("add_face", user_id=user_id))
    return Response(frames_generator, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/addFace/<user_id>")
def add_face(user_id):
    return render_template_string(
        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Video Streaming</title>
        </head>
        <body>
            <h1>Video Streaming</h1>
            <img src="{{ url_for('video_feed', user_id=user_id) }}" width="640" height="480">
            <script>
                const maxImages = {{ max_images }};
                
                const checkImageCount = setInterval(() => {
                    fetch('/image_count')
                        .then(response => response.json())
                        .then(data => {
                            if (data.count >= maxImages) {
                                clearInterval(checkImageCount);
                               if(confirm("Face Captured")){
                                window.location.href = "http://127.0.0.1:8000/admin/addFace";
                               }
                            }
                        }).catch(error => console.error('Error:', error));
                }, 1000); // Check every second
                
                document.addEventListener('keydown', function(event) {
                    if (event.keyCode === 13) {
                        window.location.reload();
                    }
                });
            </script>
        </body>
        </html>
    """,
        max_images=max_images,
        user_id=user_id,
    )

# ======================================================== Add New Face End =============================================


# ================================================== Mark Attendance Start ==============================================


def gen_attendance_frames():
    global user_recognized
    userId = None

    # Load face recognizer and the trained classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    try:
        clf.read("trainedClassifier.xml")  # Make sure this file exists
    except cv2.error as e:
        print(f"Error loading classifier: {e}")
        return

    # Load Haar Cascade for face detection
    classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, clf):
        nonlocal userId
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features = classifier.detectMultiScale(
            gray_image, scaleFactor, minNeighbours)
        coords = []

        for x, y, w, h in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, predict = clf.predict(gray_image[y: y + h, x: x + w])
            userId = id
            confidence = int((100 * (1 - predict / 300)))

            if confidence > 77:
                coords.append((x, y, w, h))
        return coords

    def recognize(img, clf, classifier):
        coords = draw_boundary(img, classifier, 1.1, 10, (0, 255, 0), clf)
        return bool(coords)

    video_capture = cv2.VideoCapture(0)
    i = 0
    while True:
        i += 1
        success, frame = video_capture.read()

        if not success:
            print("Failed to grab frame")
            break
        else:
            user = recognize(frame, clf, classifier)
            if cv2.waitKey(1) == 13 or i == 50:
                break

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                print("Failed to encode frame")
                break
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    video_capture.release()

    # if user:
    #     print("Login")
    #     print(userId)
    #     yield b"REDIRECT"
    # else:
    #     print("Logout")
    fail = 0
    if user:
        print(user)
        global user_recognized
        user_recognized = True  # Set the flag when the user is recognized
        # Log the recognition status
        print("User recognized:", user_recognized)
    else:
        fail += 1
        print(user)
        user_recognized = False
        print("User NOT recognized:", user_recognized)


@app.route("/attendance_video_feed")
def attendance_video_feed():
    frames_generator = gen_attendance_frames()
    if frames_generator is None:
        return redirect(url_for("markAttendance"))
    return Response(frames_generator, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/check_user_recognition")
def check_user_recognition():
    global user_recognized  # Make sure this flag is set correctly in gen_attendance_frames
    # Log recognition status
    print("Recognition status check:", user_recognized)
    return jsonify({"user_recognized": user_recognized})


@app.route("/markAttendance")
def mark_attendance():
    return render_template_string(
        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Video Streaming</title>
        </head>
        <body>
            <h1>Mark Attendance</h1>
            <img src="{{ url_for('attendance_video_feed') }}" id="videoFeed" width="640" height="480">
             <script>
            const videoFeed = document.getElementById('videoFeed');
            function checkUserRecognition() {
                const xhr = new XMLHttpRequest();
                xhr.open("GET", "/check_user_recognition", true);  // Poll endpoint

                xhr.onreadystatechange = function() {
                    if (xhr.readyState == 4 && xhr.status == 200) {
                        const response = JSON.parse(xhr.responseText);
                        console.log("Poll response:", response);  // Log the server response

                        if (response.user_recognized) {
                            console.log("User recognized, redirecting...");
                            window.location.href = 'https://github.com';  // Redirect to GitHub
                        } else if (response.fail_count > 10) {  // Check fail count
                            console.log("Fail count exceeded, redirecting to Google...");
                            window.location.href = 'https://www.google.com';  // Redirect to Google
                        }
                    }
                };

                xhr.send();
            }

            // Poll every 2 seconds
            setInterval(checkUserRecognition, 2000);
            </script>
        </body>
        </html>
    """
    )


# ================================================== Mark Attendance End ==================================================================


# ================================================== Train Face Start ==================================================================


def train_face():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if not os.path.exists(data_dir):
        return {"error": f"Directory {data_dir} does not exist."}

    path = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        try:
            img = Image.open(image).convert("L")
            imageNp = np.array(img, dtype=np.uint8)

            filename = os.path.split(image)[1]
            id = int(filename.split("_")[1])

            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            continue

    if len(faces) == 0 or len(ids) == 0:
        return {"error": "No faces or IDs found. Ensure there are images in the data directory."}

    print("IDs found:", ids)

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    classifier_path = os.path.join(
        os.path.dirname(__file__), "trainedClassifier.xml")
    clf.write(classifier_path)
    # print(f"Classifier trained and saved as {classifier_path}")
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    cv2.destroyAllWindows()

    return {"message": f"Training Face Data......"}


@app.route('/trainFace', methods=['GET'])
def run_train_face():
    result = train_face()

    if "error" in result:
        return jsonify(result)

    response = jsonify(result)

    response.headers["Refresh"] = "3; url=http://127.0.0.1:8000/admin/addFace"
    return response

# ============================================== Train Face End ===========================================================


@app.route("/image_count")
def image_count():
    global image_counter
    return jsonify(count=image_counter)


@app.route("/")
def index():
    return render_template_string(
        """<script>window.location.href = "/markAttendance";</script>"""
        # """<script>window.location.href = "/addFace/{id}";</script>"""
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
