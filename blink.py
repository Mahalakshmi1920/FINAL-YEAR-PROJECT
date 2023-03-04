import cv2
import dlib
from scipy.spatial import distance

# Load video file
cap = cv2.VideoCapture('video.mp4')

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize blink counter and frame counter
blink_counter = 0
frame_counter = 0

# Set threshold for blink detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 20

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if we have reached the end of the video
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over the faces
    for face in faces:
        # Determine the facial landmarks for the face
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        # Extract the left and right eye landmarks
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        # Compute the eye aspect ratios for the left and right eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Compute the average eye aspect ratio
        ear = (left_ear + right_ear) / 2.0

        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                frame_counter = 0
            blink_counter = 0

    # Increment the frame counter
    frame_counter += 1

    # Pause the video if there is no eye blink for more than 30 frames
    if frame_counter > 30:
        cv2.imshow('Video', frame)
        cv2.waitKey(0)
        frame_counter = 0

    # Show the frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()