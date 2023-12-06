
import pyautogui as ag

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#print(f"mediapipe version: {mp.__version__}")
#print(f"openCV version: {cv2.__version__}")

"""

This program tracks the movement of the right eye. The iris moves relative to four points: the left and right corners of the eye and points on the upper and lower eye lid.
The location of the iris relative to these four points directs the mouse cursor.

TO DO:

Movement along the y axis is wonky, this is because the iris doesn't move much relative to the upper and lower eyelids
however, the eyelids squint as the eye focuses downward and opens as the eye looks up, this relationship might help to correct
movement along the y axis.

Also, adjust x axis for when head tilts left to right.

"""

#screen size for cursor conversion
screen_x, screen_y = ag.size()

#camera color adjustment for better clarity
alpha = 1.5 # Contrast control
beta = 10 # Brightness control

#annotation marker attributes 
color = (0, 255, 0)
markerType = cv2.MARKER_CROSS
markerSize = 15
thickness = 1

#camera input settings
cap = cv2.VideoCapture(0)
cap.set(10,100)

#face mesh model
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def resize(image):
  width = 700
  height = 300
  dim = (width, height)
  
  resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return resized

  

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  annotated_image = cv2.convertScaleAbs(annotated_image, alpha=alpha, beta=beta)
  
  x_min, x_max = 0, 0
  y_min, y_max = 0, 0

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]


    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    #locate right eye landmarks, 
    r_eye_right  = face_landmarks[33]
    r_eye_left   = face_landmarks[133]
    r_eye_upper  = face_landmarks[159]
    r_eye_lower  = face_landmarks[145]
    r_iris       = face_landmarks[468]

    #collect image dimensions
    img_y, img_x, _ = annotated_image.shape

    #determine iris location
    x, y = r_iris.x, r_iris.y

    #determine eye boundaries box
    x_min, x_max = r_eye_right.x, r_eye_left.x
    y_min, y_max = r_eye_upper.y, r_eye_lower.y


    ### control mouse code

    #x position
    x_range = x_max-x_min
    x_midpoint = x_range/2
    x_pos = x-x_min
    #makes eye movements larger to affect cursor
    x_accel = (x_pos-x_midpoint)*2 
    x_adj = 1-(x_pos+x_accel)/x_range

    #y position [[TO DO]]
    y_range = y_max-y_min
    y_midpoint = y_range/2
    y_pos = y-y_min
    y_adj = (y_pos)/y_range
    
    ag.moveTo(screen_x*x_adj, screen_y*y_adj)

    ### camera output code
    
    #mark the iris center
    cv2.drawMarker(annotated_image, (int(x*img_x), int(y*img_y)), color, markerType, markerSize, thickness)

    """
    #for generating an outline of the iris

    solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_iris_connections_style())
    """


  #annotated_image = annotated_image[y_min:y_max, x_min:x_max]
  #annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY)
  #annotated_image = resize(image=annotated_image)
    
  return annotated_image
    


def video_analysis():
  while True:
      success, frame = cap.read()
      if not success:
          print("Empty image, check camera.")
          continue
        
      frame_cp = frame.copy()
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_cp)
      detection_result = detector.detect(mp_image)

      try:
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("annotated_vid", annotated_image)
        
      except:
        print("No frame data, failed to analyze.")
        
      if cv2.waitKey(1) & 0xFF == ord('q'): 
          break

  cap.release()



def single_image_analysis():
  ### for generating landmarks mesh over single image.


  # STEP 3: Load the input image.
  image = mp.Image.create_from_file("business-person.png"); #print(type(image))

  # STEP 4: Detect face landmarks from the input image.
  detection_result = detector.detect(image)

  # STEP 5: Process the detection result. In this case, visualize it.
  annotated_image = cv2.cvtColor(draw_landmarks_on_image(image.numpy_view(), detection_result), cv2.COLOR_RGB2GRAY)


  cv2.imshow("", annotated_image)

  #print(detection_result.facial_transformation_matrixes)



### run script

#single_image_analysis()
video_analysis()

