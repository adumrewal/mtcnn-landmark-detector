# Copyright (c) 2023 Amol Dumrewal

# Sample usage:
#     import cv2
#     from mtcnn_model import MTCNNModel

#     # Load the input image
#     image = cv2.imread('cropped_face_image.jpg')

#     # Initialize the MTCNN model with the given model path
#     model_path = 'landmark_detector_weights.npy'
#     face_landmark_model = FaceLandmarkModel(model_path)

#     # Check if the input image contains a face and detect the landmarks
#     score, landmarks = model_path.face_landmarks(image)

#     if score > 0.95:
#         print('The input image contains a face with landmarks: ', landmarks)
#     else:
#         print('The input image does not contain a clear face.')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import detector as detector_network

class FaceLandmarkModel:
    def __init__(self, model_path) -> None:
        # Initialize the MTCNN model with the given model path
        self.detector = self._get_detector_network(model_path)
    
    def face_landmarks(self, image):
        '''
        Detect face landmarks in the given image.
        
        Parameters
        ----------
        image : numpy.ndarray
            The input cropped face image with shape (h, w, 3) in BGR format (OpenCV default)
            
        Returns
        -------
        score: float
            The confidence score of the face landmark detection in the input image.
        
        face_landmarks : numpy.ndarray
            The face landmarks detected in the input image.
            Shape: (5, 2)
            Value: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]] # fractional float values in [0, 1]
            Order: [left_eye, right_eye, nose, left_mouth_centre, right_mouth_centre]
        '''
        score, face_landmarks = self._get_face_landmarks(image)
        return score, face_landmarks
    
    def _get_detector_network(self, model_path):
        # Create a new TensorFlow graph and session
        with tf.Graph().as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=None, log_device_placement=False))
            with sess.as_default():
                # Load the ONet model from the given model path
                detector = detector_network.create_detector_network(sess, model_path)
        return detector
    
    def _get_face_landmarks(self, image):
        # Detect face landmarks using the ONet model
        score, face_landmarks = detector_network.detect_landmarks(self.detector, image)
        return score, face_landmarks
