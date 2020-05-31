# Person_Age_Gender_Prediction

## Assumption:-

	“No Persons are overlapping on Each other during the video feed”
			
## The centroid tracking algorithm
	1.	Accepting bounding box coordinates for each person in every frame (presumably by some person detector).
	2.	Computing the Euclidean distance between the centroids of the input bounding boxes and the centroids of existing person 
	3.	Updating the tracked person centroids to their new centroid locations based on the new centroid with the smallest Euclidean distance.

## Gender Detection Algorithm

First, the photo is taken from the webcam stream live by the cv2 module.
Second, we turn the image to grayscale and use the cv2 module's CascadeClassifier class to detect faces in the image
The variable faces return by the detectMultiScale method is a list of detected face coordinates [x, y, w, h]. After known the faces' coordinates, we need to crop those faces before feeding to the neural network model. We add the 40% margin to the face area so that the full head is included.
Then we are ready to feed those cropped faces to the model, it's as simple as calling the predict method. Last but not least we draw the result and render the image. The gender prediction is a binary classification task. The model outputs value between 0~1, where the higher the value, the more confidence the model think the face is a male.
Each image before feeding into the model we did the same pre-processing step shown above, detect the face and add margin.
The feature extraction part of the neural network uses the WideResNet architecture, short for Wide Residual Networks. It leverages the power of Convolutional Neural Networks (or ConvNets for short) to learn the features of the face. From less abstract features like edges and corners to more abstract features like eyes and mouth.
