#Morphed faced detection using watchlist 
shamitha

paper:

Fei Peng, Le Qin, Min Long, “Face morphing attack detection and attacker identification based on a watchlist”
Signal Processing: Image Communication, Volume 107, 2022,
116748,
ISSN 0923-5965,
https://doi.org/10.1016/j.image.2022.116748


Construct a watchlist containing bona fide facial images as biometric references.

Extract facial features from both the suspect image and the biometric references in the watchlist.

Calculate similarity scores between the suspect image and each entry in the watchlist using the extracted facial features.

Employ a threshold on similarity scores to classify the image as morphed or genuine.

If morphed, identify the morphing attacker. And evaluate the model.


Face Recognition and Comparison
This file contains Python scripts for face recognition and comparison 

run using
pip install streamlit

streamlit run <path>

eg: streamlit run "D:/appy.py"

Usage:
streamlit application called appy is used to run the project in the local host.
it has option to run the model for either bonafide or morphed images directory. Based on selection it runs either of the 2 
watchlist and outputs the facial landmarks image of the input and the similarity score and other result in the form of a web page.

it is named as appy.py

the below given files were combined into one single streamlit app.
link to the app:
https://shamithanaik-mad-appy-2eb8uk.streamlit.app/

Face Recognition
face_recognition.py: This script performs face recognition on input images.

Place your input images in a directory.
Run the script and provide the path to the input image directory when prompted.
Follow the on-screen instructions to see face recognition results.

Face Comparison
comparison.py: This script compares faces in an input image with a watchlist of known faces.

The script will compare faces and display the most similar image from the watchlist.
 directory for face detection and landmark prediction.
Adjust threshold values and other parameters in the scripts as needed

face similarity
similarity: this finds the similarity between input and the ones in morphed watchlist with different types of morphes to output the 
similarity scores.
