import os
import cv2
import dlib
import face_recognition
import streamlit as st
import requests

# Developer Names
st.sidebar.title('Developers:')
st.sidebar.title('Information Technology NITK')
st.sidebar.write("Shamitha Naik - 211IT086")
st.sidebar.write("Bhavitha Naramamidi - 211IT044")
st.sidebar.write("Pari Poptani - 211IT045")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_watchlist_encodings(watchlist_directory):
    watchlist_encodings = []
    for filename in os.listdir(watchlist_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(watchlist_directory, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            watchlist_encodings.append(encoding)
    return watchlist_encodings

def detect_face_similarity(input_image, watchlist_directory, threshold=0.6):
    with open("temp_image.jpg", "wb") as f:
        f.write(input_image.read())
    watchlist_encodings = load_watchlist_encodings(watchlist_directory)
    image = cv2.imread("temp_image.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        face_encoding = face_recognition.face_encodings(image, [(y1, x2, y2, x1)])[0]

        for idx, watchlist_encoding in enumerate(watchlist_encodings):
            results = face_recognition.compare_faces([watchlist_encoding], face_encoding, tolerance=threshold)
            if results[0] == True:
                st.write(f"Face in input image matches with watchlist image {os.listdir(watchlist_directory)[idx]}")

        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        output_image_path = "temp_image_output.jpg"
        cv2.imwrite(output_image_path, image)
        st.image(output_image_path, caption='Output image with facial landmarks', use_column_width=True)

        return output_image_path  

def compare_with_watchlist(input_image_path, watchlist_directory):
    known_image = face_recognition.load_image_file(input_image_path)
    known_image_encoding = face_recognition.face_encodings(known_image)[0]

    max_similarity = 0
    most_similar_image_path = None

    for filename in os.listdir(watchlist_directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img2_path = os.path.join(watchlist_directory, filename)

            unknown_image = face_recognition.load_image_file(img2_path)
            face_encodings = face_recognition.face_encodings(unknown_image)

            if len(face_encodings) > 0:
                face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]
                similarity = (1 - face_distance) * 100

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_image_path = img2_path

    if most_similar_image_path:
        st.write(f"Most similar image: {most_similar_image_path}")
        st.write(f"Similarity between input and most similar image : {round(max_similarity, 2)}%")
    else:
        st.write("No similar image found in the watchlist directory.")

def detect_similarity_with_watchlist(input_image_path, watchlist_directory):
    watchlist_encodings = load_watchlist_encodings(watchlist_directory)
    input_image = face_recognition.load_image_file(input_image_path)
    input_encoding = face_recognition.face_encodings(input_image)[0]

    similarity_scores = []
    for watchlist_encoding in watchlist_encodings:
        similarity_score = face_recognition.face_distance([watchlist_encoding], input_encoding)[0]
        similarity_scores.append(1 - similarity_score)

    return similarity_scores

def main():
    st.title("Face Similarity Detection")
    input_image_path = st.file_uploader("Upload an image", type=["jpg", "png"])
    if input_image_path:
        st.image(input_image_path, caption='Uploaded Image', use_column_width=True)

    watchlist_choice = st.radio("Which watchlist do you want to use for similarity comparison?",
                                ('Normal watchlist', 'Expressions watchlist'))

    if watchlist_choice == 'Normal watchlist':
        watchlist_directory = "Watchlist"
        if st.button('Detect Similarity'):
            output_image_path = detect_face_similarity(input_image_path, watchlist_directory)
            compare_with_watchlist(output_image_path, watchlist_directory)
    
    else:
        watchlist_directory = "M_watchlist"
        if input_image_path is not None:
            if st.button('Detect Similarity'):
                similarity_scores = detect_similarity_with_watchlist(input_image_path, watchlist_directory)
                st.write("Similarity scores with watchlist images:")
                for idx, score in enumerate(similarity_scores):
                    st.write(f"Watchlist image {idx + 1}: {score}")
   
    st.markdown(
    """
    <style>
    .reportview-container {
        background: url("backgorund.jpg")
    }
   .sidebar .sidebar-content {
        background: url("backgorund.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    main()
