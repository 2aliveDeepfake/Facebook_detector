from inference_video_face import load_face_model, detect_face
from load_model import load_models, detect_fake
import os

# VIDEO_PATH = "./test_videos/test.mp4"

# 모델을 다 불러와서 session 에 올려놓고 시작
# face 모델 불러오기
f_sess, f_detection_graph, f_category_index = load_face_model()
# 가짜 특징 모델 다 불러옴
sess, detection_graph, category_index = load_models()

# 처리할 비디오 경로
# folder_path = "./test_videos/"
folder_path = "G:\\deepfake-detection-challenge\\test_videos\\"
folder_list = os.listdir(folder_path)

# num = input("영상에서 몇 프레임마다 detect 할지 입력 : ")
# count = input("몇번째 프레임을 확인할지 입력 : ")
# 모든 프레임에서 검출
num = 10
count = 0
for video_item in folder_list:
    # 영상을 몇 프레임마다 추출할 건지 입력
    number = int(num)
    # 몇번째 프레임을 추출할지 입력
    count_num = int(count)

    PATH_TO_VIDEO = folder_path+video_item

    # video에서 얼굴 찾는 모델 불러와서 얼굴 넘기기
    face_list, load_face_model_time = \
        detect_face(PATH_TO_VIDEO, number, count_num, f_sess, f_detection_graph, f_category_index)
    load_model_time = \
        detect_fake(face_list, PATH_TO_VIDEO, number, count_num, sess, detection_graph, category_index)
