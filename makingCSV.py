#
# [ Preprocessing Data ]
# 입력으로 비디오(캠)를 받아서 움직이는 물체에 대해서 OpenCV를 통해서 labeling(boxing)을 한 후
# 그 안에서 mediapipe 를 통해 사람이라면 skeleton 을 찾아낸다. "행동" 이라는 걸 정의하기 위해
# 현재는 3프레임 동안 skeleton 이 유지되면 3프레임의 pose 의 좌표에서 0프레임의 pose 의 좌표를
# 빼는 방식으로 행동 데이터를 저장한다.
#
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import mediapipe as mp
import csv
import os

def body_check(vi_list):
    threshold = 0.5
    try:
        if vi_list[16] > threshold and vi_list[14] > threshold and vi_list[12] > threshold \
                and vi_list[11] > threshold and vi_list[13] > threshold and vi_list[15] > threshold \
                and vi_list[24] > threshold and vi_list[26] > threshold and vi_list[28] > threshold \
                and vi_list[23] > threshold and vi_list[25] > threshold and vi_list[27] > threshold:
            return True
    except:
        return False
    return False


# 반복 횟수 (비디오 개수)
Repeat = 1
File_index = 0

# 행동들
action_list = ["Drunk", "Fight", "Rob"]

# 첫번째줄 표시 플래그
first = True

# 로지스틱 회귀용
for j in range(len(action_list)):
    for i in range(Repeat):
        print("Now index : " + str(i))

        # CSV
        f = open("Data/result.csv", 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        if first: # 처음에만 표시해야 함
            wr.writerow(["Frame", "Index",
                         "RElbow_x", "RElbow_y", "RElbow_z",
                         "RShoulder_x", "RShoulder_y", "RShoulder_z",
                         "RWrist_x", "RWrist_y", "RWrist_z",
                         "RHip_x", "RHip_y", "RHip_z",
                         "RKnee_x", "RKnee_y", "RKnee_z",
                         "RAnkle_x", "RAnkle_y", "RAnkle_z",
                         "LElbow_x", "LElbow_y", "LElbow_z",
                         "LShoulder_x", "LShoulder_y", "LShoulder_z",
                         "LWrist_x", "LWrist_y", "LWrist_z",
                         "LHip_x", "LHip_y", "LHip_z",
                         "LKnee_x", "LKnee_y", "LKnee_z",
                         "LAnkle_x", "LAnkle_y", "LAnkle_z",
                         "Action"
                         ])
        first = False

        # MediaPipe
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        mpDraw = mp.solutions.drawing_utils

        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # 카메라/비디오 캡쳐

        # 로지스틱 회귀용
        cap = cv2.VideoCapture("Video/" + action_list[j] + "/" + str(i) + ".mp4")
        # cap = cv2.VideoCapture("test_cafe.mp4")
        # cap = cv2.VideoCapture(0)

        # 프레임 별 변수
        prev_index = 0
        frame = 0
        skeleton_index = [0 for i in range(5)]
        prev_pose = [mpPose.Pose for i in range(5)]
        prev_frame = [0 for i in range(5)]

        # 프레임 시작
        while cap.isOpened():
            # 프레임 증가
            frame += 1

            # 프레임 당 Skeleton window 삭제
            for i in range(prev_index):
                cv2.destroyWindow("Skeleton" + str(i))

            # load the image and resize it to (1) reduce detection time
            # and (2) improve detection accuracy
            success, image = cap.read()

            if not success:
                print("Video finished")
                break

            # 이미지 리사이즈 ( FHD 이상일 경우 )
            if image.shape[0] > 1080 and image.shape[1] > 1920:
                print("resized")
                image = cv2.resize(image, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)

            height, width, color = image.shape
            orig = image.copy()

            # detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)
            # draw the original bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            # draw the final bounding boxes

            # TODO 한 프레임안에 들어있는 pick 의 이미지들 전부 윈도우로 띄우기
            index = 0

            for (xA, yA, xB, yB) in pick:
                resize_size = 20
                irs = 150  # 무시하는 정사각형 사이즈 ( 하이퍼 파라미터 )

                # 박스 최대사이즈 예외처리
                resize_xA = (xA - resize_size) if xA - resize_size > 0 else 0
                resize_yA = (yA - resize_size) if yA - resize_size > 0 else 0
                resize_xB = (xB + resize_size) if xB + resize_size < width else width
                resize_yB = (yB + resize_size) if yB + resize_size < height else height

                # 일정 크기 이하의 사각형 무시
                if resize_xB - resize_xA < irs or resize_yB - resize_yA < irs:
                    # print("ignored")
                    continue

                cv2.rectangle(image, (resize_xA, resize_yA), (resize_xB, resize_yB), (0, 255, 0), 2)
                new_img = image[resize_yA:resize_yB, resize_xA:resize_xB].copy()

                # TODO MediaPipe Processing

                imgRGB = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
                results = pose.process(imgRGB)
                vi = []
                allPass = True

                # TODO 3프레임동안 skeleton 이 유지되면 그것을 하나의 행동으로 보고 각 관절의 이동벡터를 저장한다
                if results.pose_landmarks:
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w, c = new_img.shape
                        vi.append(lm.visibility)
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(new_img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
                    #     allPass = body_check(vi)
                    # if allPass:
                    # print(vi)
                    mpDraw.draw_landmarks(new_img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    cv2.imshow("Skeleton" + str(index), new_img)
                    cv2.moveWindow("Skeleton" + str(index), index * 250, 100)

                    # 스켈레톤 인덱스 0에서 pose 저장
                    # TODO 사람이 여러명 나올 때 문제 발생 ( 사람마다의  prev_pose 가 저장되어야 함 )
                    if skeleton_index[index] == 0:
                        prev_pose[index] = results.pose_landmarks
                        prev_frame[index] = frame

                    # 스켈레톤 인덱스가 3까지 유지되면 0으로 초기화 하고 현재 프레임의 pose 에서
                    # 스켈레톤 인덱스 0일 때의 pose 를 빼서 저장한다 ( 이동 벡터 생성 )
                    if skeleton_index[index] == 3:
                        skeleton_index[index] = 0
                        wr.writerow([str(prev_frame[index]) + "-" + str(frame), index,
                                     round(prev_pose[index].landmark[14].x - results.pose_landmarks.landmark[14].x, 2),
                                     round(prev_pose[index].landmark[14].y - results.pose_landmarks.landmark[14].y, 2),
                                     round(prev_pose[index].landmark[14].z - results.pose_landmarks.landmark[14].z, 2),
                                     round(prev_pose[index].landmark[12].x - results.pose_landmarks.landmark[12].x, 2),
                                     round(prev_pose[index].landmark[12].y - results.pose_landmarks.landmark[12].y, 2),
                                     round(prev_pose[index].landmark[12].z - results.pose_landmarks.landmark[12].z, 2),
                                     round(prev_pose[index].landmark[16].x - results.pose_landmarks.landmark[16].x, 2),
                                     round(prev_pose[index].landmark[16].y - results.pose_landmarks.landmark[16].y, 2),
                                     round(prev_pose[index].landmark[16].z - results.pose_landmarks.landmark[16].z, 2),
                                     round(prev_pose[index].landmark[24].x - results.pose_landmarks.landmark[24].x, 2),
                                     round(prev_pose[index].landmark[24].y - results.pose_landmarks.landmark[24].y, 2),
                                     round(prev_pose[index].landmark[24].z - results.pose_landmarks.landmark[24].z, 2),
                                     round(prev_pose[index].landmark[26].x - results.pose_landmarks.landmark[26].x, 2),
                                     round(prev_pose[index].landmark[26].y - results.pose_landmarks.landmark[26].y, 2),
                                     round(prev_pose[index].landmark[26].z - results.pose_landmarks.landmark[26].z, 2),
                                     round(prev_pose[index].landmark[28].x - results.pose_landmarks.landmark[28].x, 2),
                                     round(prev_pose[index].landmark[28].y - results.pose_landmarks.landmark[28].y, 2),
                                     round(prev_pose[index].landmark[28].z - results.pose_landmarks.landmark[28].z, 2),
                                     round(prev_pose[index].landmark[13].x - results.pose_landmarks.landmark[13].x, 2),
                                     round(prev_pose[index].landmark[13].y - results.pose_landmarks.landmark[13].y, 2),
                                     round(prev_pose[index].landmark[13].z - results.pose_landmarks.landmark[13].z, 2),
                                     round(prev_pose[index].landmark[11].x - results.pose_landmarks.landmark[11].x, 2),
                                     round(prev_pose[index].landmark[11].y - results.pose_landmarks.landmark[11].y, 2),
                                     round(prev_pose[index].landmark[11].z - results.pose_landmarks.landmark[11].z, 2),
                                     round(prev_pose[index].landmark[15].x - results.pose_landmarks.landmark[15].x, 2),
                                     round(prev_pose[index].landmark[15].y - results.pose_landmarks.landmark[15].y, 2),
                                     round(prev_pose[index].landmark[15].z - results.pose_landmarks.landmark[15].z, 2),
                                     round(prev_pose[index].landmark[23].x - results.pose_landmarks.landmark[23].x, 2),
                                     round(prev_pose[index].landmark[23].y - results.pose_landmarks.landmark[23].y, 2),
                                     round(prev_pose[index].landmark[23].z - results.pose_landmarks.landmark[23].z, 2),
                                     round(prev_pose[index].landmark[25].x - results.pose_landmarks.landmark[25].x, 2),
                                     round(prev_pose[index].landmark[25].y - results.pose_landmarks.landmark[25].y, 2),
                                     round(prev_pose[index].landmark[25].z - results.pose_landmarks.landmark[25].z, 2),
                                     round(prev_pose[index].landmark[27].x - results.pose_landmarks.landmark[27].x, 2),
                                     round(prev_pose[index].landmark[27].y - results.pose_landmarks.landmark[27].y, 2),
                                     round(prev_pose[index].landmark[27].z - results.pose_landmarks.landmark[27].z, 2),
                                     j
                                     ])
                    else:
                        # 스켈레톤 인덱스 증가
                        skeleton_index[index] += 1
                else:
                    # print(index)
                    skeleton_index[index] = 0

                index += 1
            # show some information on the number of bounding boxes

            # show the output images
            # cv2.imshow("Before NMS", orig)
            cv2.imshow("After NMS", image)

            prev_index = index

            if cv2.waitKey(1) & 0xFF == 27:
                break

File_index += 1
