import pandas as pd 
import numpy as np 
import os 
import glob 
import cv2 
import json
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, AudioFileClip
import zipfile

def make_frame_dataset(json_path):  # return : (start_frame, end_frame, 말한 문장)
    with open(json_path, 'rb') as f: 
        data = json.load(f)

    fps = 25

    frame_datas = []
    sentence_infos = data[0].get('Sentence_info')
    for voice in sentence_infos:
        start_frame = int(voice['start_time']*fps)
        end_frame = int(voice['end_time']*fps)
        sentence_text = voice['sentence_text']
        frame_datas.append((start_frame, end_frame, sentence_text))
    return frame_datas

def fps_29_to_25(video_path, fps=25): 
    video = VideoFileClip(video_path)
    video = video.set_fps(fps)
    # 변경된 fps로 비디오 파일 저장
    video.write_videofile(video_path.split('.')[0]+'_25fps.mp4', fps=fps)

def cut_video_audio(video_path, start, end, video_save_path, audio_save_path):
    cap = cv2.VideoCapture(video_path)  # 영상용
    start -=15
    end += 15
    
    # 프레임 속성 알아내기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)  

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_save_path, fourcc, fps, (1080, 720))         # 여기서 512 로 맞춰줬음

    # 지정된 프레임 추출 및 저장
    for i in tqdm(range(end-start)):   
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (int(frame_width//1.5), int(frame_height//1.5)))          # frame_width//2.109도 512에 맞춘거라 바꾸려면 바꿔야함 
            crop_frame = resized_frame[:,100:1180]
            out.write(crop_frame)
        else:
            break

    # 비디오 파일 닫기
    cap.release()
    out.release()

    # 비디오 파일 열기
    video = VideoFileClip(video_path)       # 비디오용

    # 프레임 범위에 해당하는 프레임 추출
    frames = video.subclip(start/video.fps, end/video.fps)
    # 추출한 프레임에 해당하는 오디오 데이터 추출
    sub_audio = frames.audio
    # 추출한 프레임과 오디오를 함께 저장
    sub_audio.write_audiofile(audio_save_path)

def combine_vid_aud(vid_path, aud_path, out_path):
    # 비디오와 오디오 파일 열기
    video_clip = VideoFileClip(vid_path)
    audio_clip = AudioFileClip(aud_path)

    # 오디오와 비디오를 합치기
    final_clip = video_clip.set_audio(audio_clip)

    # 최종 파일로 내보내기
    final_clip.write_videofile(out_path)

