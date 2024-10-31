import os
import numpy as np
import pandas as pd
import h5py
import random

# 파일 경로 설정
npy_dir = 'preprocessed/train_data'
csv_dir = 'preprocessed/train_labels'
output_dir = 'processed_clips1'

# 출력 폴더가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 설정값
clip_duration_ms = 400  # 음의 최대 클립 길이 400ms
buffer_duration_ms = 30  # 앞뒤 버퍼 30ms
total_clip_duration_ms = clip_duration_ms + 2 * buffer_duration_ms  # 총 460ms
sampling_rate = 44100
hop_length = 441  # 10ms hop length (441 samples)
clip_samples = int(clip_duration_ms * sampling_rate / 1000)  # 400ms에 해당하는 샘플 수
buffer_samples = int(buffer_duration_ms * sampling_rate / 1000)  # 30ms에 해당하는 샘플 수
clip_duration_frames = clip_samples // hop_length  # 400ms에 해당하는 프레임 수
total_frames = (clip_samples + 2 * buffer_samples) // hop_length  # 460ms에 해당하는 프레임 수

count = 1

# 악기 번호와 이름 매핑
instrument_mapping = {
    1: 'Grand_Piano', 41: 'Violin', 42: 'Viola', 43: 'Cello', 61: 'Horn',
    71: 'Bassoon', 72: 'Clarinet', 7: 'Harpsichord', 44: 'Contrabass',
    69: 'Oboe', 74: 'Flute'
}

def process_and_save_clip(npy_file, csv_file, index):
    # mel-spectrogram 데이터 로드
    spectrogram = np.load(npy_file)
    
    # CSV 데이터 로드
    df = pd.read_csv(csv_file)

    # 지정된 인덱스의 튜플 정보 가져오기
    row = df.iloc[index]
    start_frame = row['start_time'] // hop_length
    end_frame = row['end_time'] // hop_length

    # 1. 음이 400ms가 넘는 경우 자르고, 부족하면 패딩 추가 (400ms로 맞춤)
    original_clip = spectrogram[:, start_frame:end_frame].astype(np.float32)

    # 400ms에 맞게 클립을 자르거나 패딩 추가
    if original_clip.shape[1] > clip_duration_frames:
        clipped = original_clip[:, :clip_duration_frames]
    else:
        pad_width = clip_duration_frames - original_clip.shape[1]
        clipped = np.pad(original_clip, ((0, 0), (0, pad_width)), mode='constant')

    # 2. 400ms 클립에 30ms 버퍼 추가 (460ms로 확장)
    start_frame_buffered = max(0, start_frame - buffer_samples // hop_length)
    end_frame_buffered = start_frame_buffered + total_frames

    # 버퍼를 적용한 클립 생성
    clip_buffered = spectrogram[:, start_frame_buffered:end_frame_buffered].astype(np.float32)

    # 클립 길이가 부족하면 패딩 추가
    if clip_buffered.shape[1] < total_frames:
        pad_width = total_frames - clip_buffered.shape[1]
        clip_buffered = np.pad(clip_buffered, ((0, 0), (0, pad_width)), mode='constant')

    # 3. 저장할 시작 시간과 종료 시간 갱신
    new_start_time = row['start_time'] - buffer_duration_ms  # 30ms 앞당김
    new_end_time = new_start_time + total_clip_duration_ms  # 시작 후 460ms
    
    # 악기 및 음 정보 저장 (스칼라 값을 배열로 변환)
    instrument = row['instrument']
    instrument_name = instrument_mapping.get(instrument, 'Unknown')  # 악기 이름 변환
    note = row['note']

    # 악기별 폴더 생성
    instrument_dir = os.path.join(output_dir, instrument_name)
    if not os.path.exists(instrument_dir):
        os.makedirs(instrument_dir)

    # 클립 및 정보를 해당 악기 폴더에 저장
    output_file = os.path.join(instrument_dir, f"{os.path.splitext(os.path.basename(npy_file))[0]}_{index}.h5")
    
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('clip', data=clip_buffered, compression='gzip', compression_opts=9)
        hf.create_dataset('instrument', data=np.array([instrument]))  # 스칼라 값을 배열로 변환하여 저장
        hf.create_dataset('note', data=np.array([note]))  # 스칼라 값을 배열로 변환하여 저장
        hf.create_dataset('start_time', data=np.array([new_start_time]))  # 스칼라 값을 배열로 변환하여 저장
        hf.create_dataset('end_time', data=np.array([new_end_time]))  # 스칼라 값을 배열로 변환하여 저장

    print(f"Saved clip {output_file} with instrument {instrument_name} and note {note} "
          f"new start {new_start_time} end {new_end_time}")

# npy와 csv 파일의 목록을 가져오기
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

# npy와 csv 파일 이름이 일치하는 파일만 처리
for npy_file, csv_file in zip(npy_files, csv_files):
    npy_path = os.path.join(npy_dir, npy_file)
    csv_path = os.path.join(csv_dir, csv_file)

    if os.path.splitext(npy_file)[0] == os.path.splitext(csv_file)[0]:
        # CSV 파일 로드
        df = pd.read_csv(csv_path)

        # 전체 행 중에서 랜덤하게 100개 선택
        random_indices = random.sample(range(len(df)), min(100, len(df)))

        # 선택된 행에 대해 클립 처리 및 저장
        for i, index in enumerate(random_indices):
            print(f"{npy_file}: {i+1}번째 클립 처리: {index}번째 행")
            process_and_save_clip(npy_path, csv_path, index)

        print(f"{count}번째 파일에 대해 랜덤한 100개의 클립 처리가 완료되었습니다.")
        count += 1