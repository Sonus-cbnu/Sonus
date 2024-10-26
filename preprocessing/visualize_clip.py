import numpy as np
import matplotlib.pyplot as plt
import os
import h5py  # HDF5 파일을 다루기 위한 라이브러리

hop_length = 441

# 시각화할 .h5 파일 경로 (첫 번째 파일만 시각화)
output_dir = 'processed_clips'
h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]

if len(h5_files) > 0:
    h5_file_path = os.path.join(output_dir, h5_files[0])  # 첫 번째 .h5 파일을 불러옴
    
    # HDF5 파일에서 데이터 로드
    with h5py.File(h5_file_path, 'r') as hf:
        clip = hf['clip'][:]
        instrument = hf['instrument'][()]
        note = hf['note'][()]
        start_time = hf['start_time'][()]
        end_time = hf['end_time'][()]
    
    # 스펙트로그램 클립의 실제 길이 계산
    duration_frames = (end_time - start_time) // hop_length

    # 실제 클립 구간에 맞게 자른 후 시각화
    plt.figure(figsize=(10, 6))
    plt.imshow(clip[:, :duration_frames], aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Amplitude')
    plt.title(f'Instrument: {instrument}, Note: {note}, Start: {start_time}, End: {end_time}')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.show()

else:
    print("시각화할 .h5 파일이 없습니다.")