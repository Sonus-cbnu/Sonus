import os
import librosa
import numpy as np

# 입력 및 출력 경로 설정
input_folder = 'musicnet/train_data/'
output_folder = 'preprocessed'
os.makedirs(output_folder, exist_ok=True)

# 샘플링 주파수 및 Mel 스펙트로그램 설정
sampling_rate = 44100
n_fft = 4096  # DFT 변환을 위한 샘플 수
hop_length = int(0.01 * sampling_rate)  # 10ms 단위로 윈도우 이동
n_mels = 256  # Mel 필터의 개수

# train_data 폴더에서 320개의 .wav 파일 전처리
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')][:320]

for audio_file in audio_files:
    # 파일 경로 설정 및 로드
    audio_path = os.path.join(input_folder, audio_file)
    y, sr = librosa.load(audio_path, sr=sampling_rate)
    
    # Mel 스펙트로그램 계산
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 저장할 파일 경로 (.npy)
    output_path = os.path.join(output_folder, f'{os.path.splitext(audio_file)[0]}.npy')
    
    # Mel 스펙트로그램을 numpy 배열로 저장
    np.save(output_path, mel_spectrogram_db)
    print(f"Processed and saved: {output_path}")