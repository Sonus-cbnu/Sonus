"""
이 스크립트는 멀티프로세싱을 활용하지 않고도 음원 클립 데이터를 생성합니다.
각 npy 및 csv 파일 쌍에 대해 랜덤한 3000개의 클립을 추출하고,
각 악기별로 하위 폴더를 생성하여 클립을 저장합니다.
각 하위 폴더에는 최대 10,000개의 파일이 저장되며, 진행 상황을 시각화하여 모니터링할 수 있습니다.
로그는 파일에만 기록됩니다.
"""

import os
import numpy as np
import pandas as pd
import h5py
import random
import logging
from tqdm import tqdm  # 진행 상황 시각화를 위해 tqdm 추가
from logging.handlers import RotatingFileHandler


# 로그 기능 설정
def setup_logging():
    """로그 설정을 초기화합니다."""
    log_dir = "./log"
    ensure_dir(log_dir)  # 로그 디렉토리 생성
    log_file = os.path.join(log_dir, "clip_generator_log.txt")
    # 로그 로테이션 설정 (10MB, 최대 5개 백업)
    handler = RotatingFileHandler(
        log_file, maxBytes=10**7, backupCount=5, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler],
    )


# 파일 경로 설정
NPY_DIR = "./data/preprocessed_npy/test_data"
CSV_DIR = "./data/preprocessed_npy/test_labels"
OUTPUT_DIR = "./data/processed_test_clips"

# 설정값
CLIP_DURATION_MS = 400  # 음의 최대 클립 길이 400ms
BUFFER_DURATION_MS = 30  # 앞뒤 버퍼 30ms
TOTAL_CLIP_DURATION_MS = CLIP_DURATION_MS + 2 * BUFFER_DURATION_MS  # 총 460ms
SAMPLING_RATE = 44100
HOP_LENGTH = 441  # 10ms hop length (441 samples)
CLIP_SAMPLES = int(CLIP_DURATION_MS * SAMPLING_RATE / 1000)  # 400ms에 해당하는 샘플 수
BUFFER_SAMPLES = int(
    BUFFER_DURATION_MS * SAMPLING_RATE / 1000
)  # 30ms에 해당하는 샘플 수
CLIP_DURATION_FRAMES = CLIP_SAMPLES // HOP_LENGTH  # 400ms에 해당하는 프레임 수
TOTAL_FRAMES = (
    CLIP_SAMPLES + 2 * BUFFER_SAMPLES
) // HOP_LENGTH  # 460ms에 해당하는 프레임 수

# 악기 번호와 이름 매핑
INSTRUMENT_MAPPING = {
    1: "Grand_Piano",
    41: "Violin",
    42: "Viola",
    43: "Cello",
    61: "Horn",
    71: "Bassoon",
    72: "Clarinet",
    7: "Harpsichord",
    44: "Contrabass",
    69: "Oboe",
    74: "Flute",
}


def ensure_dir(path):
    """지정된 경로의 디렉토리가 없으면 생성합니다."""
    if not os.path.exists(path):
        os.makedirs(path)


def process_and_save_clip(npy_file, csv_file, index, instrument_file_counts):
    """
    지정된 인덱스의 클립을 처리하고 저장합니다.
    Args:
        npy_file (str): npy 파일의 경로.
        csv_file (str): csv 파일의 경로.
        index (int): 처리할 행의 인덱스.
        instrument_file_counts (dict): 악기별 파일 수를 추적하는 딕셔너리.
    """
    try:
        # mel-spectrogram 데이터 로드
        spectrogram = np.load(npy_file)

        # CSV 데이터 로드
        df = pd.read_csv(csv_file)

        # 지정된 인덱스의 튜플 정보 가져오기
        row = df.iloc[index]
        start_frame = int(row["start_time"] // HOP_LENGTH)
        end_frame = int(row["end_time"] // HOP_LENGTH)

        # 1. 음이 400ms가 넘는 경우 자르고, 부족하면 패딩 추가 (400ms로 맞춤)
        original_clip = spectrogram[:, start_frame:end_frame].astype(np.float32)

        # 400ms에 맞게 클립을 자르거나 패딩 추가
        if original_clip.shape[1] > CLIP_DURATION_FRAMES:
            clipped = original_clip[:, :CLIP_DURATION_FRAMES]
        else:
            pad_width = CLIP_DURATION_FRAMES - original_clip.shape[1]
            clipped = np.pad(original_clip, ((0, 0), (0, pad_width)), mode="constant")

        # 2. 400ms 클립에 30ms 버퍼 추가 (460ms로 확장)
        start_frame_buffered = max(0, start_frame - BUFFER_SAMPLES // HOP_LENGTH)
        end_frame_buffered = start_frame_buffered + TOTAL_FRAMES

        # 버퍼를 적용한 클립 생성
        clip_buffered = spectrogram[:, start_frame_buffered:end_frame_buffered].astype(
            np.float32
        )

        # 클립 길이가 부족하면 패딩 추가
        if clip_buffered.shape[1] < TOTAL_FRAMES:
            pad_width = TOTAL_FRAMES - clip_buffered.shape[1]
            clip_buffered = np.pad(
                clip_buffered, ((0, 0), (0, pad_width)), mode="constant"
            )

        # 3. 저장할 시작 시간과 종료 시간 갱신
        new_start_time = row["start_time"] - BUFFER_DURATION_MS  # 30ms 앞당김
        new_end_time = new_start_time + TOTAL_CLIP_DURATION_MS  # 시작 후 460ms

        # 악기 및 음 정보 저장
        instrument = row["instrument"]
        instrument_name = INSTRUMENT_MAPPING.get(
            instrument, "Unknown"
        )  # 악기 이름 변환
        note = row["note"]

        # 악기별 파일 수 초기화
        if instrument_name not in instrument_file_counts:
            instrument_file_counts[instrument_name] = 0

        # 해당 악기의 파일 수 증가
        instrument_file_counts[instrument_name] += 1

        # 현재 하위 폴더 인덱스 계산
        subdir_index = (instrument_file_counts[instrument_name] - 1) // 10000 + 1

        # 악기별 폴더 및 하위 폴더 생성
        instrument_dir = os.path.join(OUTPUT_DIR, instrument_name)
        ensure_dir(instrument_dir)

        subdir_name = f"subdir{subdir_index}"
        subdir_path = os.path.join(instrument_dir, subdir_name)
        ensure_dir(subdir_path)

        # 클립 및 정보를 해당 하위 폴더에 저장
        output_file = os.path.join(
            subdir_path, f"{os.path.splitext(os.path.basename(npy_file))[0]}_{index}.h5"
        )

        with h5py.File(output_file, "w") as hf:
            hf.create_dataset(
                "clip", data=clip_buffered, compression="gzip", compression_opts=9
            )
            hf.create_dataset("instrument", data=np.array([instrument]))
            hf.create_dataset("note", data=np.array([note]))
            hf.create_dataset("start_time", data=np.array([new_start_time]))
            hf.create_dataset("end_time", data=np.array([new_end_time]))

        log_message = (
            f"클립 저장 완료: {output_file}, 악기: {instrument_name}, 음표: {note}, "
            f"새 시작 시간: {new_start_time}, 새 종료 시간: {new_end_time}"
        )
        logging.info(log_message)
    except Exception as e:
        logging.error(f"클립 처리 중 오류 발생: {e}")


def process_file(npy_file, csv_file, seed, instrument_file_counts):
    """
    하나의 npy 및 csv 파일 쌍을 처리합니다.
    Args:
        npy_file (str): npy 파일 이름.
        csv_file (str): csv 파일 이름.
        seed (int): 랜덤 시드.
        instrument_file_counts (dict): 악기별 파일 수를 추적하는 딕셔너리.
    """
    # 랜덤 시드 설정
    random.seed(seed)

    npy_path = os.path.join(NPY_DIR, npy_file)
    csv_path = os.path.join(CSV_DIR, csv_file)

    if os.path.splitext(npy_file)[0] == os.path.splitext(csv_file)[0]:
        # CSV 파일 로드
        df = pd.read_csv(csv_path)

        # 전체 행 중에서 랜덤하게 3000개 선택
        random_indices = random.sample(range(len(df)), min(3000, len(df)))

        # 악기별 파일 수를 추적하는 딕셔너리 초기화 (전역으로 관리되므로 생략)

        # 진행 상황을 시각화하기 위해 tqdm 사용
        for i, index in enumerate(
            tqdm(random_indices, desc=f"{npy_file} 클립 처리", unit="clip")
        ):
            logging.info(f"{npy_file}: {i+1}번째 클립 처리 중: {index}번째 행")
            process_and_save_clip(npy_path, csv_path, index, instrument_file_counts)

        logging.info(f"{npy_file}: 랜덤한 3000개의 클립 처리가 완료되었습니다.")


def main():
    """메인 함수로, 전체 프로세스를 실행합니다."""
    setup_logging()

    # 출력 폴더가 없으면 생성
    ensure_dir(OUTPUT_DIR)

    # npy와 csv 파일의 목록을 가져오기 (._ 파일 무시)
    npy_files = sorted(
        [
            f
            for f in os.listdir(NPY_DIR)
            if f.endswith(".npy") and not f.startswith("._")
        ]
    )
    csv_files = sorted(
        [
            f
            for f in os.listdir(CSV_DIR)
            if f.endswith(".csv") and not f.startswith("._")
        ]
    )

    # npy와 csv 파일 이름이 일치하는 파일만 처리
    npy_csv_pairs = [
        (npy_file, csv_file)
        for npy_file, csv_file in zip(npy_files, csv_files)
        if os.path.splitext(npy_file)[0] == os.path.splitext(csv_file)[0]
    ]

    if not npy_csv_pairs:
        logging.info("일치하는 npy 및 csv 파일 쌍이 없습니다.")
        return

    # 랜덤 시드 생성
    seeds = [random.randint(0, 99999) for _ in range(len(npy_csv_pairs))]

    # 악기별 파일 수를 전역으로 추적하기 위한 딕셔너리 초기화
    instrument_file_counts = {}

    # 전체 파일 처리 진행 상황을 시각화하기 위해 tqdm 사용
    for i, (npy_file, csv_file) in enumerate(
        tqdm(npy_csv_pairs, desc="전체 파일 처리 진행 상황", unit="file")
    ):
        seed = seeds[i]
        process_file(npy_file, csv_file, seed, instrument_file_counts)

    logging.info("모든 파일의 클립 생성이 완료되었습니다.")


if __name__ == "__main__":
    main()
