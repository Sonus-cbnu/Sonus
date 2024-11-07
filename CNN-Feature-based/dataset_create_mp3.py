import json
import os
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count, Manager
import numpy as np
from collections import Counter
import psutil
import time
from numpy.random import RandomState
import hashlib
import logging  # 로깅을 위한 라이브러리

# ----------------------------- 초기 설정 -----------------------------
# 폴더 경로 설정
chord_output_dir = "./chord_output"  # 악기 샘플이 저장된 디렉토리
data_output_dir = "./data"  # 생성된 데이터와 메타데이터를 저장할 디렉토리
metadata_file = os.path.join(data_output_dir, "metadata.json")  # 메타데이터 파일 경로

# 데이터 셋 개수 설정
data_length = 231980  # 생성할 샘플의 총 개수

# 데이터 폴더가 없으면 생성
os.makedirs(data_output_dir, exist_ok=True)  # 데이터 출력 디렉토리 생성

# 상수 정의
SAMPLE_LENGTH = 4000  # 각 샘플의 길이 (밀리초 단위, 4초)

# ----------------------------- 로깅 설정 -----------------------------
# 로깅을 위한 설정
logging.basicConfig(
    level=logging.INFO,  # 기본 로깅 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 포맷 설정
)

# 파일 핸들러 추가
file_handler = logging.FileHandler(os.path.join(data_output_dir, "sample_creation.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

# 로거에 핸들러 추가
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ----------------------------- 프로세스 우선순위 조정 함수 -----------------------------
def set_low_priority():
    """
    현재 프로세스의 우선순위를 낮추는 함수.
    시스템의 다른 작업에 영향을 최소화하기 위해 사용됩니다.
    """
    try:
        p = psutil.Process(os.getpid())  # 현재 프로세스 객체 가져오기
        p.nice(10)  # 프로세스의 우선순위를 낮춤 (macOS와 Unix-like 시스템에서)
        logger.info("프로세스 우선순위 낮춤.")
    except Exception as e:
        logger.error(
            f"프로세스 우선순위 설정 실패: {e}"
        )  # 예외 발생 시 에러 메시지 출력


# ----------------------------- 유틸리티 함수들 -----------------------------
def get_subfolder_path(sample_num):
    """
    샘플 번호에 따라 서브 폴더 경로를 생성하는 함수
    :param sample_num: 샘플 번호
    :return: 서브 폴더 경로
    """
    subfolder_index = (sample_num - 1) // 10000  # 샘플 번호를 기반으로 배치 인덱스 계산
    subfolder_path = os.path.join(
        data_output_dir, f"batch_{subfolder_index:04d}"
    )  # 서브 폴더 경로 생성
    os.makedirs(subfolder_path, exist_ok=True)  # 서브 폴더 생성 (존재하지 않으면)
    return subfolder_path  # 생성된 서브 폴더 경로 반환


def get_relative_path(subfolder_path, data_output_dir):
    """
    서브 폴더의 상대 경로를 반환하는 함수
    :param subfolder_path: 서브 폴더의 절대 경로
    :param data_output_dir: 데이터 출력 디렉토리
    :return: 상대 경로
    """
    return os.path.relpath(subfolder_path, data_output_dir)  # 상대 경로 계산하여 반환


def add_silence(audio, random_state):
    """
    오디오에 무작위로 무음을 추가하는 함수
    :param audio: 원본 오디오 (AudioSegment 객체)
    :param random_state: 난수 생성기
    :return: 무음이 추가된 오디오, 시작/중간/끝 무음 길이
    """
    if random_state.rand() > 0.5:
        # 50% 확률로 무음 추가하지 않음
        return audio, 0, 0, 0

    max_silence = min(
        2000, len(audio)
    )  # 최대 무음 길이 설정 (2000ms 또는 오디오 길이 중 작은 값)
    silence_duration = random_state.randint(
        100, max_silence
    )  # 무음 길이를 랜덤하게 설정

    start_silence = end_silence = middle_silence = 0  # 초기 무음 길이 설정

    split_point = random_state.randint(0, len(audio))  # 오디오를 나눌 지점 랜덤 선택
    audio = (
        audio[:split_point]  # 나눌 지점 전까지의 오디오
        + AudioSegment.silent(duration=silence_duration)  # 무음 추가
        + audio[split_point:]  # 나눌 지점 이후의 오디오
    )
    middle_silence = silence_duration  # 중간에 추가된 무음 길이 기록

    return (
        audio,
        start_silence,
        middle_silence,
        end_silence,
    )  # 수정된 오디오와 무음 길이 반환


def process_audio(audio, random_state):
    """
    오디오 처리 함수 (볼륨 조절, 무음 추가 및 4초로 길이 조정)
    :param audio: 원본 오디오 (AudioSegment 객체)
    :param random_state: 난수 생성기
    :return: 처리된 오디오, 볼륨 변화, 시작/중간/끝 무음 길이
    """
    # 볼륨 변경 (스케일링)
    volume_change = random_state.uniform(
        0.5, 1.5
    )  # 볼륨 변화를 랜덤하게 설정 (0.5배 ~ 1.5배)
    audio = audio + (10 * np.log10(volume_change))  # 볼륨 조절 (dB 단위)

    # 무음 추가
    audio, start_silence, middle_silence, end_silence = add_silence(audio, random_state)

    # 샘플 길이를 4초로 조정
    if len(audio) > SAMPLE_LENGTH:
        audio = audio[:SAMPLE_LENGTH]  # 4초보다 길면 자름
    elif len(audio) < SAMPLE_LENGTH:
        audio = audio + AudioSegment.silent(
            duration=SAMPLE_LENGTH - len(audio)
        )  # 4초보다 짧으면 무음 추가

    return (
        audio,
        volume_change,
        start_silence,
        middle_silence,
        end_silence,
    )  # 처리된 오디오와 관련 정보 반환


def select_instruments(instrument_folders, n, random_state, instrument_usage_count):
    """
    균형있게 악기를 선택하는 함수
    :param instrument_folders: 사용 가능한 악기 폴더 목록
    :param n: 선택할 악기 수
    :param random_state: 난수 생성기
    :param instrument_usage_count: 악기 사용 횟수 카운터
    :return: 선택된 악기 목록
    """
    # 각 악기의 선택 확률을 계산 (사용 횟수가 적을수록 선택될 확률이 높음)
    weights = [
        1 / (instrument_usage_count.get(instrument, 0) + 1)
        for instrument in instrument_folders
    ]
    # 확률 분포를 정규화하여 악기 선택
    selected_instruments = random_state.choice(
        instrument_folders, size=n, replace=False, p=np.array(weights) / np.sum(weights)
    )
    return selected_instruments  # 선택된 악기 목록 반환


# ----------------------------- 샘플 생성 함수 -----------------------------
def create_sample(args):
    """
    샘플 생성 함수 (멀티프로세싱을 위해 별도 함수로 분리)
    :param args: (sample_num, shared_instrument_usage_count, shared_sample_usage_count)
    :return: 생성된 샘플의 메타데이터 (JSON 형식), 악기 사용 횟수, 샘플 사용 횟수
    """
    sample_num, shared_instrument_usage_count, shared_sample_usage_count = (
        args  # 인자 언패킹
    )

    # 고유한 시드 값을 생성하기 위해 현재 시간, 프로세스 ID, 샘플 번호를 사용
    current_time = time.time_ns()  # 현재 시간을 나노초 단위로 가져옴
    process_id = os.getpid()  # 현재 프로세스 ID 가져오기
    complex_seed = f"{current_time}_{process_id}_{sample_num}"  # 시드 문자열 생성

    hash_object = hashlib.sha256(complex_seed.encode())  # SHA-256 해시 객체 생성
    seed_value = int(hash_object.hexdigest(), 16) % (
        2**32 - 1
    )  # 해시값을 정수로 변환하여 시드 값 생성

    random_state = np.random.RandomState(seed_value)  # 랜덤 상태 객체 생성

    # 사용 가능한 악기 폴더 목록 가져오기
    instrument_folders = [
        f
        for f in os.listdir(chord_output_dir)
        if os.path.isdir(os.path.join(chord_output_dir, f))
    ]

    n = random_state.randint(1, 17)  # 선택할 악기 수를 랜덤하게 결정 (1 ~ 16)
    selected_instruments = select_instruments(
        instrument_folders, n, random_state, shared_instrument_usage_count
    )  # 악기 선택
    combined_sample = None  # 최종 결합 샘플 초기화
    instrument_details = []  # 선택된 악기들의 세부 정보 저장 리스트

    local_instrument_count = Counter()  # 로컬 악기 사용 횟수 카운터
    local_sample_count = Counter()  # 로컬 샘플 사용 횟수 카운터

    # 선택된 각 악기에 대해 샘플링 및 오디오 처리
    for instrument in selected_instruments:
        instrument_path = os.path.join(chord_output_dir, instrument)  # 악기 폴더 경로
        sample_files = [
            f
            for f in os.listdir(instrument_path)
            if f.endswith(".wav") and not f.startswith("._")
        ]  # 해당 악기의 .wav 파일 목록 가져오기
        if sample_files:
            selected_sample = random_state.choice(
                sample_files
            )  # 샘플 파일 중 하나를 랜덤 선택
            sample_path = os.path.join(
                instrument_path, selected_sample
            )  # 선택된 샘플의 전체 경로

            # 로컬 카운터 업데이트
            local_instrument_count[instrument] += 1
            local_sample_count[os.path.join(instrument, selected_sample)] += 1

            try:
                # 오디오 파일 로드 및 처리
                audio = AudioSegment.from_wav(sample_path)  # .wav 파일 로드
                audio = audio.set_channels(1)  # 모노로 변환
                (
                    processed_audio,
                    volume_change,
                    start_silence,
                    middle_silence,
                    end_silence,
                ) = process_audio(
                    audio, random_state
                )  # 오디오 처리

                # 악기 세부 정보 추가
                instrument_details.append(
                    {
                        "instrument": instrument,
                        "sample": selected_sample,
                        "volume_change": volume_change,
                        "start_silence": start_silence,
                        "middle_silence": middle_silence,
                        "end_silence": end_silence,
                        # "pitch_change": pitch_change,  # 피치 변경 정보 제거
                    }
                )

                # 처리된 오디오를 결합 샘플에 오버레이
                if combined_sample is None:
                    combined_sample = processed_audio  # 첫 번째 악기는 직접 할당
                else:
                    combined_sample = combined_sample.overlay(
                        processed_audio
                    )  # 기존 샘플에 오버레이
            except Exception as e:
                logger.error(f"오디오 처리 실패: {sample_path}, 에러: {e}")
                continue  # 에러 발생 시 다음 악기로 이동

    if combined_sample:
        combined_sample = combined_sample[:SAMPLE_LENGTH]  # 결합 샘플을 4초로 자름

        # 샘플을 저장할 서브폴더 경로 및 파일 이름 생성
        subfolder_path = get_subfolder_path(sample_num)
        output_file = os.path.join(subfolder_path, f"concerto_sample_{sample_num}.mp3")
        try:
            combined_sample.export(
                output_file, format="mp3", bitrate="128k"
            )  # MP3 형식으로 저장
            logger.info(
                f"{sample_num}번째 협주곡 샘플 저장 완료: {output_file}"
            )  # 콘솔 및 파일에 로그 기록
        except Exception as e:
            logger.error(f"샘플 저장 실패: {output_file}, 에러: {e}")
            return None  # 저장 실패 시 None 반환

        relative_path = get_relative_path(
            subfolder_path, data_output_dir
        )  # 상대 경로 계산

        return {
            "metadata": {
                "sample_name": f"concerto_sample_{sample_num}.mp3",  # 샘플 이름
                "instruments": [
                    detail["instrument"] for detail in instrument_details
                ],  # 사용된 악기 목록
                "instrument_details": instrument_details,  # 악기 세부 정보
                "relative_path": relative_path,  # 서브폴더의 상대 경로
            },
            "instrument_count": dict(local_instrument_count),  # 악기 사용 횟수
            "sample_count": dict(local_sample_count),  # 샘플 사용 횟수
        }
    else:
        logger.warning(
            f"{sample_num}번째 협주곡 샘플 생성 실패: 선택된 샘플이 없습니다."
        )
        return None  # 샘플 생성 실패 시 None 반환


# ----------------------------- 메타데이터 저장 함수 -----------------------------
def save_metadata(metadata, file_path, is_subfolder=False):
    """
    메타데이터를 JSON 파일에 저장하는 함수
    :param metadata: 저장할 메타데이터 (딕셔너리)
    :param file_path: 저장할 파일 경로
    :param is_subfolder: 서브폴더 메타데이터 여부 (기본값: False)
    """
    try:
        os.makedirs(
            os.path.dirname(file_path), exist_ok=True
        )  # 파일 경로의 디렉토리 생성 (존재하지 않으면)
        with open(file_path, "a") as f:  # 파일을 append 모드로 열기
            if is_subfolder:
                # 서브폴더 메타데이터의 경우 상대 경로 제거
                metadata_copy = metadata.copy()
                metadata_copy.pop("relative_path", None)
                json.dump(metadata_copy, f)  # JSON으로 덤프
            else:
                json.dump(metadata, f)  # 메인 메타데이터 덤프
            f.write("\n")  # 각 메타데이터 항목을 새 줄에 저장
        if is_subfolder:
            logger.info(f"서브폴더 메타데이터 저장 완료: {file_path}")
        else:
            logger.info(f"메인 메타데이터 저장 완료: {file_path}")
    except Exception as e:
        logger.error(f"메타데이터 저장 실패: {file_path}, 에러: {e}")


# ----------------------------- 메인 실행 부분 -----------------------------
if __name__ == "__main__":
    set_low_priority()  # 프로세스 우선순위 낮추기

    total_cores = cpu_count()  # 시스템의 총 CPU 코어 수 가져오기
    num_processes = max(
        1, int(total_cores * 0.6)
    )  # 사용할 프로세스 수 계산 (총 코어의 60%)

    logger.info(f"총 CPU 코어 수: {total_cores}")
    logger.info(f"사용할 프로세스 수: {num_processes}")

    # 메타데이터 파일 초기화 (내용 비움)
    try:
        with open(metadata_file, "w") as f:
            f.write("")
        logger.info(f"메타데이터 파일 초기화 완료: {metadata_file}")
    except Exception as e:
        logger.error(f"메타데이터 파일 초기화 실패: {metadata_file}, 에러: {e}")

    # 멀티프로세싱을 위한 Manager 객체 생성
    manager = Manager()
    shared_instrument_usage_count = (
        manager.dict()
    )  # 악기 사용 횟수를 공유하기 위한 딕셔너리
    shared_sample_usage_count = (
        manager.dict()
    )  # 샘플 사용 횟수를 공유하기 위한 딕셔너리

    subfolder_metadata_files = set()  # 서브폴더별 메타데이터 파일 추적을 위한 집합

    # 멀티프로세싱 풀을 사용하여 샘플 생성 작업 수행
    with Pool(processes=num_processes) as pool:
        # 각 샘플 번호와 공유 카운터를 인자로 하는 튜플 리스트 생성
        args = [
            (i, shared_instrument_usage_count, shared_sample_usage_count)
            for i in range(1, data_length + 1)
        ]
        logger.info("멀티프로세싱을 통해 샘플 생성 시작.")
        results = pool.map(create_sample, args)  # 샘플 생성 함수 병렬 실행
        logger.info("멀티프로세싱을 통한 샘플 생성 완료.")

        # 생성된 각 샘플의 결과 처리
        for result in results:
            if result:
                metadata = result["metadata"]  # 샘플의 메타데이터 가져오기
                save_metadata(metadata, metadata_file)  # 메인 메타데이터 파일에 저장

                # 서브폴더의 메타데이터 파일 경로 생성
                subfolder_path = os.path.join(
                    data_output_dir, metadata["relative_path"]
                )
                subfolder_metadata_file = os.path.join(subfolder_path, "metadata.json")
                if subfolder_metadata_file not in subfolder_metadata_files:
                    # 서브폴더별 메타데이터 파일이 없으면 생성
                    try:
                        with open(subfolder_metadata_file, "w") as f:
                            f.write("")
                        subfolder_metadata_files.add(
                            subfolder_metadata_file
                        )  # 생성된 파일을 추적 집합에 추가
                        logger.info(
                            f"서브폴더 메타데이터 파일 생성: {subfolder_metadata_file}"
                        )
                    except Exception as e:
                        logger.error(
                            f"서브폴더 메타데이터 파일 생성 실패: {subfolder_metadata_file}, 에러: {e}"
                        )
                save_metadata(
                    metadata, subfolder_metadata_file, is_subfolder=True
                )  # 서브폴더 메타데이터 파일에 저장

                # 공유 카운터 업데이트 (악기 사용 횟수)
                for instrument, count in result["instrument_count"].items():
                    shared_instrument_usage_count[instrument] = (
                        shared_instrument_usage_count.get(instrument, 0) + count
                    )

                # 공유 카운터 업데이트 (샘플 사용 횟수)
                for sample, count in result["sample_count"].items():
                    shared_sample_usage_count[sample] = (
                        shared_sample_usage_count.get(sample, 0) + count
                    )

    # Manager.dict()에서 가져온 딕셔너리를 일반 딕셔너리로 변환
    instrument_usage_count = dict(shared_instrument_usage_count)
    sample_usage_count = dict(shared_sample_usage_count)

    logger.info("모든 샘플 생성 완료.")

    # ----------------------------- 데이터셋 분석 -----------------------------
    logger.info("악기 사용 통계:")
    # 악기 사용 횟수를 내림차순으로 정렬하여 출력
    for instrument, count in sorted(
        instrument_usage_count.items(), key=lambda item: item[1], reverse=True
    ):
        logger.info(f"{instrument}: {count}")

    logger.info("샘플 사용 통계 (상위 10개):")
    # 샘플 사용 횟수를 내림차순으로 정렬하여 상위 10개만 출력
    top_samples = sorted(
        sample_usage_count.items(), key=lambda item: item[1], reverse=True
    )[:10]
    for sample, count in top_samples:
        logger.info(f"{sample}: {count}")
