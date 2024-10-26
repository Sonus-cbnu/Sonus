import os
import json
import math
import numpy as np
import librosa
from tqdm import tqdm
import h5py

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
    ReduceLROnPlateau,
)

import keras_tuner as kt
import random
import re
import gc

# ----------------------------- 설정 -----------------------------
# 배치 크기와 에폭 수를 상단에 변수로 정의
BATCH_SIZE = 128  # 배치 크기 설정
EPOCHS = 50  # 에폭 수 설정
MAX_TRIALS = 10  # 하이퍼파라미터 튜닝 최대 트라이얼 수

APPLY_SMOTE = False  # SMOTE 적용 여부 설정 (비활성화)

# 하이퍼파라미터 튜닝 활성화 여부 설정
ENABLE_HYPERPARAMETER_TUNING = False  # 하이퍼파라미터 튜닝 활성화 여부

# 로그 콜백 활성화 여부 설정
ENABLE_TENSORBOARD = True  # TensorBoard 로그 활성화 여부
ENABLE_CSVLOGGER = True  # CSVLogger 로그 활성화 여부

# 학습률 설정
LEARNING_RATE = 1e-4  # 기본 학습률

# 데이터 출력 디렉토리 설정
data_output_dir = "./data"  # 생성된 데이터셋이 저장된 디렉토리
metadata_file = os.path.join(data_output_dir, "metadata.json")  # 메타데이터 파일 경로

# 캐시 디렉토리 설정
cache_dir = "./cache"  # 캐시 데이터를 저장할 디렉토리
os.makedirs(cache_dir, exist_ok=True)
hdf5_file = os.path.join(
    cache_dir, "preprocessed_data.h5"
)  # 전처리된 데이터를 저장할 HDF5 파일 경로

# MFCC 추출을 위한 설정
n_mfcc = 40  # 추출할 MFCC 계수의 수
max_len = 174  # MFCC 벡터의 최대 길이 (시간 축)

# 모델 저장 디렉토리 설정
models_output_dir = "./models"  # 학습된 모델을 저장할 디렉토리
os.makedirs(models_output_dir, exist_ok=True)

# 로그 디렉토리 설정 (TensorBoard)
logs_dir = "./logs"
os.makedirs(logs_dir, exist_ok=True)

# Keras Tuner 디렉토리 설정
kt_directory = "./kt_dir"
os.makedirs(kt_directory, exist_ok=True)

# ----------------------------- GPU 메모리 관리 -----------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # GPU 메모리 사용을 제한
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
                ],  # 메모리 할당을 4GB로 제한
            )
        print(f"활성화된 GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)

# ----------------------------- CPU 사용 제한 설정 -----------------------------
# 환경 변수를 설정하여 NumPy 등의 라이브러리의 스레드 수 제한
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# TensorFlow의 스레드 수 제한
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)


def sanitize_name(name):
    """
    문자열에서 알파벳, 숫자, 언더스코어만 남기고 나머지는 제거하는 함수
    :param name: 원본 문자열
    :return: 정제된 문자열
    """
    return re.sub(r"[^a-zA-Z0-9_]", "", name.replace(" ", "_"))


def load_metadata(metadata_path):
    """
    메타데이터를 로드하는 함수
    :param metadata_path: 메타데이터 파일의 경로
    :return: 메타데이터 리스트
    """
    metadata = []
    with open(metadata_path, "r") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    """
    오디오 파일에서 MFCC 특징을 추출하는 함수
    :param file_path: 오디오 파일의 경로
    :param n_mfcc: 추출할 MFCC 계수의 수
    :param max_len: MFCC 벡터의 최대 길이 (시간 축)
    :return: 고정된 크기의 MFCC 넘파이 배열
    """
    try:
        # 오디오 로드
        y, sr = librosa.load(file_path, sr=None, mono=True)
        # MFCC 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # 길이 조정 (패딩 또는 자르기)
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"MFCC 추출 실패: {file_path}, 에러: {e}")
        return np.zeros((n_mfcc, max_len))  # 실패 시 0으로 채운 배열 반환


def preprocess_and_save_data(metadata, data_dir, hdf5_path):
    """
    데이터 전처리 및 HDF5 파일로 저장하는 함수
    :param metadata: 메타데이터 리스트
    :param data_dir: 오디오 데이터가 저장된 디렉토리
    :param hdf5_path: 저장할 HDF5 파일의 경로
    """
    # 전체 데이터 수
    total_samples = len(metadata)
    # HDF5 파일 생성
    with h5py.File(hdf5_path, "w") as h5f:
        # 데이터셋 생성
        X_ds = h5f.create_dataset(
            "X", shape=(total_samples, n_mfcc, max_len), dtype=np.float32
        )
        # 레이블 저장을 위한 리스트
        Y_list = []
        # 모든 데이터에 대해 전처리 수행
        for idx, data in enumerate(tqdm(metadata, desc="전처리 중")):
            # 오디오 파일의 경로 생성
            sample_path = os.path.join(
                data_dir, data["relative_path"], data["sample_name"]
            )
            # MFCC 특징 추출
            mfcc = extract_mfcc(sample_path, n_mfcc, max_len)
            # HDF5 데이터셋에 저장
            X_ds[idx] = mfcc
            # 레이블 수집
            instruments = data["instruments"]
            Y_list.append(instruments)
        # 레이블을 HDF5 파일에 저장 (JSON-encoded strings)
        Y_encoded_strings = [json.dumps(instr_list) for instr_list in Y_list]
        dt = h5py.string_dtype(encoding="utf-8")
        Y_ds = h5f.create_dataset(
            "Y", data=np.array(Y_encoded_strings, dtype=object), dtype=dt
        )


def load_preprocessed_data(hdf5_path):
    """
    전처리된 데이터를 HDF5 파일에서 로드하는 함수
    :param hdf5_path: HDF5 파일의 경로
    :return: 특징 배열 X와 레이블 리스트 Y
    """
    with h5py.File(hdf5_path, "r") as h5f:
        X = h5f["X"][:]
        Y = h5f["Y"][:]
    return X, Y


def encode_labels(Y):
    """
    레이블을 이진 벡터로 인코딩하는 함수
    :param Y: 레이블 리스트 (JSON-encoded strings)
    :return: 인코딩된 레이블 배열, 레이블 변환기 (mlb), 모든 악기 리스트
    """
    # 모든 악기의 집합을 생성하고 정렬
    all_instruments = set()
    parsed_Y = []
    for instruments in Y:
        if isinstance(instruments, bytes):
            instruments = instruments.decode("utf-8")
        if isinstance(instruments, str):
            try:
                # JSON 형식의 리스트로 파싱
                instruments = json.loads(instruments)
            except json.JSONDecodeError:
                # 파싱 실패 시, 쉼표로 분리
                instruments = instruments.split(",")
        if not isinstance(instruments, list):
            instruments = []
        # 악기 이름 정리 및 집합에 추가
        instruments = [instr.strip() for instr in instruments if instr.strip()]
        all_instruments.update(instruments)
        parsed_Y.append(instruments)
    all_instruments = sorted(list(all_instruments))

    if not all_instruments:
        print(
            "경고: 모든 악기 리스트가 비어 있습니다. 데이터에 악기 레이블이 없는 것으로 보입니다."
        )

    # MultiLabelBinarizer를 사용하여 레이블 이진 인코딩
    mlb = MultiLabelBinarizer(classes=all_instruments)
    Y_encoded = mlb.fit_transform(parsed_Y)
    return Y_encoded, mlb, all_instruments


def augment_batch_tf(batch_X):
    """
    TensorFlow 기반의 데이터 증강 함수
    :param batch_X: 배치의 MFCC 데이터 (Tensor)
    :return: 증강된 배치의 MFCC 데이터
    """
    # 볼륨 변경 (스케일링)
    scale = tf.random.uniform([], 0.8, 1.2)
    batch_X = batch_X * scale

    # 노이즈 추가
    noise = tf.random.normal(shape=tf.shape(batch_X), mean=0.0, stddev=0.05)
    batch_X = batch_X + noise

    # 주파수 축에서 마스킹 (SpecAugment 유사)
    freq_masking = tf.random.uniform([], 0, n_mfcc // 10, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, tf.maximum(n_mfcc - freq_masking, 1), dtype=tf.int32)
    # Create mask for frequency
    mask_freq = tf.concat(
        [
            tf.ones([f0, max_len, 1], dtype=batch_X.dtype),
            tf.zeros([freq_masking, max_len, 1], dtype=batch_X.dtype),
            tf.ones([n_mfcc - f0 - freq_masking, max_len, 1], dtype=batch_X.dtype),
        ],
        axis=0,
    )  # shape [n_mfcc, max_len,1]

    # Tile mask_freq to match batch size
    batch_size = tf.shape(batch_X)[0]
    mask_freq = tf.expand_dims(mask_freq, 0)  # [1, n_mfcc, max_len,1]
    mask_freq = tf.tile(
        mask_freq, [batch_size, 1, 1, 1]
    )  # [batch_size, n_mfcc, max_len,1]
    batch_X = batch_X * mask_freq

    # 시간 축에서 마스킹
    time_masking = tf.random.uniform([], 0, max_len // 10, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, tf.maximum(max_len - time_masking, 1), dtype=tf.int32)
    # Create mask for time
    mask_time = tf.concat(
        [
            tf.ones([n_mfcc, t0, 1], dtype=batch_X.dtype),
            tf.zeros([n_mfcc, time_masking, 1], dtype=batch_X.dtype),
            tf.ones([n_mfcc, max_len - t0 - time_masking, 1], dtype=batch_X.dtype),
        ],
        axis=1,
    )  # shape [n_mfcc, max_len,1]
    mask_time = tf.expand_dims(mask_time, 0)  # [1, n_mfcc, max_len,1]
    mask_time = tf.tile(
        mask_time, [batch_size, 1, 1, 1]
    )  # [batch_size, n_mfcc, max_len,1]
    batch_X = batch_X * mask_time

    return batch_X


def tuner_generator(X, Y, batch_size, shuffle=True, augment=False):
    """
    데이터 제너레이터 (Keras Tuner 및 모델 훈련용)
    :param X: 특징 배열 (numpy 배열)
    :param Y: 레이블 배열 (numpy 배열)
    :param batch_size: 배치 크기
    :param shuffle: 데이터 섞기 여부
    :param augment: 데이터 증강 여부
    :return: tf.data.Dataset 객체
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.batch(batch_size)

    # 채널 차원 추가
    dataset = dataset.map(
        lambda x, y: (tf.expand_dims(x, -1), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if augment:
        # TensorFlow 기반 데이터 증강 함수 사용
        dataset = dataset.map(
            lambda x, y: (augment_batch_tf(x), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    # 정규화 단계
    def preprocess(batch_X, batch_Y):
        # MFCC 데이터를 정규화 (평균 0, 분산 1)
        mean = tf.reduce_mean(batch_X, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(batch_X, axis=[1, 2, 3], keepdims=True)
        batch_X = (batch_X - mean) / (std + 1e-6)
        return batch_X, batch_Y

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Prefetching을 통해 GPU와 데이터 로딩 간 병목 현상 완화
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def build_model(input_shape, num_classes, dropout_rate=0.5, l2_reg=0.001):
    """
    CNN 모델을 구축하는 함수
    :param input_shape: 입력 데이터의 형태
    :param num_classes: 출력 클래스의 수
    :param dropout_rate: 드롭아웃 비율
    :param l2_reg: L2 정규화 계수
    :return: 케라스 모델 객체
    """
    from tensorflow.keras import regularizers

    inputs = Input(shape=input_shape)

    # 첫 번째 컨볼루션 블록
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # 두 번째 컨볼루션 블록
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # 세 번째 컨볼루션 블록
    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # 완전 연결 층
    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    # 출력 층 (시그모이드 활성화 함수 사용)
    outputs = Dense(num_classes, activation="sigmoid")(x)

    # 모델 생성
    model = Model(inputs, outputs)
    return model


# ----------------------------- 하이퍼파라미터 튜닝 클래스 정의 -----------------------------
class InstrumentHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        from tensorflow.keras import regularizers

        inputs = Input(shape=self.input_shape)

        # 하이퍼파라미터를 사용한 컨볼루션 블록 추가
        x = inputs
        for i in range(hp.Int("num_conv_blocks", 2, 4)):
            filters = hp.Choice(f"filters_{i}", values=[32, 64, 128])
            x = Conv2D(
                filters,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(hp.Float("l2_reg", 1e-4, 1e-2)),
            )(x)
            x = MaxPooling2D((2, 2))(x)
            x = BatchNormalization()(x)

        x = Flatten()(x)
        for j in range(hp.Int("num_dense_layers", 1, 3)):
            units = hp.Int(f"units_{j}", min_value=64, max_value=512, step=64)
            x = Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(
                    hp.Float("l2_reg_dense", 1e-4, 1e-2)
                ),
            )(x)
            dropout_rate = hp.Float("dropout_rate", 0.2, 0.5, step=0.1)
            x = Dropout(dropout_rate)(x)

        outputs = Dense(self.num_classes, activation="sigmoid")(x)

        model = Model(inputs, outputs)

        # 하이퍼파라미터로 학습률 조정
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model


# ----------------------------- 메인 실행 부분 -----------------------------
if __name__ == "__main__":
    # 1. 메타데이터 로드
    print("메타데이터 로드 중...")
    metadata = load_metadata(metadata_file)
    print(f"메타데이터 로드 완료: {len(metadata)} 샘플")

    # 2. 데이터 전처리 및 저장 (이미 전처리된 데이터가 없을 경우)
    if not os.path.exists(hdf5_file):
        print("데이터 전처리 시작...")
        preprocess_and_save_data(metadata, data_output_dir, hdf5_file)
        print("데이터 전처리 및 저장 완료.")
    else:
        print("전처리된 데이터 파일이 존재합니다. 로드합니다.")

    # 3. 전처리된 데이터 로드
    print("전처리된 데이터 로드 중...")
    X, Y = load_preprocessed_data(hdf5_file)
    print(f"전처리된 데이터 로드 완료: X shape={X.shape}, Y shape={Y.shape}")

    # 4. 레이블 인코딩
    print("레이블 인코딩 중...")
    Y_encoded, mlb, all_instruments = encode_labels(Y)
    print(f"레이블 인코딩 완료: {len(all_instruments)} 종류의 악기")

    if len(all_instruments) == 0:
        raise ValueError("레이블 인코딩 실패: 악기 리스트가 비어 있습니다.")

    # 5. 데이터셋 분할
    print("데이터셋 분할 중...")
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y_encoded, test_size=0.2, random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42
    )
    print(f"데이터셋 분할 완료:")
    print(f" - 훈련 세트: {X_train.shape[0]} 샘플")
    print(f" - 검증 세트: {X_val.shape[0]} 샘플")
    print(f" - 테스트 세트: {X_test.shape[0]} 샘플")

    # 6. 개별 악기별 모델 구축 및 학습
    print("개별 악기별 모델 구축 및 학습 시작...")

    for idx, instrument in enumerate(all_instruments):
        sanitized_instrument = sanitize_name(instrument)
        print(f"\n=== 악기: {instrument} ({idx + 1}/{len(all_instruments)}) ===")

        # 이진 레이블 생성
        Y_train_binary = Y_train[:, idx]
        Y_val_binary = Y_val[:, idx]
        Y_test_binary = Y_test[:, idx]

        # SMOTE를 사용하여 데이터 불균형 처리 여부 결정
        if APPLY_SMOTE:
            print("SMOTE를 사용하여 데이터 불균형 처리 중...")
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=42)
            try:
                X_train_balanced, Y_train_balanced = smote.fit_resample(
                    X_train.reshape(X_train.shape[0], -1), Y_train_binary
                )
                # 재구성된 X_train_balanced를 원래의 형태로 변환
                X_train_balanced = X_train_balanced.reshape(-1, n_mfcc, max_len)
                print(f"SMOTE 적용 완료: {X_train_balanced.shape[0]} 샘플")
            except Exception as e:
                print(f"SMOTE 적용 실패: {e}")
                # SMOTE 실패 시 원본 데이터 사용
                X_train_balanced, Y_train_balanced = X_train, Y_train_binary
        else:
            print("SMOTE를 사용하지 않습니다.")
            X_train_balanced, Y_train_balanced = X_train, Y_train_binary

        # 클래스 가중치 설정
        if not APPLY_SMOTE:
            # 클래스 가중치 계산 (binary classification)
            classes = np.unique(Y_train_binary)
            class_weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=Y_train_binary
            )
            class_weight_dict = {
                cls: weight for cls, weight in zip(classes, class_weights)
            }
            print(f"클래스 가중치 계산 완료: {class_weight_dict}")
        else:
            # SMOTE 적용 시 클래스 가중치 설정하지 않음
            class_weight_dict = None
            print("SMOTE 적용으로 클래스 가중치 설정하지 않습니다.")

        if ENABLE_HYPERPARAMETER_TUNING:
            # 하이퍼파라미터 튜닝을 사용하는 경우
            print("하이퍼파라미터 튜닝 시작...")
            input_shape = (n_mfcc, max_len, 1)
            hypermodel = InstrumentHyperModel(input_shape, num_classes=1)

            # 프로젝트 이름을 악기 이름으로 고유하게 생성
            project_name = f"hypermodel_{sanitized_instrument}"

            tuner = kt.BayesianOptimization(
                hypermodel,
                objective="val_accuracy",
                max_trials=MAX_TRIALS,
                executions_per_trial=1,
                directory=kt_directory,
                project_name=project_name,
                overwrite=True,
            )

            # 데이터셋 생성
            tuner_train_gen = tuner_generator(
                X_train_balanced,
                Y_train_balanced,
                batch_size=BATCH_SIZE,
                shuffle=True,
                augment=True,
            )
            tuner_val_gen = tuner_generator(
                X_val,
                Y_val_binary,
                batch_size=BATCH_SIZE,
                shuffle=False,
                augment=False,
            )

            # 학습률 스케줄러 설정
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
            )

            # 튜너 검색 수행
            tuner.search(
                tuner_train_gen,
                validation_data=tuner_val_gen,
                epochs=EPOCHS,
                steps_per_epoch=math.ceil(len(Y_train_balanced) / BATCH_SIZE),
                validation_steps=math.ceil(len(Y_val_binary) / BATCH_SIZE),
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=3,
                        restore_best_weights=True,
                    ),
                    lr_scheduler,
                ],
            )

            # 최적의 하이퍼파라미터와 모델을 가져오기
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            print(f"최적의 하이퍼파라미터: {best_hyperparameters.values}")

        else:
            # 하이퍼파라미터 튜닝을 사용하지 않는 경우
            # 모델 구축
            input_shape = (n_mfcc, max_len, 1)
            num_classes = 1  # 이진 분류
            model = build_model(
                input_shape, num_classes, dropout_rate=0.5, l2_reg=0.001
            )

            # 옵티마이저 설정 (legacy optimizer 사용)
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)

            # 모델 컴파일
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # 모델 요약 출력
            model.summary()

            best_model = model  # 모델 할당

        # 모델 훈련을 위한 데이터셋 생성
        train_gen = tuner_generator(
            X_train_balanced,
            Y_train_balanced,
            batch_size=BATCH_SIZE,
            shuffle=True,
            augment=True,
        )
        val_gen = tuner_generator(
            X_val, Y_val_binary, batch_size=BATCH_SIZE, shuffle=False, augment=False
        )
        test_gen = tuner_generator(
            X_test, Y_test_binary, batch_size=BATCH_SIZE, shuffle=False, augment=False
        )

        # 데이터셋의 크기 계산 (ceil 사용)
        train_steps = math.ceil(len(Y_train_balanced) / BATCH_SIZE)
        val_steps = math.ceil(len(Y_val_binary) / BATCH_SIZE)
        test_steps = math.ceil(len(Y_test_binary) / BATCH_SIZE)

        # 모델 저장 디렉토리 설정
        tensorboard_log_dir = os.path.join(logs_dir, f"{sanitized_instrument}")
        os.makedirs(tensorboard_log_dir, exist_ok=True)

        # CSVLogger 로그 파일 경로 설정
        csv_logger_path = os.path.join(logs_dir, f"{sanitized_instrument}_training.log")

        # 콜백 설정 (EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger)
        callbacks = []
        if ENABLE_TENSORBOARD:
            callbacks.append(
                TensorBoard(
                    log_dir=tensorboard_log_dir,
                    histogram_freq=0,
                    write_graph=False,
                    write_images=False,
                    update_freq="epoch",
                )
            )
        if ENABLE_CSVLOGGER:
            callbacks.append(CSVLogger(csv_logger_path, append=True, separator=","))
        # EarlyStopping 및 ReduceLROnPlateau는 이미 포함됨 if hyperparameter tuning is enabled
        if not ENABLE_HYPERPARAMETER_TUNING:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,  # 과적합 방지를 위해 patience를 3으로 설정
                    verbose=1,
                    restore_best_weights=True,
                )
            )
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
            )
            callbacks.append(lr_scheduler)

        callbacks.append(
            ModelCheckpoint(
                os.path.join(
                    models_output_dir, f"best_model_{sanitized_instrument}.h5"
                ),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
                save_weights_only=False,
            )
        )

        # 모델 훈련
        print("모델 훈련 시작...")
        try:
            history = best_model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                validation_data=val_gen,
                validation_steps=val_steps,
                epochs=EPOCHS,
                callbacks=callbacks,
                class_weight=class_weight_dict,
            )
            print("모델 훈련 완료.")
        except Exception as e:
            print(f"모델 훈련 중 오류 발생: {e}")
            continue

        # 최상의 모델 로드 (.h5 형식)
        try:
            best_model = tf.keras.models.load_model(
                os.path.join(
                    models_output_dir, f"best_model_{sanitized_instrument}.h5"
                ),
                compile=False,
            )
            # 옵티마이저 재설정
            best_model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            continue

        # 모델 평가
        print("모델 평가 중...")
        try:
            loss, accuracy = best_model.evaluate(test_gen, steps=test_steps)
            print(f"테스트 손실: {loss}")
            print(f"테스트 정확도: {accuracy}")
        except Exception as e:
            print(f"모델 평가 중 오류 발생: {e}")

        # 예측 값 계산
        print("예측 계산 중...")
        try:
            Y_pred = best_model.predict(test_gen, steps=test_steps)
            Y_pred_binary = (Y_pred > 0.5).astype(int).reshape(-1, 1)
        except Exception as e:
            print(f"예측 계산 중 오류 발생: {e}")
            Y_pred_binary = np.zeros_like(Y_test_binary).reshape(-1, 1)

        # F1-score 계산
        if len(np.unique(Y_test_binary)) > 1:
            f1 = f1_score(Y_test_binary, Y_pred_binary, average="binary")
            print(f"F1-score: {f1}")
        else:
            print("F1-score 계산 불가: 테스트 세트에 하나의 클래스만 존재합니다.")

        # 분류 리포트 출력
        if len(np.unique(Y_test_binary)) > 1:
            report = classification_report(
                Y_test_binary,
                Y_pred_binary,
                target_names=["Not " + instrument, instrument],
            )
            print(report)
        else:
            print("분류 리포트 출력 불가: 테스트 세트에 하나의 클래스만 존재합니다.")

        # 모델 저장 (이미 ModelCheckpoint로 저장됨)
        print(f"모델 저장 완료: best_model_{sanitized_instrument}.h5")

        # ----------------------------- 세션 및 메모리 정리 -----------------------------
        # 현재 모델과 데이터 제너레이터를 메모리에서 삭제
        try:
            del best_model
            del train_gen
            del val_gen
            del test_gen
        except:
            pass
        gc.collect()
        tf.keras.backend.clear_session()

    print("\n모든 악기별 모델 학습 및 저장 완료.")

    # 7. 전체 모델 평가 (선택 사항)
    print("\n모든 악기별 모델을 사용하여 테스트 세트에서 악기 식별 중...")
    predictions = {}
    for idx, instrument in enumerate(all_instruments):
        sanitized_instrument = sanitize_name(instrument)
        print(f"예측 중: {instrument}")
        # 모델 로드 (.h5 형식)
        model_path = os.path.join(
            models_output_dir, f"best_model_{sanitized_instrument}.h5"
        )
        if not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}")
            continue
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"모델 로드 중 오류 발생 ({instrument}): {e}")
            continue

        # 예측
        try:
            Y_pred = model.predict(test_gen, steps=test_steps)
            Y_pred_binary = (Y_pred > 0.5).astype(int).reshape(-1, 1)
            predictions[instrument] = Y_pred_binary.flatten()
        except Exception as e:
            print(f"모델 예측 중 오류 발생 ({instrument}): {e}")
            predictions[instrument] = np.zeros(len(Y_test_binary), dtype=int)

        # 메모리 정리
        try:
            del model
        except:
            pass
        gc.collect()
        tf.keras.backend.clear_session()

    print("악기 식별 완료.")

    # 8. 사용자 평가 준비 (생략)
    print("사용자 평가를 위한 예측 결과를 저장합니다.")
    # 예측 결과를 저장할 배열 생성
    predictions_list = [
        predictions[instrument]
        for instrument in all_instruments
        if instrument in predictions
    ]
    predictions_array = np.array(predictions_list).T  # shape=(samples, instruments)
    np.save(os.path.join(data_output_dir, "predictions.npy"), predictions_array)
    print("예측 결과 저장 완료: predictions.npy")
