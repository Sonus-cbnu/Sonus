# ----------------------------- 라이브러리 임포트 -----------------------------
import os
import json
import numpy as np
import h5py
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import re
import gc
from typing import List, Tuple, Dict, Any

# ----------------------------- 설정 -----------------------------
# 일반 설정
BATCH_SIZE = 256  # 배치 크기 설정

# 데이터 관련 설정
data_output_dir = "./data"  # 데이터 출력 디렉토리
metadata_file = os.path.join(data_output_dir, "metadata.json")  # 메타데이터 파일 경로
cache_dir = "./cache"  # 캐시 디렉토리 설정
os.makedirs(cache_dir, exist_ok=True)
hdf5_file = os.path.join(cache_dir, "preprocessed_data.h5")  # 전처리된 데이터 저장 경로
n_mfcc = 40  # MFCC 계수의 수
max_len = 174  # MFCC 벡터의 최대 길이

# 모델 및 로그 디렉토리 설정
models_output_dir = "./models"  # 모델 저장 디렉토리
os.makedirs(models_output_dir, exist_ok=True)

# ----------------------------- GPU 및 CPU 설정 -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}\n")

# CPU 사용 제한 설정
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
torch.set_num_threads(4)


# ----------------------------- 전처리 함수 정의 -----------------------------
def sanitize_name(name: str) -> str:
    """
    문자열에서 알파벳, 숫자, 언더스코어만 남기고 나머지는 제거합니다.

    Args:
        name (str): 원본 문자열.

    Returns:
        str: 정제된 문자열.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "", name.replace(" ", "_"))


def load_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    """
    메타데이터를 로드합니다.

    Args:
        metadata_path (str): 메타데이터 파일의 경로.

    Returns:
        List[Dict[str, Any]]: 메타데이터 리스트.
    """
    metadata = []
    with open(metadata_path, "r") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def load_preprocessed_data(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    전처리된 데이터를 HDF5 파일에서 로드합니다.

    Args:
        hdf5_path (str): HDF5 파일의 경로.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 특징 배열 X와 레이블 리스트 Y.
    """
    with h5py.File(hdf5_path, "r") as h5f:
        X = h5f["X"][:]
        Y = h5f["Y"][:]
    return X, Y


def encode_labels(Y: np.ndarray) -> Tuple[np.ndarray, MultiLabelBinarizer, List[str]]:
    """
    레이블을 이진 벡터로 인코딩합니다.

    Args:
        Y (np.ndarray): 레이블 리스트 (JSON-encoded strings).

    Returns:
        Tuple[np.ndarray, MultiLabelBinarizer, List[str]]: 인코딩된 레이블 배열, 레이블 변환기, 모든 악기 리스트.
    """
    all_instruments = set()
    parsed_Y = []
    for instruments in Y:
        if isinstance(instruments, bytes):
            instruments = instruments.decode("utf-8")
        if isinstance(instruments, str):
            try:
                instruments = json.loads(instruments)
            except json.JSONDecodeError:
                instruments = instruments.split(",")
        if not isinstance(instruments, list):
            instruments = []
        instruments = [instr.strip() for instr in instruments if instr.strip()]
        all_instruments.update(instruments)
        parsed_Y.append(instruments)
    all_instruments = sorted(list(all_instruments))

    if not all_instruments:
        print("경고: 모든 악기 리스트가 비어 있습니다.")

    mlb = MultiLabelBinarizer(classes=all_instruments)
    Y_encoded = mlb.fit_transform(parsed_Y)
    return Y_encoded, mlb, all_instruments


# ----------------------------- 데이터셋 클래스 정의 -----------------------------
class MFCCDataset(Dataset):
    """
    MFCC 데이터를 위한 PyTorch Dataset 클래스
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, augment: bool = False):
        """
        초기화 함수

        Args:
            X (np.ndarray): 특징 데이터.
            Y (np.ndarray): 레이블 데이터.
            augment (bool): 데이터 증강 여부.
        """
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self) -> int:
        """
        데이터셋의 길이를 반환합니다.

        Returns:
            int: 데이터셋의 길이.
        """
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인덱스에 해당하는 데이터를 반환합니다.

        Args:
            idx (int): 데이터 인덱스.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 특징과 레이블 텐서.
        """
        x = self.X[idx]
        y = self.Y[idx]

        x = np.expand_dims(x, axis=0)
        x = x.transpose(0, 2, 1)
        x = torch.from_numpy(x).float()

        if self.augment:
            x = augment_batch(x)

        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-6)

        y = torch.tensor(y, dtype=torch.float32)

        return x, y


# ----------------------------- 모델 정의 -----------------------------
class CNNModel(nn.Module):
    """
    CNN 모델 정의 클래스
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        dropout_rate: float = 0.5,
        l2_reg: float = 0.001,
    ):
        """
        모델을 초기화합니다.

        Args:
            input_shape (Tuple[int, int, int]): 입력 데이터의 형태.
            num_classes (int): 출력 클래스의 수.
            dropout_rate (float): 드롭아웃 비율.
            l2_reg (float): L2 정규화 계수.
        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding="same")
        self.pool1 = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding="same")
        self.pool2 = nn.MaxPool2d((2, 2))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding="same")
        self.pool3 = nn.MaxPool2d((2, 2))
        self.bn3 = nn.BatchNorm2d(32)

        conv_output_size = self._get_conv_output(input_shape)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(64, num_classes)
        self.l2_reg = l2_reg

    def _get_conv_output(self, shape: Tuple[int, int, int]) -> int:
        """
        합성곱 계층의 출력을 계산합니다.

        Args:
            shape (Tuple[int, int, int]): 입력 데이터의 형태.

        Returns:
            int: 합성곱 계층 출력의 크기.
        """
        bs = 1
        input = torch.zeros(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        합성곱 계층을 통과하는 부분입니다.

        Args:
            x (torch.Tensor): 입력 텐서.

        Returns:
            torch.Tensor: 합성곱 계층의 출력.
        """
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.bn3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파를 정의합니다.

        Args:
            x (torch.Tensor): 입력 텐서.

        Returns:
            torch.Tensor: 모델의 출력.
        """
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def l2_regularization(self) -> torch.Tensor:
        """
        L2 정규화를 계산합니다.

        Returns:
            torch.Tensor: L2 정규화 손실 값.
        """
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return self.l2_reg * l2_norm


# ----------------------------- 모델 평가 함수 정의 -----------------------------
def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float, List[float], List[float]]:
    """
    모델을 평가합니다.

    Args:
        model (nn.Module): 평가할 모델.
        test_loader (DataLoader): 테스트 데이터 로더.
        device (torch.device): 장치 (CPU 또는 GPU).

    Returns:
        Tuple[float, float, List[float], List[float]]: 테스트 손실, 정확도, 예측 값 리스트, 실제 값 리스트.
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    all_preds = []
    all_labels = []

    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            outputs = model(batch_X)
            outputs = outputs.squeeze()
            loss = criterion(outputs, batch_Y)

            test_loss += loss.item() * batch_X.size(0)
            preds = (outputs > 0.5).float()
            test_correct += (preds == batch_Y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_acc = test_correct / len(test_loader.dataset)

    return avg_test_loss, avg_test_acc, all_preds, all_labels


def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """
    모델을 지정된 경로에서 로드합니다.

    Args:
        model (nn.Module): 로드할 모델 구조.
        path (str): 모델 파일 경로.
        device (torch.device): 장치 (CPU 또는 GPU).

    Returns:
        nn.Module: 로드된 모델.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ----------------------------- 메인 실행 부분 -----------------------------
if __name__ == "__main__":
    # 메타데이터 로드
    print("메타데이터 로드 중...")
    metadata = load_metadata(metadata_file)
    print(f"메타데이터 로드 완료: {len(metadata)} 샘플")

    # 전처리된 데이터 로드
    print("전처리된 데이터 로드 중...")
    X, Y = load_preprocessed_data(hdf5_file)
    print(f"전처리된 데이터 로드 완료: X shape={X.shape}, Y shape={Y.shape}")

    # 레이블 인코딩
    print("레이블 인코딩 중...")
    Y_encoded, mlb, all_instruments = encode_labels(Y)
    print(f"레이블 인코딩 완료: {len(all_instruments)} 종류의 악기")

    if len(all_instruments) == 0:
        raise ValueError("레이블 인코딩 실패: 악기 리스트가 비어 있습니다.")

    # 데이터셋 분할
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

    # 개별 악기별 모델 평가
    print("개별 악기별 모델 평가 시작...")

    predictions = {}
    for idx, instrument in enumerate(all_instruments):
        sanitized_instrument = sanitize_name(instrument)
        print(f"\n=== 악기: {instrument} ({idx + 1}/{len(all_instruments)}) ===")

        # 모델 경로 설정
        model_path = os.path.join(
            models_output_dir, f"best_model_{sanitized_instrument}.pt"
        )

        # 모델 파일 존재 여부 확인
        if not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}. 건너뜁니다.")
            continue

        # 모델 초기화 및 로드
        input_shape = (1, n_mfcc, max_len)
        num_classes = 1
        model = CNNModel(input_shape, num_classes).to(device)
        try:
            model = load_model(model, model_path, device)
            print("모델 로드 완료.")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}. 건너뜁니다.")
            continue

        # 이진 레이블 추출
        Y_test_binary = Y_test[:, idx]

        # 테스트 데이터셋 및 데이터로더 생성 (이진 레이블 사용)
        test_dataset = MFCCDataset(X_test, Y_test_binary, augment=False)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # 모델 평가
        print("모델 평가 중...")
        avg_test_loss, avg_test_acc, all_preds, all_labels = evaluate_model(
            model=model, test_loader=test_loader, device=device
        )
        print(f"테스트 손실: {avg_test_loss:.4f}")
        print(f"테스트 정확도: {avg_test_acc:.4f}")

        # F1-score 계산
        if len(np.unique(Y_test_binary)) > 1:
            f1 = f1_score(all_labels, all_preds, average="binary")
            print(f"F1-score: {f1:.4f}")
        else:
            print("F1-score 계산 불가: 테스트 세트에 하나의 클래스만 존재합니다.")

        # 분류 리포트 출력
        if len(np.unique(Y_test_binary)) > 1:
            report = classification_report(
                all_labels,
                all_preds,
                target_names=["Not " + instrument, instrument],
            )
            print(report)
        else:
            print("분류 리포트 출력 불가: 테스트 세트에 하나의 클래스만 존재합니다.")

        # 예측 저장
        predictions[instrument] = np.array(all_preds)

        # 메모리 정리
        try:
            del model
            del test_loader
        except:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n모든 악기별 모델 평가 완료.")

    # 예측 결과 저장
    print("\n사용자 평가를 위한 예측 결과를 저장합니다.")
    predictions_list = [
        predictions[instrument]
        for instrument in all_instruments
        if instrument in predictions
    ]
    if predictions_list:
        predictions_array = np.stack(
            predictions_list, axis=1
        )  # shape=(samples, instruments)
        np.save(os.path.join(data_output_dir, "predictions.npy"), predictions_array)
        print("예측 결과 저장 완료: predictions.npy")
    else:
        print("저장할 예측 결과가 없습니다.")
