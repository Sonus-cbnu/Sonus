# ----------------------------- 라이브러리 임포트 -----------------------------
import os
import json
import numpy as np
import librosa
from tqdm import tqdm
import h5py

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
import re
import gc
from typing import List, Tuple, Dict, Any

# ----------------------------- 설정 -----------------------------
# 일반 설정
BATCH_SIZE = 256  # 배치 크기 설정
EPOCHS = 20  # 에포크 수 설정
LEARNING_RATE = 1e-4  # 학습률 설정

# 데이터 관련 설정
data_output_dir = "./data"  # 데이터 출력 디렉토리
metadata_file = os.path.join(data_output_dir, "metadata.json")  # 메타데이터 파일 경로
cache_dir = "./data"  # 캐시 디렉토리 설정
os.makedirs(cache_dir, exist_ok=True)
hdf5_file = os.path.join(cache_dir, "preprocessed_data.h5")  # 전처리된 데이터 저장 경로
n_mfcc = 40  # MFCC 계수의 수
max_len = 174  # MFCC 벡터의 최대 길이

# 모델 및 로그 디렉토리 설정
models_output_dir = "./src/models"  # 모델 저장 디렉토리
os.makedirs(models_output_dir, exist_ok=True)
logs_dir = "./src/logs"  # 로그 디렉토리
os.makedirs(logs_dir, exist_ok=True)

# 하이퍼파라미터 튜닝 설정
APPLY_HYPERPARAMETER_TUNING = False  # 하이퍼파라미터 튜닝 적용 여부

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


def extract_mfcc(file_path: str, n_mfcc: int = 40, max_len: int = 174) -> np.ndarray:
    """
    오디오 파일에서 MFCC 특징을 추출합니다.

    Args:
        file_path (str): 오디오 파일의 경로.
        n_mfcc (int): 추출할 MFCC 계수의 수.
        max_len (int): MFCC 벡터의 최대 길이.

    Returns:
        np.ndarray: 고정된 크기의 MFCC 배열.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"MFCC 추출 실패: {file_path}, 에러: {e}")
        return np.zeros((n_mfcc, max_len))


def preprocess_and_save_data(
    metadata: List[Dict[str, Any]], data_dir: str, hdf5_path: str
) -> None:
    """
    데이터 전처리 및 HDF5 파일로 저장합니다.

    Args:
        metadata (List[Dict[str, Any]]): 메타데이터 리스트.
        data_dir (str): 오디오 데이터가 저장된 디렉토리.
        hdf5_path (str): 저장할 HDF5 파일의 경로.
    """
    total_samples = len(metadata)
    with h5py.File(hdf5_path, "w") as h5f:
        X_ds = h5f.create_dataset(
            "X", shape=(total_samples, n_mfcc, max_len), dtype=np.float32
        )
        Y_list = []
        for idx, data in enumerate(tqdm(metadata, desc="전처리 중")):
            sample_path = os.path.join(
                data_dir, data["relative_path"], data["sample_name"]
            )
            mfcc = extract_mfcc(sample_path, n_mfcc, max_len)
            X_ds[idx] = mfcc
            instruments = data["instruments"]
            Y_list.append(instruments)
        Y_encoded_strings = [json.dumps(instr_list) for instr_list in Y_list]
        dt = h5py.string_dtype(encoding="utf-8")
        Y_ds = h5f.create_dataset(
            "Y", data=np.array(Y_encoded_strings, dtype=object), dtype=dt
        )


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


def augment_batch(batch_X: torch.Tensor) -> torch.Tensor:
    """
    데이터 증강을 수행합니다.

    Args:
        batch_X (torch.Tensor): 배치의 MFCC 데이터. (channels, time, frequency)

    Returns:
        torch.Tensor: 증강된 배치의 MFCC 데이터.
    """
    # 데이터 증강 예시 (필요에 따라 수정)
    scale = random.uniform(0.8, 1.2)
    batch_X = batch_X * scale

    noise = torch.randn_like(batch_X) * 0.05
    batch_X = batch_X + noise

    # 주파수 마스킹
    freq_masking = random.randint(0, n_mfcc // 10)
    if freq_masking > 0:
        f0 = random.randint(0, max(n_mfcc - freq_masking - 1, 0))
        batch_X[:, :, f0 : f0 + freq_masking] = 0

    # 시간 마스킹
    time_masking = random.randint(0, max_len // 10)
    if time_masking > 0:
        t0 = random.randint(0, max(max_len - time_masking - 1, 0))
        batch_X[:, t0 : t0 + time_masking, :] = 0

    return batch_X


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


# ----------------------------- 모델 학습 및 평가 함수 정의 -----------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
    device: torch.device,
    sanitized_instrument: str,
) -> None:
    """
    모델을 훈련합니다.

    Args:
        model (nn.Module): 학습할 모델.
        train_loader (DataLoader): 훈련 데이터 로더.
        val_loader (DataLoader): 검증 데이터 로더.
        criterion: 손실 함수.
        optimizer: 옵티마이저.
        num_epochs (int): 에포크 수.
        device (torch.device): 장치 (CPU 또는 GPU).
        sanitized_instrument (str): 악기 이름 (파일명에 사용).
    """
    best_val_loss = float("inf")
    patience = 10
    trigger_times = 0

    print("모델 학습 시작...\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0

        # 훈련 단계
        train_bar = tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] 훈련 중", leave=False
        )
        for batch_X, batch_Y in train_bar:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            outputs = outputs.squeeze()
            loss = criterion(outputs, batch_Y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            preds = (outputs > 0.5).float()
            total_correct += (preds == batch_Y).sum().item()

            # 배치별 손실 업데이트
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = total_correct / len(train_loader.dataset)

        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] 검증 중", leave=False
            )
            for batch_X, batch_Y in val_bar:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                outputs = model(batch_X)
                outputs = outputs.squeeze()
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == batch_Y).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_Y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / len(val_loader.dataset)

        # F1-score, Precision, Recall 계산
        if len(np.unique(all_labels)) > 1:
            epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
            epoch_precision = precision_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )
            epoch_recall = recall_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )
        else:
            epoch_f1 = 0.0
            epoch_precision = 0.0
            epoch_recall = 0.0

        # 에포크별 성능 지표 출력
        tqdm.write(
            f"Epoch [{epoch+1}/{num_epochs}] 완료 - "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, "
            f"Val Precision: {epoch_precision:.4f}, Val Recall: {epoch_recall:.4f}, Val F1-Score: {epoch_f1:.4f}"
        )

        # Early Stopping 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            save_model(
                model,
                os.path.join(
                    models_output_dir, f"best_model_{sanitized_instrument}.pt"
                ),
            )
            tqdm.write(f"--> 최고 검증 손실 기록: {best_val_loss:.4f} 저장됨.\n")
        else:
            trigger_times += 1
            tqdm.write(f"--> EarlyStopping 트리거 횟수: {trigger_times}/{patience}\n")
            if trigger_times >= patience:
                tqdm.write("Early stopping 발생. 학습 중단.\n")
                break

        # 학습률 조정
        if trigger_times > 0 and trigger_times % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
            tqdm.write(f"--> 학습률이 {optimizer.param_groups[0]['lr']}로 감소됨.\n")

    tqdm.write(f"모델 학습 완료. 최고 검증 손실: {best_val_loss:.4f}\n")


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, criterion, device: torch.device
) -> Tuple[float, float, List[float], List[float]]:
    """
    모델을 평가합니다.

    Args:
        model (nn.Module): 평가할 모델.
        test_loader (DataLoader): 테스트 데이터 로더.
        criterion: 손실 함수.
        device (torch.device): 장치 (CPU 또는 GPU).

    Returns:
        Tuple[float, float, List[float], List[float]]: 테스트 손실, 정확도, 예측 값 리스트, 실제 값 리스트.
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    all_preds = []
    all_labels = []

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


def save_model(model: nn.Module, path: str) -> None:
    """
    모델을 지정된 경로에 저장합니다.

    Args:
        model (nn.Module): 저장할 모델.
        path (str): 저장할 파일 경로.
    """
    torch.save(model.state_dict(), path)


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
    return model


# ----------------------------- 하이퍼파라미터 튜닝 함수 정의 -----------------------------
if APPLY_HYPERPARAMETER_TUNING:
    import optuna

    def objective(
        trial: optuna.Trial,
        X_train: np.ndarray,
        Y_train_binary: np.ndarray,
        X_val: np.ndarray,
        Y_val_binary: np.ndarray,
        input_shape: Tuple[int, int, int],
        device: torch.device,
        sanitized_instrument: str,
    ) -> float:
        """
        Optuna를 사용한 하이퍼파라미터 최적화 목적 함수

        Args:
            trial (optuna.Trial): Optuna trial 객체
            X_train (np.ndarray): 훈련 데이터 특징
            Y_train_binary (np.ndarray): 훈련 데이터 레이블
            X_val (np.ndarray): 검증 데이터 특징
            Y_val_binary (np.ndarray): 검증 데이터 레이블
            input_shape (Tuple[int, int, int]): 모델 입력 형태
            device (torch.device): 장치 (CPU 또는 GPU)
            sanitized_instrument (str): 악기 이름 (파일명에 사용)

        Returns:
            float: 검증 손실 값
        """
        # 하이퍼파라미터 범위 설정
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.3, 0.7)
        num_epochs = trial.suggest_int("num_epochs", 10, 30)

        # 모델 정의
        num_classes = 1
        model = CNNModel(input_shape, num_classes, dropout_rate=dropout_rate).to(device)

        # 옵티마이저와 손실 함수 설정
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        # 데이터셋 및 데이터로더 생성
        train_dataset = MFCCDataset(X_train, Y_train_binary, augment=True)
        val_dataset = MFCCDataset(X_val, Y_val_binary, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # 모델 훈련
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            sanitized_instrument=sanitized_instrument,
        )

        # 검증 손실 계산
        avg_val_loss, _, _, _ = evaluate_model(
            model=model, test_loader=val_loader, criterion=criterion, device=device
        )

        return avg_val_loss


# ----------------------------- 메인 실행 부분 -----------------------------
if __name__ == "__main__":
    # 메타데이터 로드
    print("메타데이터 로드 중...")
    metadata = load_metadata(metadata_file)
    print(f"메타데이터 로드 완료: {len(metadata)} 샘플")

    # 데이터 전처리 및 저장
    if not os.path.exists(hdf5_file):
        print("데이터 전처리 시작...")
        preprocess_and_save_data(metadata, data_output_dir, hdf5_file)
        print("데이터 전처리 및 저장 완료.")
    else:
        print("전처리된 데이터 파일이 존재합니다. 로드합니다.")

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

    # 개별 악기별 모델 구축 및 학습
    print("개별 악기별 모델 구축 및 학습 시작...")

    for idx, instrument in enumerate(all_instruments):
        sanitized_instrument = sanitize_name(instrument)
        print(f"\n=== 악기: {instrument} ({idx + 1}/{len(all_instruments)}) ===")

        # 이진 레이블 생성
        Y_train_binary = Y_train[:, idx]
        Y_val_binary = Y_val[:, idx]
        Y_test_binary = Y_test[:, idx]

        # 클래스 가중치 설정
        classes = np.unique(Y_train_binary)
        if len(classes) > 1:
            class_weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=Y_train_binary
            )
            class_weight_dict = {
                cls: weight for cls, weight in zip(classes, class_weights)
            }
            print(f"클래스 가중치 계산 완료: {class_weight_dict}")
        else:
            class_weight_dict = None
            print("클래스가 하나이므로 클래스 가중치를 적용하지 않습니다.")

        # 하이퍼파라미터 튜닝 적용 여부에 따른 처리
        if APPLY_HYPERPARAMETER_TUNING:
            print("하이퍼파라미터 튜닝을 수행합니다...")
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(
                    trial,
                    X_train,
                    Y_train_binary,
                    X_val,
                    Y_val_binary,
                    input_shape=(1, max_len, n_mfcc),
                    device=device,
                    sanitized_instrument=sanitized_instrument,
                ),
                n_trials=20,
            )

            # 최적의 하이퍼파라미터로 모델 재훈련
            best_params = study.best_params
            print("최적의 하이퍼파라미터:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")

            learning_rate = best_params["learning_rate"]
            dropout_rate = best_params["dropout_rate"]
            num_epochs = best_params["num_epochs"]
        else:
            learning_rate = LEARNING_RATE
            dropout_rate = 0.5
            num_epochs = EPOCHS

        # 모델 구축
        input_shape = (1, max_len, n_mfcc)
        num_classes = 1
        model = CNNModel(input_shape, num_classes, dropout_rate=dropout_rate).to(device)

        # 옵티마이저와 손실 함수 설정
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        # 데이터셋 및 데이터로더 생성
        train_dataset = MFCCDataset(X_train, Y_train_binary, augment=True)
        val_dataset = MFCCDataset(X_val, Y_val_binary, augment=False)
        test_dataset = MFCCDataset(X_test, Y_test_binary, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # 모델 훈련
        print("모델 훈련 시작...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            sanitized_instrument=sanitized_instrument,
        )
        print("모델 훈련 완료.")

        # 최상의 모델 로드
        try:
            model = load_model(
                model,
                os.path.join(
                    models_output_dir, f"best_model_{sanitized_instrument}.pt"
                ),
                device,
            )
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            continue

        # 모델 평가
        print("모델 평가 중...")
        avg_test_loss, avg_test_acc, all_preds, all_labels = evaluate_model(
            model=model, test_loader=test_loader, criterion=criterion, device=device
        )
        print(f"테스트 손실: {avg_test_loss}")
        print(f"테스트 정확도: {avg_test_acc}")

        # F1-score 계산
        if len(np.unique(Y_test_binary)) > 1:
            f1 = f1_score(all_labels, all_preds, average="binary")
            print(f"F1-score: {f1}")
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

        # 모델 저장
        print(f"모델 저장 완료: best_model_{sanitized_instrument}.pt")

        # 메모리 정리
        try:
            del model
            del train_loader
            del val_loader
            del test_loader
        except:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n모든 악기별 모델 학습 및 저장 완료.")

    # 전체 모델 평가
    print("\n모든 악기별 모델을 사용하여 테스트 세트에서 악기 식별 중...")
    predictions = {}
    for idx, instrument in enumerate(all_instruments):
        sanitized_instrument = sanitize_name(instrument)
        print(f"예측 중: {instrument}")
        # 모델 로드
        model_path = os.path.join(
            models_output_dir, f"best_model_{sanitized_instrument}.pt"
        )
        if not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}")
            continue
        try:
            model = CNNModel(input_shape, num_classes).to(device)
            model = load_model(model, model_path, device)
            model.eval()
        except Exception as e:
            print(f"모델 로드 중 오류 발생 ({instrument}): {e}")
            continue

        # 예측
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                outputs = outputs.squeeze()
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
            predictions[instrument] = np.array(all_preds)
        # 메모리 정리
        try:
            del model
        except:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("악기 식별 완료.")

    # 예측 결과 저장
    print("사용자 평가를 위한 예측 결과를 저장합니다.")
    predictions_list = [
        predictions[instrument]
        for instrument in all_instruments
        if instrument in predictions
    ]
    predictions_array = np.array(predictions_list).T  # shape=(samples, instruments)
    np.save(os.path.join(data_output_dir, "predictions.npy"), predictions_array)
    print("예측 결과 저장 완료: predictions.npy")
