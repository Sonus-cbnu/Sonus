import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# HDF5 파일이 저장된 디렉토리 설정
h5_dir = 'processed_clips/'
batch_size = 32
epochs = 30

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# 악기 MIDI 번호에서 인덱스 매핑을 정의
instrument_mapping = {
    1: 0,   # Grand Piano
    41: 1,  # Violin
    42: 2,  # Viola
    43: 3,  # Cello
    61: 4,  # Horn
    71: 5,  # Bassoon
    72: 6,  # Clarinet
    7:  7,  # Harpsichord
    44: 8,  # Contrabass
    69: 9,  # Oboe
    74: 10  # Flute
}

# 데이터셋 로드 함수
def load_h5_file(h5_path):
    with h5py.File(h5_path, 'r') as f:
        clip = np.array(f['clip'])
        instrument = np.array(f['instrument']).flatten()
        note = np.array(f['note']).flatten() 
        
        instrument_class = np.array([instrument_mapping.get(instr, -1) for instr in instrument])
        if -1 in instrument_class:
            raise ValueError("Unrecognized MIDI instrument number found.")
    
    return clip, instrument_class, note

# HDF5 파일들을 배치로 불러오는 데이터 제너레이터
class H5DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_dir, batch_size=32, shuffle=True):
        self.h5_files = [os.path.join(h5_dir, fname) for fname in os.listdir(h5_dir) if fname.endswith('.h5')]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.h5_files) // self.batch_size
    
    def __getitem__(self, index):
        batch_files = self.h5_files[index * self.batch_size:(index + 1) * self.batch_size]
        clips, instruments, notes = [], [], []
        
        for file in batch_files:
            clip, instrument, note = load_h5_file(file)
            clips.append(clip)
            instruments.append(instrument)
            notes.append(note)
        
        clips = np.array(clips)
        instruments = np.array(instruments)
        notes = np.array(notes)

        return [clips, instruments], notes

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.h5_files)

# 멀티 브랜치 CNN 모델 생성
def build_model(input_shape=(256, 46, 1), num_classes=11):
    input_layer = layers.Input(shape=input_shape)

    # 첫 번째 합성곱 레이어
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 두 번째 합성곱 레이어
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 세 번째 합성곱 레이어
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Flatten
    flatten_spectrogram = layers.Flatten()(x)
    
    # 노트 정보 입력 레이어 (음높이)
    note_input = layers.Input(shape=(1,), name="note_input")

    # Fully Connected Layer
    combined = layers.Concatenate()([flatten_spectrogram, note_input])
    fc1 = layers.Dense(512, activation='relu')(combined)
    fc1 = layers.Dropout(0.5)(fc1)
    fc2 = layers.Dense(256, activation='relu')(fc1)
    output_layer = layers.Dense(num_classes, activation='softmax')(fc2)  
    
    model = models.Model(inputs=[input_layer, note_input], outputs=output_layer)
    return model

# 모델 학습 및 평가
def train_model(h5_dir, batch_size, epochs):
    train_gen = H5DataGenerator(h5_dir, batch_size=batch_size)
    
    model = build_model(input_shape=(256, 46, 1), num_classes=11)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',  # 정수 레이블에 맞는 손실 함수
                  metrics=['accuracy'])

    lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    
    history = model.fit(train_gen, epochs=epochs, callbacks=[lr_reduce, early_stop], validation_data=train_gen)

    return model, history

# 모델 학습 실행
model, history = train_model(h5_dir, batch_size, epochs)
