import os
import pickle
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

# Load env variables
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "ml", "model"
)

NUMBERS_MODEL_PATH = os.getenv(
    "MODEL_SAVE_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "sign_speak_lstm.keras")
)
NUMBERS_LABEL_ENCODER_PATH = os.getenv(
    "LABEL_ENCODER_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "label_encoder.pkl")
)

WORDS_MODEL_PATH = os.getenv(
    "WORDS_MODEL_SAVE_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "sign_speak_words_lstm.keras")
)
WORDS_LABEL_ENCODER_PATH = os.getenv(
    "WORDS_LABEL_ENCODER_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "label_encoder_words.pkl")
)

MODE_CONFIGS: Dict[str, Dict[str, str]] = {
    "numbers": {
        "name": "Numbers (Digits 0-9)",
        "model_path": NUMBERS_MODEL_PATH,
        "label_encoder_path": NUMBERS_LABEL_ENCODER_PATH,
    },
    "words": {
        "name": "Words (Alphabet)",
        "model_path": WORDS_MODEL_PATH,
        "label_encoder_path": WORDS_LABEL_ENCODER_PATH,
    },
}

DEFAULT_MODE = "numbers"


class ModelLoaderService:
    """Thread-safe Singleton Service that can load multiple classification models,
    one per prediction mode (e.g. 'numbers' for digits, 'words' for the alphabet).

    The default 'numbers' model is loaded eagerly at startup; other modes are
    loaded lazily on first use.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelLoaderService, cls).__new__(cls)
                cls._instance._initialized = False
                cls._instance._models: Dict[str, "tf.keras.Model"] = {}
                cls._instance._label_encoders: Dict[str, object] = {}
            return cls._instance

    def initialize(self):
        """Loads the default model so the API is ready to serve immediately.
        This is thread-safe and runs once.
        """
        with self._lock:
            if self._initialized:
                return

            self._load_model(DEFAULT_MODE)
            self._initialized = True
            print(f"[ModelLoader] '{DEFAULT_MODE}' model initialized successfully!")

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    @staticmethod
    def available_modes() -> list:
        return list(MODE_CONFIGS.keys())

    def mode_name(self, mode: str) -> str:
        return MODE_CONFIGS.get(mode, {}).get("name", mode)

    def is_mode_available(self, mode: str) -> bool:
        config = MODE_CONFIGS.get(mode)
        if config is None:
            return False
        return os.path.exists(config["model_path"]) and os.path.exists(config["label_encoder_path"])

    def _load_model(self, mode: str) -> None:
        with self._lock:
            if mode in self._models:
                return

            config = MODE_CONFIGS.get(mode)
            if config is None:
                raise ValueError(
                    f"Unknown prediction mode: '{mode}'. Supported modes: {list(MODE_CONFIGS)}"
                )

            model_path = config["model_path"]
            label_encoder_path = config["label_encoder_path"]

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file for '{mode}' mode not found at: {model_path}"
                )
            if not os.path.exists(label_encoder_path):
                raise FileNotFoundError(
                    f"Label encoder for '{mode}' mode not found at: {label_encoder_path}"
                )

            print(f"[ModelLoader] Loading '{mode}' model from: {model_path}")
            self._models[mode] = tf.keras.models.load_model(model_path)

            print(f"[ModelLoader] Loading '{mode}' label encoder from: {label_encoder_path}")
            with open(label_encoder_path, "rb") as f:
                self._label_encoders[mode] = pickle.load(f)

    def _get_model(self, mode: str = DEFAULT_MODE) -> Tuple["tf.keras.Model", object]:
        if not self._initialized:
            raise RuntimeError("ModelLoaderService is not initialized. Call initialize() first.")
        mode = mode or DEFAULT_MODE
        if mode not in self._models:
            self._load_model(mode)
        return self._models[mode], self._label_encoders[mode]

    def predict(self, sequence: np.ndarray, mode: str = DEFAULT_MODE) -> np.ndarray:
        """Runs predictions for one ``(frames, features)`` landmark sequence using the model for the given mode."""
        model, _ = self._get_model(mode)

        expected_shape = tuple(model.input_shape[1:])
        if tuple(sequence.shape) != expected_shape:
            raise ValueError(
                f"Model expects a sequence shaped {expected_shape}, got {sequence.shape}."
            )

        with self._lock:
            predictions = model(np.expand_dims(sequence, axis=0), training=False)
            return predictions.numpy()[0]

    def decode_label(self, class_index: int, mode: str = DEFAULT_MODE) -> str:
        """Maps a model class index back to the original label for the given mode."""
        _, label_encoder = self._get_model(mode)
        return str(label_encoder.classes_[class_index])
