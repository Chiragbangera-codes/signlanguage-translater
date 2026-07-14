import os
import pickle
import threading

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
MODEL_SAVE_PATH = os.getenv(
    "MODEL_SAVE_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "sign_speak_model.keras")
)
LABEL_ENCODER_PATH = os.getenv(
    "LABEL_ENCODER_PATH",
    os.path.join(DEFAULT_MODEL_DIR, "label_encoder.pkl")
)

class ModelLoaderService:
    """Thread-safe Singleton Service to load the Keras model and label encoder,

    and run predictions.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelLoaderService, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        """Loads the model and label encoder from configured paths.
        This is thread-safe and runs once.
        """
        with self._lock:
            if self._initialized:
                return

            print(f"[ModelLoader] Loading model from: {MODEL_SAVE_PATH}")
            if not os.path.exists(MODEL_SAVE_PATH):
                raise FileNotFoundError(f"Model file not found at: {MODEL_SAVE_PATH}")
            self.model = tf.keras.models.load_model(MODEL_SAVE_PATH)

            print(f"[ModelLoader] Loading label encoder from: {LABEL_ENCODER_PATH}")
            if not os.path.exists(LABEL_ENCODER_PATH):
                raise FileNotFoundError(f"Label encoder file not found at: {LABEL_ENCODER_PATH}")
            with open(LABEL_ENCODER_PATH, "rb") as f:
                self.label_encoder = pickle.load(f)

            self._initialized = True
            print("[ModelLoader] Initialization successful!")

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    def predict(self, features_127: np.ndarray) -> np.ndarray:
        """Runs predictions on the input features using a thread-safe lock."""
        if not self._initialized:
            raise RuntimeError("ModelLoaderService is not initialized. Call initialize() first.")

        with self._lock:
            input_arr = features_127.reshape(1, -1)
            # Run inference
            predictions = self.model(input_arr, training=False)
            return predictions.numpy()[0]

    def decode_label(self, class_index: int) -> str:
        """Maps numerical class index back to original character (A-Z)."""
        # If it is index 0-25, decode using classes mapping
        # classes are strings like '0', '1', etc.
        class_str = self.label_encoder.classes_[class_index]
        # Convert index '0' -> 'A'
        return chr(65 + int(class_str))
