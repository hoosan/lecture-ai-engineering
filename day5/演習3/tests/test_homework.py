import os
import time
import pickle
import pytest
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/titanic_model.pkl")


@pytest.fixture(scope="module")
def model():
    """保存済みモデルをロード"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが見つからないためテストをスキップします")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def test_data(sample_data):
    """推論用データ (特徴量とラベル) を準備"""
    X_test = sample_data.drop("Survived", axis=1)
    y_test = sample_data["Survived"].astype(int)
    return X_test, y_test


def test_inference_accuracy(model, test_data):
    """推論精度が 0.80 以上か検証"""
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.80, f"推論精度が基準 (0.80) を下回っています: {acc:.3f}"


def test_inference_latency(model, test_data):
    """バッチ推論の実行時間が 1 秒未満か検証"""
    X_test, _ = test_data
    start = time.time()
    model.predict(X_test)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"推論時間が 1 秒を超えています: {elapsed:.3f} 秒"
