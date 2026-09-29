import os
import sys
import types
import unittest
from unittest.mock import patch

from training_device import device_status, select_training_device


class TrainingDeviceTests(unittest.TestCase):
    def fake_torch(self, available):
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: available,
                get_device_name=lambda _: "Test GPU",
            ),
            version=types.SimpleNamespace(cuda="13.0"),
        )

    def test_auto_uses_cpu_without_cuda(self):
        with patch.dict(os.environ, {"TRAIN_DEVICE": "auto"}), patch.dict(
            sys.modules, {"torch": self.fake_torch(False)}
        ):
            self.assertEqual(select_training_device(), "cpu")

    def test_auto_uses_gpu_when_available(self):
        with patch.dict(os.environ, {"TRAIN_DEVICE": "auto"}), patch.dict(
            sys.modules, {"torch": self.fake_torch(True)}
        ):
            self.assertEqual(select_training_device(), "cuda")
            self.assertEqual(device_status()["gpu_name"], "Test GPU")

    def test_explicit_cuda_fails_instead_of_falling_back(self):
        with patch.dict(os.environ, {"TRAIN_DEVICE": "cuda"}), patch.dict(
            sys.modules, {"torch": self.fake_torch(False)}
        ):
            with self.assertRaisesRegex(RuntimeError, "PyTorch cannot access CUDA"):
                select_training_device()

    def test_explicit_cpu_overrides_available_gpu(self):
        with patch.dict(os.environ, {"TRAIN_DEVICE": "cpu"}), patch.dict(
            sys.modules, {"torch": self.fake_torch(True)}
        ):
            self.assertEqual(select_training_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
