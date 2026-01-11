
from ComReader import UARTReader
from Analyzer import *
from Model import load_model_pipeline, predict_new_sample

import numpy as np

import time
from typing import List, Literal
from pathlib import Path
import yaml
import logging

# 配置日志
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, path: Path):
        self._serial_cache = None
        with open(path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)

    @property
    def action_channel_map(self):
        return self._cfg["action_channel_map"]

    @property
    def serial(self):
        if self._serial_cache is not None:
            return self._serial_cache

        raw = self._cfg["serial"]

        serial_cfg = {
            "port": raw["port"],
            "baudrate": raw["baudrate"],
            "data_num": raw["data_num"],
            "header": bytes.fromhex(raw["header"].removeprefix('0x')),
            "footer": bytes.fromhex(raw["footer"].removeprefix('0x')),
            "byteorder": raw['byteorder']
        }

        self._serial_cache = serial_cfg
        return serial_cfg

    @property
    def csv_file(self):
        return self._cfg['data']['csv']['file']

    @property
    def csv_title(self):
        return self._cfg['data']['csv']['title']
    
    @property
    def sample_rate(self):
        return self._cfg['data']['csv']['sample_rate']
    
    @property
    def model_paths(self):
        return self._cfg["model"]["paths"]


class Manager:
    def __init__(self, config_path:Path):
        self.config = Config(config_path)

        self.fs = self.config.sample_rate  # 采样率
        self.csv_file = self.config.csv_file

        self.uart_reader = UARTReader(
            **self.config.serial,
            csv_file=self.csv_file,
            csv_title=self.config.csv_title,
        )
        self.analyzer = EMGAnalyzer()

    def open_com_reader(self):
        self.uart_reader.start()

    def close_com_reader(self):
        self.uart_reader.stop()

    def get_data(self) -> np.ndarray:
        """
        用于获取数据的抽象层
        计划将原始数据格式重构为 bin + meta 的架构
        Returns:
            np.ndarray: 二维数组，每个元素为一个数据点
        """
        data = np.loadtxt(self.csv_file, delimiter=',', skiprows=1)[:, 1:]
        return data

    def analyze_emg(self, affected: Literal['L', 'R', 'N'], action:str) -> np.ndarray:
        """
        
        Arg:
            Affected: 患侧是左还是右 或者没有患侧
        Returns:
            np.ndarray: size (5,), order [rms, mav, mf, rmsf, rvf]
            每个特征的不对称性指标
        """
        left_ch, right_ch = self.config.action_channel_map[action]
        data = self.get_data()

        signal = (
            EMGSignal(
                "Left",
                True if affected == "L" else False,
                action,
                data[:, left_ch],
                self.fs,
            ),
            EMGSignal(
                "Right",
                True if affected == "R" else False,
                action,
                data[:, right_ch],
                self.fs,
            ),
        )

        return self.analyzer.analyze_symmetry_emg(signal)

    def show_model(self, sample: np.ndarray)->None:
        print("\n=== Model Loading Demo ===")     

        model_paths = self.config.model_paths

        pipeline = load_model_pipeline(
            model_paths["model"],
            model_paths["scaler"],
            model_paths["config"],
        )

        print("Model loaded successfully!")
        print(f"Model type: {pipeline['config']['model_type']}")
        print(f"Features: {pipeline['config']['feature_columns']}")

        # 演示对新样本的预测
        prediction = predict_new_sample(pipeline, sample)
        print(f"Sample prediction: {prediction['prediction']}")
        if prediction['probabilities'] is not None:
            print(f"Prediction probabilities: {prediction['probabilities']}")


if __name__ == '__main__':
    # 用法示例
    manager = Manager(config_path='config/config.yaml')
    try:
        manager.open_com_reader()
        while not manager.uart_reader.isStop():
            time.sleep(0.1)
        manager.close_com_reader()

        data = manager.get_data()
        print(f'Got data size: {data.shape}')
        feature_vector = manager.analyze_emg('L', 'by')
        print(f'Got feature vector: {feature_vector}')

        manager.show_model(feature_vector)

    except KeyboardInterrupt:
        manager.close_com_reader()
