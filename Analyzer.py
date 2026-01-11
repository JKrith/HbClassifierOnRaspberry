import numpy as np
from scipy.signal import butter, filtfilt, welch

from typing import Literal
from dataclasses import dataclass
import logging

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class EMGSignal:
    side: Literal['Left', 'Right']
    affected: bool
    action: Literal['ym',"by","lcx","mz","dz"]
    data: np.ndarray
    fs: float

class EMGAnalyzer:
    """
    EMG信号分析类，用于处理和分析肌电信号数据
    """
    
    def __init__(self, fs=1000, lowcut=20, highcut=450, filter_order=4):
        """
        初始化EMG分析器
        
        Args:

        """
        # 信号处理参数
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order

    
    def bandpass_filter(self, data, lowcut=None, highcut=None, fs=None, order=None):
        """
        带通滤波函数
        
        Args:
            data: 输入信号数据
            lowcut: 高通截止频率，默认为类参数
            highcut: 低通截止频率，默认为类参数
            fs: 采样频率，默认为类参数
            order: 滤波器阶数，默认为类参数
            
        Returns:
            滤波后的信号
        """
        lowcut = lowcut or self.lowcut
        highcut = highcut or self.highcut
        fs = fs or self.fs
        order = order or self.filter_order
        
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)
    
    def calc_asymmetry(self, left_features: np.ndarray, right_features: np.ndarray) -> np.ndarray:
        """
        计算不对称性
        输入可以是单个特征值或特征数组
        """
        asymmetry = np.abs(left_features - right_features) / (left_features + right_features)
        return asymmetry

    def frequency_features(self, signal:np.ndarray, fs=None):
        """
        提取频域特征
        
        Args:
            signal: 输入信号
            fs: 采样频率，默认为类参数
            
        Returns:
            mf: 中位频率
            rmsf: 均方根频率
            rvf: 频率方差
        """
        fs = fs or self.fs
        
        f, Pxx = welch(signal, fs=fs, nperseg=1024)
        total_power = np.sum(Pxx)
        cumsum_power = np.cumsum(Pxx)
        mf = f[np.where(cumsum_power >= total_power / 2)[0][0]]  # 中位频率
        
        mpf = np.sum(f * Pxx) / total_power                        # 平均功率频率
        rmsf = np.sqrt(np.sum((f ** 2) * Pxx) / total_power)    # 均方根频率
        rvf = np.sqrt(np.sum(((f - mpf) ** 2) * Pxx) / np.sum(Pxx)) 
        return mf, rmsf, rvf
    
    def time_features(self, signal:np.ndarray):
        """
        提取时域特征
        
        Args:
            signal: 输入信号
            
        Returns:
            rms: 均方根值
            iemg: 积分肌电值
            mav: 平均绝对值
        """
        rms = np.sqrt(np.mean(signal ** 2))
        iemg = np.sum(np.abs(signal))
        mav = np.mean(np.abs(signal))
        return rms, iemg, mav    

    def analyze_symmetry_emg(self, signals: tuple[EMGSignal, EMGSignal]) -> np.ndarray:
        """
        分析两个EMG信号的对称性
        
        Args:
            signals: 包含两个EMGSignal对象的元组，分别为患(左)侧和健(右)侧信号
        Returns:
            np.ndarray: size (5,), order [rms, mav, mf, rmsf, rvf]
            每个特征的不对称性指标
        """
        left, right = signals
        
        # 检查信号是否匹配
        if left.action != right.action:
            raise ValueError(f"Input signals must have the same action. Got {left.action} and {right.action}")
        # 必须一左一右 允许调换顺序
        if left.side == 'Left' and right.side == 'Right':
            pass
        elif left.side == 'Right' and right.side == 'Left':
            left, right = right, left
        else:
            raise ValueError("Input signals must be for Left and Right sides respectively.")
        
        # 拆包
        l_data = left.data
        r_data = right.data 

        # 计算特征
        left_rms, left_iemg, left_mav = self.time_features(l_data)
        right_rms, right_iemg, right_mav = self.time_features(r_data)
        left_mf, left_rmsf, left_rvf = self.frequency_features(l_data)
        right_mf, right_rmsf, right_rvf = self.frequency_features(r_data)
        """
        mav 就是 iemg 的归一化版本，所以只保留 mav 即可
        """
        # 计算不对称性
        asym_time = self.calc_asymmetry(left_features = np.array([left_rms, left_mav]), 
                                        right_features = np.array([right_rms, right_mav]))
        asym_freq = self.calc_asymmetry(left_features = np.array([left_mf, left_rmsf, left_rvf]), 
                                        right_features = np.array([right_mf, right_rmsf, right_rvf]))
        # 拼接时间域和频域不对称性
        return np.concatenate((asym_time, asym_freq))


def main():
    """主函数，用于演示类的使用"""
    # 创建分析器实例
    analyzer = EMGAnalyzer(
        excel_file="data_set.xlsx",
        sheet_name="病人",
        fs=1000,
        lowcut=20,
        highcut=450
    )
    
    # 要分析的患者列表
    patient_names = ["xazsh", "lp", "myf", "zm", "xjh"]
    
    try:
        # 批量分析患者数据
        analyzer.analyze_multiple_patients(
            patient_names=patient_names,
            actions=['dz'],  # 只分析嘟嘴动作
            enable_filter=False,  # 不启用滤波
            plot_results=True    # 绘制结果
        )
                
    finally:
        # 关闭工作簿
        analyzer.close()


if __name__ == "__main__":
    main()