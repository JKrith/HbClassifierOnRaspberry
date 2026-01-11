import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict, permutation_test_score
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from seaborn import heatmap
import joblib
import json
import os
from datetime import datetime
import logging
import re

# 配置日志
logger = logging.getLogger(__name__)

#-----------------------------------------------
# Load data
#-----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    normalize=True,
    title="Confusion Matrix"
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def load_model_pipeline(model_path, scaler_path, config_path):
    """
    加载保存的模型管道
    
    Args:
        model_path: 模型文件路径
        scaler_path: 标准化器文件路径
        config_path: 配置文件路径
    
    Returns:
        dict: 包含模型、标准化器和配置信息的字典
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'scaler': scaler,
        'config': config
    }

def predict_new_sample(model_pipeline, new_data):
    """
    使用保存的模型管道对新样本进行预测
    
    Args:
        model_pipeline: 加载的模型管道字典
        new_data: 新样本数据 (numpy array 或 pandas DataFrame)
    
    Returns:
        numpy array: 预测结果
    """
    model = model_pipeline['model']
    scaler = model_pipeline['scaler']
    
    # 标准化新数据
    if hasattr(new_data, 'values'):
        new_data = new_data.values
    
    new_data_scaled = scaler.transform(new_data.reshape(1, -1))
    
    # 进行预测
    prediction = model.predict(new_data_scaled)
    probabilities = None
    
    # 如果模型支持概率预测，也返回概率
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(new_data_scaled)
    
    return {
        'prediction': prediction[0],
        'probabilities': probabilities[0] if probabilities is not None else None
    }

if __name__ == "__main__":
    # feature_cols = ['RMS-ym','RMS-by','MAV-ym','MAV-by','MF-ym','RMSF-by','RMSF-mz','RVF-ym','RVF-by','RMSF-dz']
    feature_cols = ['RMS-ym','RMS-by','MAV-ym','MAV-by','MF-ym']
    label_col = 'hb'

    # 演示如何加载和使用保存的模型
    print("\n=== Model Loading Demo ===")
    final_save_paths = {
        'model_path': 'saved_models/svm_loocv_20260110_121421.pkl',
        'scaler_path': 'saved_models/svm_loocv_scaler_20260110_121421.pkl',
        'config_path': 'saved_models/svm_loocv_config_20260110_121421.json',
    }
    try:
        loaded_pipeline = load_model_pipeline(final_save_paths['model_path'], 
                                            final_save_paths['scaler_path'], 
                                            final_save_paths['config_path'])
        
        print("Model loaded successfully!")
        print(f"Model type: {loaded_pipeline['config']['model_type']}")
        print(f"Features: {loaded_pipeline['config']['feature_columns']}")
        
        # 演示对新样本的预测
        X = np.ndarray([[0.1,0.2,0.3,0.4,0.5],[0,0,0,0,0,]])
        if len(X) > 0:
            sample = X[0]  # 使用第一个样本作为示例
            prediction = predict_new_sample(loaded_pipeline, sample)
            print(f"Sample prediction: {prediction['prediction']}")
            if prediction['probabilities'] is not None:
                print(f"Prediction probabilities: {prediction['probabilities']}")
                
    except Exception as e:
        print(f"Error loading model: {e}")

