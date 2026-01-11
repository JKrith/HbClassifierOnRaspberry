from pathlib import Path
from datetime import date
import time
import logging

from Manager import Manager

def setup_logging(log_dir: Path, level=logging.INFO):
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{date.today().isoformat()}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    setup_logging(Path("logs/analyzer"))

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