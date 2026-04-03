import serial
import csv
import time
import threading
import queue
import logging
from typing import Literal, Optional
import RPi.GPIO as GPIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BUTTON_START = 17
BUTTON_SWITCH = 27


class UARTReader:
    def __init__(
        self,
        port: str,
        baudrate: int,
        data_num: int,
        header: bytes,
        footer: bytes,
        byteorder: Literal["little", "big"],
        data_type: str = "uint16",
        csv_file: str = "./action_1.csv",
        csv_title=None
    ):
        self.port = port
        self.baudrate = baudrate
        self.csv_file = csv_file
        self.data_type = data_type
        self.serial: Optional[serial.Serial] = None
        self.header = header
        self.footer = footer
        self.byteorder = byteorder
        self.data_num = data_num
        self.csv_title = csv_title if csv_title is not None else []
        self.frame_length = 2 + (2 * self.data_num) + 2

        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.read_thread: Optional[threading.Thread] = None
        self.write_thread: Optional[threading.Thread] = None

        self.csv_file_handle = None
        self.csv_writer = None
        self.start_time = None

    def open_serial(self):
        if self.serial and self.serial.is_open:
            logger.info("串口已打开，跳过重复打开。")
            return

        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        logger.info(f"串口 {self.port} 已打开，波特率 {self.baudrate}")

    def close_serial(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info(f"串口 {self.port} 已关闭。")
        self.serial = None

    def close_csv(self):
        if self.csv_file_handle:
            self.csv_file_handle.close()
            logger.info(f"CSV 文件 {self.csv_file} 已关闭。")
            self.csv_file_handle = None
            self.csv_writer = None

    def read_data(self):
        buffer = bytearray()

        while not self.stop_event.is_set():
            try:
                if self.serial is None or not self.serial.is_open:
                    time.sleep(0.05)
                    continue

                data = self.serial.read(1024)
                if not data:
                    continue

                buffer.extend(data)

                while len(buffer) >= self.frame_length:
                    if (
                        buffer[0:2] == self.header
                        and buffer[self.frame_length - 2:self.frame_length] == self.footer
                    ):
                        frame = buffer[2:self.frame_length - 2]
                        data_points = [
                            int.from_bytes(frame[i:i + 2], byteorder=self.byteorder)
                            for i in range(0, len(frame), 2)
                        ]
                        self.data_queue.put(data_points)
                        buffer = buffer[self.frame_length:]
                    else:
                        buffer.pop(0)

            except Exception as e:
                logger.error(f"读取串口数据失败: {e}")
                time.sleep(0.05)

    def write_to_csv(self):
        row_count = 0

        while not self.stop_event.is_set() or not self.data_queue.empty():
            try:
                data = self.data_queue.get(timeout=0.1)
                elapsed_ms = int((time.time() - self.start_time) * 1000) if self.start_time else 0
                row_data = [elapsed_ms] + data

                if self.csv_writer:
                    self.csv_writer.writerow(row_data)
                    row_count += 1

                if row_count >= 50000:
                    logger.info("达到最大采集行数，自动停止。")
                    self.stop_event.set()
                    break

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"写入 CSV 失败: {e}")

    def start(self):
        if self.read_thread and self.read_thread.is_alive():
            print("已经在采集中，忽略重复启动")
            return

        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        self.stop_event.clear()
        self.start_time = time.time()

        self.open_serial()

        self.csv_file_handle = open(self.csv_file, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file_handle)

        if self.csv_title:
            self.csv_writer.writerow(self.csv_title)

        self.read_thread = threading.Thread(target=self.read_data, daemon=True)
        self.write_thread = threading.Thread(target=self.write_to_csv, daemon=True)

        self.read_thread.start()
        self.write_thread.start()
        logger.info(f"开始采集，保存到 {self.csv_file}")

    def stop(self):
        self.stop_event.set()

        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)

        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=1.0)

        self.close_serial()
        self.close_csv()
        logger.info("采集已停止。")

    def update_csv(self, new_file: str):
        self.csv_file = new_file
        logger.info(f"CSV 文件切换为: {new_file}")


def setup_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_START, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def reset_csv(file_path, csv_title):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_title)


if __name__ == "__main__":
    PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7A101239-if00"
    BAUDRATE = 230400
    DATA_LENGTH = 11
    CSV_TITLE = ["time(ms)"] + [f"ch{i+1}" for i in range(DATA_LENGTH)]
    csv_files = [f"./action_{i}.csv" for i in range(1, 6)]

    current_action = 0
    running = False

    uart_reader = UARTReader(
        port=PORT,
        baudrate=BAUDRATE,
        data_num=DATA_LENGTH,
        header=bytes.fromhex("A111"),
        footer=bytes.fromhex("5111"),
        byteorder="big",
        csv_file=csv_files[current_action],
        csv_title=CSV_TITLE
    )

    def start_stop_callback(channel):
        nonlocal_running = globals()
        global running

        if not running:
            print("▶ 开始采集")
            try:
                uart_reader.start()
                running = True
            except Exception as e:
                print(f"启动采集失败: {e}")
                running = False
        else:
            print("⏹ 停止采集")
            uart_reader.stop()
            running = False

    def switch_action_callback(channel):
        global current_action, running

        if running:
            uart_reader.stop()
            running = False

        current_action = (current_action + 1) % len(csv_files)
        file_path = csv_files[current_action]
        print(f"切换到动作 {current_action + 1}: {file_path}")

        reset_csv(file_path, CSV_TITLE)
        uart_reader.update_csv(file_path)

    try:
        setup_gpio()

        try:
            GPIO.remove_event_detect(BUTTON_START)
        except RuntimeError:
            pass

        try:
            GPIO.remove_event_detect(BUTTON_SWITCH)
        except RuntimeError:
            pass

        GPIO.add_event_detect(
            BUTTON_START,
            GPIO.FALLING,
            callback=start_stop_callback,
            bouncetime=300
        )

        GPIO.add_event_detect(
            BUTTON_SWITCH,
            GPIO.FALLING,
            callback=switch_action_callback,
            bouncetime=300
        )

        print("系统启动，等待按键...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("退出程序")

    finally:
        try:
            uart_reader.stop()
        except Exception:
            pass
        GPIO.cleanup()