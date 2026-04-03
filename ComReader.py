import serial
import csv
import time
import threading
import queue
import logging
from typing import Literal
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)


class UARTReader:
    def __init__(
        self,
        port: str,
        baudrate: int,
        data_num: int,
        header: bytes,
        footer: bytes,
        byteorder: Literal["little", "big"],
        data_type: str = "Only uint16",
        csv_file: str = "./output.csv",
        csv_title="",
    ):
        self.port = port
        self.baudrate = baudrate
        self.csv_file = csv_file
        self.data_type = 'uint16'
        self.serial = None
        self.header = header
        self.footer = footer
        self.byteorder = byteorder
        self.data_num = data_num  # Number of data points in a frame
        self.csv_title = csv_title
        self.frame_length = 2 + (2 * self.data_num) + 2  # Header + Data + Footer
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.read_thread = None

    def open_serial(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            logger.info(f"Serial port {self.port} opened at {self.baudrate} baud.")
        except serial.SerialException as e:
            logger.error(f"Could not open serial port: {e}")
            raise

    def close_serial(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info(f"Serial port {self.port} closed.")
        if hasattr(self, 'csv_file_handle') and self.csv_file_handle:
            self.csv_file_handle.close()
            logger.info(f"CSV file {self.csv_file} closed.")

    def read_data(self):
        buffer = bytearray()
        while not self.stop_event.is_set():
            try:
                # Read available data from the serial port
                data = self.serial.read(1024)  # Read up to 1024 bytes at a time
                if data:
                    buffer.extend(data)

                    # Process complete frames
                    while len(buffer) >= self.frame_length:
                        # Check for frame header and footer
                        if buffer[0:2] == self.header and buffer[self.frame_length-2:self.frame_length] == self.footer:
                            # Extract the data points (2 bytes each)
                            frame = buffer[2:self.frame_length-2]
                            data_points = [int.from_bytes(frame[i:i+2], byteorder='big') for i in range(0, len(frame), 2)]
                            self.data_queue.put(data_points)  # Add data to the queue

                            # Remove the processed frame from the buffer
                            buffer = buffer[self.frame_length:]
                        else:
                            # Remove the first byte if the frame is invalid
                            buffer.pop(0)
            except Exception as e:
                logger.error(f"Error reading data: {e}")

    def write_to_csv(self):
        row_count = 0  # 行计数器
        while not self.stop_event.is_set() or not self.data_queue.empty():
            try:
                data = self.data_queue.get(timeout=0.1)

                # 添加时间戳
                row_data = data

                self.csv_writer.writerow(row_data)
                row_count += 1  # 每写入一行，计数器加1
                logger.debug(
                    f"Stop event set: {self.stop_event.is_set()}, row count: {row_count}"
                )

                # 如果写入达到50行，停止写入并关闭文件
                if row_count >= 5000:
                    logger.info("Reached 50 rows, stopping...")
                    self.stop_event.set()  # 设置停止事件
                    break
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Could not write to CSV: {e}")

    def start(self):
        self.open_serial()

        # 创建CSV文件并写入标题行
        self.csv_file_handle = open(self.csv_file, mode='w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file_handle)

        # 写入标题行：time, ch1, ch2, ..., chN
        self.csv_writer.writerow(self.csv_title)

        self.stop_event.clear()
        self.read_thread = threading.Thread(target=self.read_data)
        self.write_thread = threading.Thread(target=self.write_to_csv)

        self.read_thread.start()
        self.write_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.read_thread:
            self.read_thread.join()
        if getattr(self, "write_thread", None):
            self.write_thread.join()
        if self.csv_file_handle:
            self.csv_file_handle.close()  # 确保文件关闭
            logger.info(f"CSV file {self.csv_file} closed.")

    def isStop(self):
        return self.stop_event.is_set()


if __name__ == "__main__":
    # Configuration
    PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7A101239-if00"  # Replace with your UART port (e.g., /dev/ttyS0 or /dev/ttyAMA0)
    # PORT = 'COM12'
    BAUDRATE = 230400      # Set baudrate (e.g., 230400 or 460800)
    CSV_FILE = "./output.csv"
    DATA_LENGTH = 11       # Number of 2-byte data points in a frame
    CSV_TITLE = ['time(ms)'] + [f'ch{i+1}' for i in range(DATA_LENGTH-1)]

    uart_reader = UARTReader(
    port=PORT,
    baudrate=BAUDRATE,
    data_num=DATA_LENGTH,
    header=bytes.fromhex("A111"),
    footer=bytes.fromhex("5111"),
    byteorder='big',
    csv_file=CSV_FILE,
    csv_title=CSV_TITLE
)

    try:
        uart_reader.start()
        logger.info("You can press Ctrl+C to stop the UART")
        # 等待线程完成
        while not uart_reader.stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        uart_reader.stop()
