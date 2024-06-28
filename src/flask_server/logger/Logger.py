import datetime
import json
import logging
import logging.config
import os


class Logger:
    def __init__(self, config_path='logger_config.json'):
        self.config_path = config_path
        self.log_dir = 'logs'
        self.configure_logger()
        self.log_file = 'app.log'

    @staticmethod
    def add_timestamp_to_filename(filename='app.log'):
        log_name = filename.split('.')[0]
        return f"{log_name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"

    def configure_logger(self):
        # Create the logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Update the configuration file with the correct log file path
        try:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)

            # Set the log file path
            log_name = config['handlers']['fileHandler']['filename']
            self.log_file = self.add_timestamp_to_filename(filename=log_name)

            log_file_path = os.path.join(self.log_dir, self.log_file)
            config['handlers']['fileHandler']['filename'] = log_file_path

            # Configure the logger
            logging.config.dictConfig(config)
        except Exception as e:
            print(f"Error configuring logger: {e}")
            raise

    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)

    def delete_old_logs(self, days=7):
        """Delete log files older than the specified number of days."""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        for filename in os.listdir(self.log_dir):
            file_path = os.path.join(self.log_dir, filename)
            if os.path.isfile(file_path):
                file_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                if file_creation_time < cutoff_date:
                    os.remove(file_path)
                    print(f"Deleted old log file: {file_path}")

    def log_message(self, level, message):
        """Log a message with the specified logging level."""
        getattr(self.get_logger(__name__), level)(message)


# Usage example
if __name__ == "__main__":
    logger_instance = Logger(config_path='../configs/logger_config.json')
    logger = logger_instance.get_logger(__name__)

    # Log messages
    logger_instance.log_message('debug', "This is a debug message.")
    logger_instance.log_message('info', "This is an info message.")
    logger_instance.log_message('warning', "This is a warning message.")
    logger_instance.log_message('error', "This is an error message.")
    logger_instance.log_message('critical', "This is a critical message.")

    # Delete logs older than 7 days
    logger_instance.delete_old_logs(days=7)
