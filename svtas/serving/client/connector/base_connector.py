'''
Author       : Thyssen Wen
Date         : 2023-10-30 14:48:08
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 14:54:46
Description  : file content
FilePath     : /SVTAS/svtas/serving/client/connector/base_connector.py
'''
import abc
from typing import Dict, Any
from svtas.utils.logger import BaseLogger, get_root_logger_instance

class BaseClientConnector(metaclass=abc.ABCMeta):
    logger: BaseLogger
    def __init__(self,
                 server_url: str) -> None:
        self.server_url = server_url
        self.logger = get_root_logger_instance()
    
    @abc.abstractmethod
    def connect(self, server_url: str):
        pass

    @abc.abstractmethod
    def disconnect(self):
        pass

    @abc.abstractmethod
    def send_infer_request(self, data_dict: Dict[str, Any]) -> bool:
        pass
    
    @abc.abstractmethod
    def get_infer_results(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def init_port(self) -> None:
        pass

    @abc.abstractmethod
    def shutdown_port(self) -> None:
        pass
