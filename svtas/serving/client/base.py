'''
Author       : Thyssen Wen
Date         : 2023-10-27 16:16:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 19:42:48
Description  : file content
FilePath     : \ETESVS\svtas\serving\client\base.py
'''
import abc
from typing import Dict, Any
from svtas.utils.logger import BaseLogger, get_root_logger_instance
from svtas.loader import BaseDataloader
from .connector import BaseClientConnector
from .visualizer import BaseClientViusalizer
from svtas.model.post_processings import BasePostProcessing
from svtas.utils import AbstractBuildFactory

class BaseClient(metaclass=abc.ABCMeta):
    logger: BaseLogger
    dataloader: BaseDataloader
    post_processing: BasePostProcessing
    connector: BaseClientConnector
    visualizer: BaseClientViusalizer
    
    def __init__(self,
                 connector: BaseClientConnector | Dict,
                 dataloader: BaseDataloader | Dict,
                 post_processing: BasePostProcessing | Dict,
                 visualizer: BaseClientViusalizer| Dict = None) -> None:
        self.logger = get_root_logger_instance()
        # init dataloader
        if not isinstance(dataloader, BaseDataloader):
            self.dataloader = AbstractBuildFactory.create_factory('dataloader').create(dataloader)
        else:
            self.dataloader = dataloader
        # init post processing
        if not isinstance(post_processing, BasePostProcessing):
            self.post_processing = AbstractBuildFactory.create_factory('post_processing').create(post_processing)
        else:
            self.post_processing = post_processing
        # init connector
        if not isinstance(post_processing, BaseClientConnector):
            self.connector = AbstractBuildFactory.create_factory('serving_client_connector').create(connector)
        else:
            self.connector = post_processing
        # init connector
        if visualizer is not None:
            self.visualizer = AbstractBuildFactory.create_factory('serving_client_visualizer').create(visualizer)
        else:
            self.visualizer = visualizer
    
    def set_dataloader(self, dataloader: BaseDataloader):
        assert isinstance(dataloader, BaseDataloader)
        self.dataloader = dataloader
    
    def post_processing_is_init(self):
        if self.post_processing is not None:
            return self.post_processing.init_flag
        else:
            return False
    
    def set_post_processing_init_flag(self, val: bool):
        if self.post_processing is not None:
            self.post_processing.init_flag = val

    def init_post_processing(self, input_data: Dict) -> None:
        if self.post_processing is not None:
            batch_size = list(input_data.values())[0].shape[0]
            self.post_processing.init_scores()

    def update_post_processing(self, model_outputs, input_data) -> None:
        if self.post_processing is not None:
            score = model_outputs['output']
            output = self.post_processing.update(score)
        else:
            output = None
        return output

    def output_post_processing(self, cur_vid, model_outputs = None, input_data = None):
        if self.post_processing is not None:
            # get pred result
            pred_score_list, pred_cls_list = self.post_processing.output()
            outputs = dict(predict=pred_cls_list,
                        output_np=pred_score_list)
            output_dict = dict(
                vid=cur_vid,
                outputs=outputs
            )
            return output_dict
        else:
            return {}

    def init_infer(self) -> None:
        self.set_post_processing_init_flag(False)
        self.dataloader.shuffle_dataloader(1)
        self.connector.init_port()

    def end_infer(self) -> None:
        self.connector.shutdown_port()

    @abc.abstractmethod
    def init_client(self, server_url: str = None):
        self.connector.connect(server_url)
        self.visualizer.init()

    @abc.abstractmethod
    def shutdown(self):
        self.connector.disconnect()
        self.visualizer.shutdown()

    @abc.abstractmethod
    def run(self):
        """
        run function processing
        ```
        +-----------------------+
        |  init client (outside)|
        +-----------------------+
        |  init infer           |
        +-----------------------+
        |  send infer request   |
        +-----------------------+
        |  get infer result     |
        +-----------------------+
        |  end infer            |
        +-----------------------+
        |  shutdown (outside)   |
        +-----------------------+
        """
        pass