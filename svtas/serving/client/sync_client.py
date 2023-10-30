'''
Author       : Thyssen Wen
Date         : 2023-10-30 15:17:17
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 19:46:39
Description  : file content
FilePath     : \ETESVS\svtas\serving\client\sync_client.py
'''
from typing import Dict
from svtas.loader import BaseDataloader
from svtas.model.post_processings import BasePostProcessing
from svtas.serving.client.connector import BaseClientConnector
from .base import BaseClient
from .visualizer import BaseClientViusalizer
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('serving_client')
class SynchronousClient(BaseClient):
    """
    Synchronous Client
    """
    def __init__(self,
                 connector: BaseClientConnector | Dict,
                 dataloader: BaseDataloader | Dict,
                 post_processing: BasePostProcessing | Dict,
                 visualizer: BaseClientViusalizer| Dict = None) -> None:
        super().__init__(connector, dataloader, post_processing, visualizer)

    def init_client(self, server_url: str = None):
        return super().init_client(server_url)
    
    def shutdown(self):
        return super().shutdown()
    
    def run(self):
        """
        run function processing
        ```
        +-----------------------+
        |  init client (outside)|
        +-----------------------+
        |  connect              |
        +-----------------------+
        |  init infer           |
        +-----------------------+
        |  send infer request   |
        +-----------------------+
        |  get infer result     |
        +-----------------------+
        |  end infer            |
        +-----------------------+
        |  disconnect           |
        +-----------------------+
        |  shutdown (outside)   |
        +-----------------------+
        """
        self.init_infer()
        self.visualizer.show()
        for iter_cnt, data in enumerate(self.dataloader):
            self.connector.send_infer_request(data_dict=data)
            results = self.connector.get_infer_results()
            if not self.post_processing_is_init():
                self.init_post_processing(data)
            self.update_post_processing(model_outputs=results, input_data=data)
            output_dict = self.output_post_processing(f"inferring_{iter_cnt}")
            if self.visualizer:
                show_data_dict = {}
                show_data_dict.update(output_dict)
                show_data_dict.update(data)
                self.visualizer.update_show_data(show_data_dict)
        self.end_infer()