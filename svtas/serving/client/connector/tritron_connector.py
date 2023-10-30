'''
Author       : Thyssen Wen
Date         : 2023-10-30 14:47:55
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 16:45:11
Description  : file content
FilePath     : /SVTAS/svtas/serving/client/connector/tritron_connector.py
'''
import os
import sys
import numpy as np
from typing import Dict, Any, List
from functools import partial
from svtas.loader import BaseDataloader
from svtas.model.post_processings import BasePostProcessing

from svtas.utils import (is_tritonclient_available,
                         AbstractBuildFactory)
from .base_connector import BaseClientConnector
from svtas.utils import AbstractBuildFactory

if is_tritonclient_available():
    import tritonclient.grpc as grpcclient
    import tritonclient.grpc.model_config_pb2 as mc
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))
    
def convert_http_metadata_config(_metadata, _config):
    # NOTE: attrdict broken in python 3.10 and not maintained.
    # https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
    try:
        from attrdict import AttrDict
    except ImportError:
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)

@AbstractBuildFactory.register('serving_client_connector')
class TritronConnector(BaseClientConnector):
    """
    
    """
    batch_size: int
    triton_client: None
    user_data: UserData
    async_requests: List
    responses: List

    def __init__(self,
                 model_name: str,
                 server_url: str,
                 model_version: str = "",
                 async_serving: bool = False,
                 verbose: bool = False,
                 protocol: str = "http",
                 streaming_infer: bool = False) -> None:
        super().__init__(server_url)
        assert protocol in ['http', 'grpc']
        self.verbose = verbose
        self.model_name = model_name
        self.model_version = model_version
        self.protocol = protocol
        self.streaming_infer = streaming_infer
        self.async_serving = async_serving
        self.batch_size = 1
        self.sent_count = 1

        if self.streaming_infer and self.protocol != "grpc":
            raise Exception("Streaming is only allowed with gRPC protocol")

    @staticmethod
    def parse_model(model_metadata, model_config):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an temporal action segmentation network (as expected by
        this client)
        """
        output_metadata = model_metadata.outputs[0]
                
        def parse_io_tensor(tensor_metadata, tensor_config):
            return (tensor_metadata.name,
                    tensor_metadata.shape, tensor_metadata.datatype)
        # data_dict
        # name, shape, datatype
        input_dict = {}
        for input_metadata, input_config in zip(model_metadata.inputs, model_config.input):
            name, shape, datatype = parse_io_tensor(tensor_metadata=input_metadata,
                                                            tensor_config=input_config)
            input_dict[name] = dict(
                shape = shape,
                datatype = datatype
            )

        output_dict = {}
        for output_metadata, output_config in zip(model_metadata.outputs, model_config.output):
            name, shape, datatype = parse_io_tensor(tensor_metadata=output_metadata,
                                                            tensor_config=output_config)
            output_dict[name] = dict(
                shape = shape,
                datatype = datatype
            )
        return model_config.max_batch_size, input_dict, output_dict

    def init_port(self):
        super().init_port()
        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        self.user_data = UserData()
        if self.streaming_infer:
            self.triton_client.start_stream(partial(completion_callback, self.user_data))
        self.sent_count = 1
        self.logger.info("Init Client Successfully!")
    
    def shutdown_port(self):
        if self.streaming_infer:
            self.triton_client.stop_stream()
        self.logger.info("Shutdown Client Successfully!")

    def connect(self, server_url: str):
        if server_url is None:
            server_url = self.server_url

        try:
            if self.protocol == "grpc":
                # Create gRPC client for communicating with the server
                triton_client = grpcclient.InferenceServerClient(
                    url = server_url, verbose = self.verbose
                )
            else:
                # Specify large enough concurrency to handle the
                # the number of requests.
                concurrency = 20 if self.async_serving else 1
                triton_client = httpclient.InferenceServerClient(
                    url = server_url, verbose = self.verbose, concurrency=concurrency
                )
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name = self.model_name, model_version = self.model_version
            )
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        try:
            model_config = triton_client.get_model_config(
                model_name = self.model_name, model_version = self.model_version
            )
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        if self.protocol.lower() == "grpc":
            model_config = model_config.config
        else:
            model_metadata, model_config = convert_http_metadata_config(
                model_metadata, model_config
            )

        max_batch_size, input_dict, output_dict = self.parse_model(model_metadata, model_config)

        supports_batching = max_batch_size > 0
        if not supports_batching and self.batch_size != 1:
            print("ERROR: This model doesn't support batching.")
            sys.exit(1)

        # Holds the handles to the ongoing HTTP async requests.
        self.async_requests = []
        self.responses = []

        self.triton_client = triton_client
        self.max_batch_size = max_batch_size
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.logger.info("Connect Client Successfully!")
    
    def disconnect(self):
        self.logger.info("Disconnect Client Successfully!")
    
    @staticmethod
    def input_check(tensor: np.array, shape: List[int], dtype) -> bool:
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        def shape_check(tensor_shape, critrion_shape):
            for dim, c_dim in zip(tensor_shape, critrion_shape):
                if dim != c_dim and c_dim != -1:
                    return False
                elif c_dim == -1:
                    continue
            return True
        
        assert isinstance(tensor, np.ndarray), "Input data must be np.array!"
        assert shape_check(tensor.shape, shape), f"Input data shape must be {shape}!"
        # assert tensor.dtype == dtype, f"Input data type must be {dtype}!"
        return True
    
    def send_infer_request(self, data_dict: Dict[str, Any]) ->bool:
        # Preprocess the images into input data according to model
        # requirements
        for name, value in data_dict.items():
            if name in self.input_dict:
                self.input_check(tensor=value, shape=self.input_dict[name]['shape'],
                                dtype=self.input_dict[name]['datatype'])

        # Send request
        try:
            if self.protocol == "grpc":
                client = grpcclient
            else:
                client = httpclient

            inputs = []
            outputs = []
            for name, value in data_dict.items():
                if name in self.input_dict:
                    # Set the input data
                    input_class = client.InferInput(name, self.input_dict[name]['shape'], self.input_dict[name]['datatype'])
                    input_class.set_data_from_numpy(value)
                    inputs.append(input_class)

            for name, value in self.output_dict.items():
                outputs.append(client.InferRequestedOutput(name))

            if self.streaming_infer:
                self.triton_client.async_stream_infer(
                    self.model_name,
                    inputs,
                    request_id=str(self.sent_count),
                    model_version=self.model_version,
                    outputs=outputs,
                )
            elif self.async_serving:
                if self.protocol == "grpc":
                    self.triton_client.async_infer(
                        self.model_name,
                        inputs,
                        partial(completion_callback, self.user_data),
                        request_id=str(self.sent_count),
                        model_version=self.model_version,
                        outputs=outputs,
                    )
                else:
                    self.async_requests.append(
                        self.triton_client.async_infer(
                            self.model_name,
                            inputs,
                            request_id=str(self.sent_count),
                            model_version=self.model_version,
                            outputs=outputs,
                        )
                    )
            else:
                self.responses.append(
                    self.triton_client.infer(
                        self.model_name,
                        inputs,
                        request_id=str(self.sent_count),
                        model_version=self.model_version,
                        outputs=outputs,
                    )
                )
            self.sent_count += 1
        except InferenceServerException as e:
            self.logger.error("inference failed: " + str(e))
            if self.streaming_infer:
                self.triton_client.stop_stream()
            return False
        self.logger.log(f"Infer request id: {self.sent_count - 1} send successfully...")
        return True

    def get_infer_results(self) -> Dict[str, Any]:
        if self.protocol == "grpc":
            if self.streaming_infer or self.async_serving:
                processed_count = 0
                while processed_count < self.sent_count:
                    (results, error) = self.user_data._completed_requests.get()
                    processed_count += 1
                    if error is not None:
                        self.logger.error("inference failed: " + str(error))
                        sys.exit(1)
                    self.responses.append(results)
        else:
            if self.async_serving:
                # Collect results from the ongoing async requests
                # for HTTP Async requests.
                for async_request in self.async_requests:
                    self.responses.append(async_request.get_result())

        output_dict = {}
        for response, (name, value) in zip(self.responses, self.output_dict.items()):
            if self.protocol == "grpc":
                this_id = response.get_response().id
            else:
                this_id = response.get_response()["id"]
            self.logger.log("Request {}, batch size {}".format(this_id, self.batch_size))
            output_array = response.as_numpy(name)
            if self.max_batch_size > 0 and len(output_array) != self.batch_size:
                raise Exception(
                    "expected {} results, got {}".format(self.batch_size, len(output_array))
                )
            output_dict[name] = output_array
        return output_dict