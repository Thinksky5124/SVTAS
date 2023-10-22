'''
Author       : Thyssen Wen
Date         : 2023-09-25 18:57:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-21 16:53:47
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/loss_record.py
'''
from typing import Dict, List
from .base_record import BaseRecord
from .meter import AverageMeter
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('record')
class ValueRecord(BaseRecord):
    def __init__(self,
                 addition_record: List[Dict] = [
                    dict(name='loss', fmt='.7f'),
                    dict(name='Acc', fmt='.5f'),
                    dict(name='lr', fmt='.5f')],
                 accumulate_type: Dict[str, str] = {'loss': 'avg', 'Acc': 'avg'}) -> None:
        assert isinstance(addition_record, list), "You must input list!"
        assert isinstance(accumulate_type, dict), "You must input dict!"
        addition_record += [
            dict(name='iter_time', fmt='.5f'),
            dict(name='batch_time', fmt='.5f'),
            dict(name='reader_time', fmt='.5f')]

        accumulate_type['iter_time'] = 'avg'
        accumulate_type['batch_time'] = 'avg'
        accumulate_type['reader_time'] = 'avg'

        super().__init__(addition_record)
        for a in self.addition_record:
            self._record[a['name']] = AverageMeter(**a)
        
        for key in self._record.keys():
            if key not in accumulate_type:
                accumulate_type[key] = 'avg'
            else:
                assert accumulate_type[key] in ['avg', 'sum', 'val'], f'Unsupport accumulate_type: {accumulate_type}!'
        self.accumulate_type = accumulate_type
    
    @property
    def record_dict(self) -> Dict[str, AverageMeter]:
        return self._record
    
    def add_record(self, name, fmt='f'):
        self._record[name] = AverageMeter(name=name, fmt=fmt)
    
    def update_one_record(self, name, value, n = 1):
        self.record_dict[name].update(value, n)
    
    def get_one_record(self, name, accumulate_type = None):
        if accumulate_type is None:
            accumulate_type = self.accumulate_type[name]
        return getattr(self.record_dict[name], accumulate_type)
        
    def init_record(self):
        for key, value in self.record_dict.items():
            value.reset()
    
    def update_record(self, update_dict: Dict):
        for key, value in update_dict.items():
            if key in self.record_dict:
                self.record_dict[key].update(value)
    
    def accumulate_record(self):
        pass

@AbstractBuildFactory.register('record')
class StreamValueRecord(ValueRecord):
    def __init__(self,
                 addition_record: List[Dict] = [
                    dict(name='loss', fmt='.7f'),
                    dict(name='Acc', fmt='.5f'),
                    dict(name='lr', fmt='.5f')],
                 accumulate_type: Dict[str, str] = {'loss': 'avg', 'Acc': 'avg'}) -> None:
        super().__init__(addition_record, accumulate_type)
        self.loss_dict: Dict[str, AverageMeter] = {}
    
    def init_record(self):
        super().init_record()
        if self.record_dict is not None:
            for key, value in self.record_dict.items():
                if key.endswith("loss"):
                    if key not in self.loss_dict:
                        self.loss_dict[key] = AverageMeter(name=key, fmt=value.fmt)
                    else:
                        self.loss_dict[key].reset()

    def stream_update_dict(self, update_dict: Dict):
        for key, value in update_dict.items():
            if key in self.loss_dict:
                self.loss_dict[key].update(value)
    
    def accumulate_loss_dict(self):
        for key, value in self.loss_dict.items():
            self.record_dict[key].update(getattr(value, self.accumulate_type[key]))
    
    def accumulate_record(self):
        self.accumulate_loss_dict()
        super().accumulate_record()

@AbstractBuildFactory.register('record')
class ExtractRecord(BaseRecord):
    def __init__(self,
                 addition_record: List[Dict] = [],
                 accumulate_type: Dict[str, str] = {}) -> None:
        assert isinstance(addition_record, list), "You must input list!"
        assert isinstance(accumulate_type, dict), "You must input dict!"
        addition_record += [
            dict(name='iter_time', fmt='.5f'),
            dict(name='batch_time', fmt='.5f'),
            dict(name='reader_time', fmt='.5f')]

        accumulate_type['iter_time'] = 'avg'
        accumulate_type['batch_time'] = 'avg'
        accumulate_type['reader_time'] = 'avg'
        super().__init__(addition_record)
        for a in self.addition_record:
            self._record[a['name']] = AverageMeter(**a)
        
        for key in self._record.keys():
            if key not in accumulate_type:
                accumulate_type[key] = 'avg'
            else:
                assert accumulate_type[key] in ['avg', 'sum', 'val'], f'Unsupport accumulate_type: {accumulate_type}!'
        self.accumulate_type = accumulate_type