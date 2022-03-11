import logging
import os
import sys
import datetime
import torch

Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}


def coloring(message, color="OKGREEN"):
    assert color in Color.keys()
    if os.environ.get('COLORING', True):
        return Color[color] + str(message) + Color["ENDC"]
    else:
        return message


logger_initialized = []


def setup_logger(output=None, name="ETETS", level="INFO"):
    """
    Initialize the ETETS logger and set its verbosity level to "INFO".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
    Returns:
        logging.Logger: a logger
    """
    def time_zone(sec, fmt):
        real_time = datetime.datetime.now()
        return real_time.timetuple()
    logging.Formatter.converter = time_zone

    logger = logging.getLogger(name)
    if level == "INFO":
        logger.setLevel(logging.INFO)
    elif level=="DEBUG":
        logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if level == "DEBUG":
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S")
    else:
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%m/%d %H:%M:%S")

    # stdout logging: master only
    local_rank = 0
    if local_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)

        # PathManager.mkdirs(os.path.dirname(filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # fh = logging.StreamHandler(_cached_log_stream(filename)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    logger_initialized.append(name)
    return logger


def get_logger(name, output=None):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    return setup_logger(name=name, output=name)

logger = get_logger("ETETS")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name='', fmt='f', need_avg=True):
        self.name = name
        self.fmt = fmt
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        if isinstance(val, torch.Tensor):
            val = val.cpu().detach().numpy()[0]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}_sum: {s:{self.fmt}} min'.format(s=self.sum / 60,
                                                            self=self)

    @property
    def mean(self):
        return '{self.name}_avg: {self.avg:{self.fmt}}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}'.format(self=self)


def log_batch(metric_list, batch_id, epoch_id, total_epoch, mode, ips):
    batch_cost = str(metric_list['batch_time'].value) + ' sec,'
    reader_cost = str(metric_list['reader_time'].value) + ' sec,'

    metric_values = []
    for m in metric_list:
        if not (m == 'batch_time' or m == 'reader_time'):
            metric_values.append(metric_list[m].value)
    metric_str = ' '.join([str(v) for v in metric_values])
    epoch_str = "epoch:[{:>3d}/{:<3d}]".format(epoch_id, total_epoch)
    step_str = "{:s} step:{:<4d}".format(mode, batch_id)

    logger.info("{:s} {:s} {:s} {:s} {:s} {}".format(
        coloring(epoch_str, "HEADER") if batch_id == 0 else epoch_str,
        coloring(step_str, "PURPLE"), coloring(metric_str, 'OKGREEN'),
        coloring(batch_cost, "OKGREEN"), coloring(reader_cost, 'OKGREEN'), ips))


def log_epoch(metric_list, epoch, mode, ips):
    batch_cost = 'avg_' + str(metric_list['batch_time'].value) + ' sec,'
    reader_cost = 'avg_' + str(metric_list['reader_time'].value) + ' sec,'
    batch_sum = str(metric_list['batch_time'].total) + ' sec,'

    metric_values = []
    for m in metric_list:
        if not (m == 'batch_time' or m == 'reader_time'):
            metric_values.append(metric_list[m].mean)
    metric_str = ' '.join([str(v) for v in metric_values])

    end_epoch_str = "END epoch:{:<3d}".format(epoch)

    logger.info("{:s} {:s} {:s} {:s} {:s} {:s} {}".format(
        coloring(end_epoch_str, "RED"), coloring(mode, "PURPLE"),
        coloring(metric_str, "OKGREEN"), coloring(batch_cost, "OKGREEN"),
        coloring(reader_cost, "OKGREEN"), coloring(batch_sum, "OKGREEN"), ips))