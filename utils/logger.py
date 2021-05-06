# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


default_fmt = '%(asctime)s %(levelname)-8s %(message)s'
default_date_fmt = '%Y-%m-%d %H:%M:%S'

default_log = 'output.log'
default_error_log = 'error.log'

log_max_bytes = 2 ** 30 * 1
log_backup_count = 2


class EscapeCoder:

    foreground = {
        'Black': '30', 'Red': '31', 'Green': '32', 'Yellow': '33',
        'Blue': '34', 'Magenta': '35', 'Cyan': '36', 'White': '37',
        'Bright Black': '90', 'Bright Red': '91', 'Bright Green': '92',
        'Bright Yellow': '93', 'Bright Blue': '94', 'Bright Magenta': '95',
        'Bright Cyan': '96', 'Bright White': '97'
    }

    background = {
        'Black': '40', 'Red': '41', 'Green': '42', 'Yellow': '43',
        'Blue': '44', 'Magenta': '45', 'Cyan': '46', 'White': '47',
        'Bright Black': '100', 'Bright Red': '101', 'Bright Green': '102',
        'Bright Yellow': '103', 'Bright Blue': '104', 'Bright Magenta': '105',
        'Bright Cyan': '106', 'Bright White': '107'
    }

    bold = '1'
    italic = '3'
    underline = '4'

    @classmethod
    def format(
            cls, text, foreground=None, background=None,
            bold=False, italic=False, underline=False):
        codes = []
        if foreground:
            codes.append(cls.foreground[foreground])
        if background:
            codes.append(cls.background[background])
        if bold:
            codes.append(cls.bold)
        if italic:
            codes.append(cls.italic)
        if underline:
            codes.append(cls.underline)
        codes_concat = ';'.join(codes)
        if codes_concat:
            fmt = '\x1b[%sm{}\x1b[0m' % (codes_concat)
        else:
            fmt = '{}'

        return fmt.format(text)

    @classmethod
    def get_formatter(
            cls, foreground=None, background=None,
            bold=False, italic=False, underline=False):
        codes = []
        if foreground:
            codes.append(cls.foreground[foreground])
        if background:
            codes.append(cls.background[background])
        if bold:
            codes.append(cls.bold)
        if italic:
            codes.append(cls.italic)
        if underline:
            codes.append(cls.underline)
        codes_concat = ';'.join(codes)
        if codes_concat:
            fmt = '\x1b[%sm{}\x1b[0m' % (codes_concat)
        else:
            fmt = '{}'

        def format(message):
            return fmt.format(message)

        return format

    @classmethod
    def get_debug(cls):
        return cls.get_formatter('Cyan', bold=True)

    @classmethod
    def get_info(cls):
        return cls.get_formatter(bold=True)

    @classmethod
    def get_warning(cls):
        return cls.get_formatter('Yellow', bold=True)

    @classmethod
    def get_error(cls):
        return cls.get_formatter('Red', bold=True)

    @classmethod
    def get_critical(cls):
        return cls.get_formatter('Bright Red', bold=True, underline=True)

    @classmethod
    def get_formatter_by_level(cls, level):
        if level == logging.DEBUG:
            return cls.get_debug()
        elif level == logging.INFO:
            return cls.get_info()
        elif level == logging.WARNING:
            return cls.get_warning()
        elif level == logging.ERROR:
            return cls.get_error()
        elif level == logging.CRITICAL:
            return cls.get_critical()
        else:
            raise NotImplementedError()

    @classmethod
    def get_date(cls):
        return cls.get_formatter('Green')


class ColorFormatter(logging.Formatter):

    def __init__(self, fmt=default_fmt, date_fmt=default_date_fmt, style='%'):
        super().__init__(fmt, date_fmt, style)

    def format(self, record):

        message_decorate = EscapeCoder.get_formatter_by_level(record.levelno)
        record.msg = message_decorate(record.msg)
        record.levelname = message_decorate('%s:' % record.levelname)

        formatted = super().format(record)
        return formatted

    def formatTime(self, record, date_fmt=None):
        date_decorate = EscapeCoder.get_date()
        res = super().formatTime(record, date_fmt)
        return date_decorate(res)


def parse_level(level):
    if type(level) == str:
        if level == 'DEBUG':
            return logging.DEBUG
        elif level == 'INFO':
            return logging.INFO
        elif level == 'WARNING':
            return logging.WARNING
        elif level == 'ERROR':
            return logging.ERROR
        elif level == 'CRITICAL':
            return logging.CRITICAL
        else:
            raise NotImplementedError()
    else:
        return level


class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.ERROR, logging.CRITICAL)


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.DEBUG, logging.INFO, logging.WARNING)


def get_logger(
        name, log_dir=None, level=logging.INFO, split_error=False,
        fmt=default_fmt, use_tqdm=False):
    logger = logging.getLogger(name)
    if getattr(logger, 'initialized', False):
        return logger
    logger.initialized = True

    logger.setLevel(parse_level(level))

    del logger.handlers[:]

    if log_dir:
        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / default_log
        file_handler = RotatingFileHandler(
            str(log_file), maxBytes=log_max_bytes, backupCount=log_backup_count)
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

        if split_error:
            file_handler.addFilter(InfoFilter())

            error_log_file = log_dir / default_error_log
            error_file_handler = RotatingFileHandler(
                error_log_file, maxBytes=log_max_bytes,
                backupCount=log_backup_count)
            error_file_handler.formatter = logging.Formatter(fmt)
            error_file_handler.addFilter(ErrorFilter())
            logger.addHandler(error_file_handler)

    if use_tqdm:
        from tqdm import tqdm
        class TqdmHandler(logging.StreamHandler):
            def emit(self, record):
                msg = self.format(record)
                tqdm.write(msg)
        stream_handler = TqdmHandler()
    else:
        stream_handler = logging.StreamHandler()
    stream_handler.formatter = ColorFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
