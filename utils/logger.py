import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('PipelineLogger')
    return logger