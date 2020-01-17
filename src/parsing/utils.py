import logging


LOGGING_NAMESPACE = "diora"


def configure_logger():
    """
    Simple logging configuration.
    """

    # Create logger.
    logger = logging.getLogger(LOGGING_NAMESPACE)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Log to stdout.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # HACK: Weird fix that counteracts other libraries (i.e. allennlp) modifying
    # the global logger.
    if len(logger.parent.handlers) > 0:
        logger.parent.handlers.pop()

    return logger


def get_logger():
    return logging.getLogger(LOGGING_NAMESPACE)
