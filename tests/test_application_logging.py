import logging
import unittest

from app.main import logger


class ApplicationLoggingTests(unittest.TestCase):
    def test_application_logger_keeps_info_diagnostics_enabled(self) -> None:
        self.assertTrue(logger.isEnabledFor(logging.INFO))
        self.assertFalse(logger.propagate)
        self.assertTrue(logger.handlers)
        self.assertTrue(any(handler.level <= logging.INFO for handler in logger.handlers))
