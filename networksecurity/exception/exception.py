import sys
import traceback


class NetworkSecurityException(Exception):
    """
    Custom exception that captures file name, line number, and original message.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.filename = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = "N/A"
            self.filename = "N/A"

        self.error_message = error_message

    def __str__(self):
        return (
            f"Error in script: [{self.filename}] "
            f"at line [{self.lineno}] — {self.error_message}"
        )
