import sys  # for accessing current exception info
import traceback  # to extract full traceback details
from typing import Optional, cast  # for type hints and safe type casting


class DocumentPortalException(Exception):
    """Custom exception for Document Portal that captures detailed traceback info"""

    def __init__(self, error_message, error_details: Optional[object] = None):
        # --- Step 1: Normalize the incoming message ---
        # If an Exception object is passed, convert it to a readable string
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # --- Step 2: Determine how to extract traceback info ---
        # Supports three cases:
        # 1. No error_details → use current exception from sys.exc_info()
        # 2. error_details = sys module (old pattern)
        # 3. error_details = Exception object (new pattern)
        exc_type = exc_value = exc_tb = None  # placeholders for exception info

        if error_details is None:
            # Case 1: automatically capture current exception context
            exc_type, exc_value, exc_tb = sys.exc_info()

        else:
            if hasattr(error_details, "exc_info"):  # e.g., sys object passed
                # Case 2: extract info from sys.exc_info()
                exc_info_obj = cast(sys, error_details)  # tell type checker it's sys-like
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()

            elif isinstance(error_details, BaseException):
                # Case 3: direct Exception object
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__

            else:
                # Fallback: still try sys.exc_info()
                exc_type, exc_value, exc_tb = sys.exc_info()

        # --- Step 3: Walk down the traceback to find the last frame ---
        # This ensures we report the *deepest* point where the error occurred
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # --- Step 4: Capture filename and line number ---
        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1

        # --- Step 5: Store final error message ---
        self.error_message = norm_msg

        # --- Step 6: Format the full traceback text (if available) ---
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        # Call base Exception initializer with the formatted message
        super().__init__(self.__str__())

    def __str__(self):
        # --- Step 7: Define how the error is displayed or logged ---
        # Compact form suitable for structured logs or console display
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"

        # Add traceback text only if available
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        # --- Step 8: Developer-friendly representation ---
        # Useful when inspecting objects in debugger or logs
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"



