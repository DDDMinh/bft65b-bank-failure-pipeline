"""Phase 1 acquisition package."""

from bft65b.acquire.client import FDICClient, FDICRequestError, PageResult, TotalDriftError
from bft65b.acquire.dictionary import (
    DictionaryError,
    build_field_dictionary,
    dictionary_violations,
    export_field_dictionary,
    load_metadata,
)
from bft65b.acquire.downloader import (
    AcquisitionDownloader,
    AcquisitionError,
    CompletenessError,
    DownloadResult,
    Manifest,
)
from bft65b.acquire.schema import FieldRegistry, SchemaError
from bft65b.acquire.summary import (
    build_acquisition_summary,
    write_acquisition_summary,
)
from bft65b.acquire.verify import VerificationError, verify_acquisition

__all__ = [
    "AcquisitionDownloader",
    "AcquisitionError",
    "CompletenessError",
    "DictionaryError",
    "DownloadResult",
    "FDICClient",
    "FDICRequestError",
    "FieldRegistry",
    "Manifest",
    "PageResult",
    "SchemaError",
    "TotalDriftError",
    "VerificationError",
    "build_acquisition_summary",
    "build_field_dictionary",
    "dictionary_violations",
    "export_field_dictionary",
    "load_metadata",
    "verify_acquisition",
    "write_acquisition_summary",
]
