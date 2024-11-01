from tcbench.core import StringEnum

class DATASET_NAME(StringEnum):
    MIRAGE19 = "mirage19"
    MIRAGE20 = "mirage20"
    MIRAGE22 = "mirage22"
    MIRAGE24 = "mirage24"
    UCDAVIS19 = "ucdavis19"
    UTMOBILENET21 = "utmobilenet21"

class DATASET_TYPE(StringEnum):
    RAW = "raw"
    CURATE = "curate"

