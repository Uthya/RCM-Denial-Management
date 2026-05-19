import enum


class FileType(str, enum.Enum):
    edi_837 = "edi_837"
    edi_835 = "edi_835"


class ClaimStatus(str, enum.Enum):
    submitted = "submitted"
    paid = "paid"
    denied = "denied"
    partially_paid = "partially_paid"
    void = "void"


class CodeType(str, enum.Enum):
    carc = "carc"
    rarc = "rarc"
    pos = "pos"
    claim_status = "claim_status"


class RelationshipType(str, enum.Enum):
    corrected = "corrected"
    replacement = "replacement"
    resubmission = "resubmission"
    void = "void"
