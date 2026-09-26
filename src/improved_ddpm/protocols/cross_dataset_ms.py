"""Controlled bidirectional MSU-MFSD <-> SiW-Mv2 benchmark specs."""

from improved_ddpm.protocols.protocol_a import ProtocolASpec


CROSS_DATASET_MS_REGISTRY = {
    "M_TO_S": ProtocolASpec("M_to_S", ("M",), "S"),
    "S_TO_M": ProtocolASpec("S_to_M", ("S",), "M"),
}


def normalize_cross_dataset_name(name: str) -> str:
    text = str(name).strip().upper().replace("-", "_").replace("→", "_TO_")
    text = text.replace(" ", "").replace("->", "_TO_")
    return {"MS": "M_TO_S", "M_S": "M_TO_S", "MTS": "M_TO_S",
            "SM": "S_TO_M", "S_M": "S_TO_M", "STM": "S_TO_M"}.get(text, text)


def get_cross_dataset_spec(name: str) -> ProtocolASpec:
    key = normalize_cross_dataset_name(name)
    if key not in CROSS_DATASET_MS_REGISTRY:
        raise ValueError(f"Unknown setting {name!r}; choose M_to_S or S_to_M")
    return CROSS_DATASET_MS_REGISTRY[key]
