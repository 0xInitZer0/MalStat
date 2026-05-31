"""Shared helpers for PE feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000
IMAGE_SCN_MEM_WRITE = 0x80000000

DLLCHAR_DYNAMIC_BASE = 0x0040
DLLCHAR_NX_COMPAT = 0x0100
DLLCHAR_GUARD_CF = 0x4000

IMAGE_FILE_EXECUTABLE_IMAGE = 0x0002
IMAGE_FILE_DLL = 0x2000
IMAGE_SUBSYSTEM_NATIVE = 0x0001


def _normalized_suffix(file_path: str | Path | None) -> str:
    if file_path in (None, ""):
        return ""
    return Path(file_path).suffix.lstrip(".").casefold()


def derive_pe_file_flags(
    characteristics: int,
    subsystem: int,
    *,
    file_path: str | Path | None = None,
) -> dict[str, bool]:
    is_executable_image = bool(characteristics & IMAGE_FILE_EXECUTABLE_IMAGE)
    is_dll = bool(characteristics & IMAGE_FILE_DLL)
    is_native_subsystem = is_executable_image and not is_dll and subsystem == IMAGE_SUBSYSTEM_NATIVE
    suffix = _normalized_suffix(file_path)
    is_driver = is_native_subsystem and (suffix == "sys" if suffix else True)
    is_exe = is_executable_image and not is_dll and not is_driver
    return {
        "is_executable_image": is_executable_image,
        "is_dll": is_dll,
        "is_native_subsystem": is_native_subsystem,
        "is_driver": is_driver,
        "is_exe": is_exe,
    }


def classify_pe_kind(
    characteristics: int,
    subsystem: int,
    is_pe_valid: bool = True,
    *,
    file_path: str | Path | None = None,
) -> str:
    if not is_pe_valid:
        return "non_pe"

    flags = derive_pe_file_flags(characteristics, subsystem, file_path=file_path)
    if flags["is_dll"]:
        return "dll"
    if flags["is_driver"]:
        return "driver"
    if flags["is_exe"]:
        return "exe"
    return "other_pe"


def decode_section_name(section: Any) -> str:
    return getattr(section, "Name", b"").rstrip(b"\x00").decode(
        "utf-8",
        errors="ignore",
    )


def section_contains_rva(section: Any, rva: int) -> bool:
    start = int(getattr(section, "VirtualAddress", 0) or 0)
    virtual_size = int(getattr(section, "Misc_VirtualSize", 0) or 0)
    raw_size = int(getattr(section, "SizeOfRawData", 0) or 0)
    size = max(virtual_size, raw_size)
    if size <= 0:
        return False
    return start <= rva < start + size


def get_section_bytes(pe_bytes: bytes, section: Any) -> bytes:
    offset = int(getattr(section, "PointerToRawData", 0) or 0)
    raw_size = int(getattr(section, "SizeOfRawData", 0) or 0)
    virt_size = int(getattr(section, "Misc_VirtualSize", 0) or 0)
    # Use the smaller of VirtualSize and SizeOfRawData: VirtualSize is the actual
    # code/data size; SizeOfRawData is disk-aligned and may include unrelated data.
    size = min(raw_size, virt_size) if virt_size > 0 else raw_size
    if offset < 0 or size <= 0:
        return b""
    return pe_bytes[offset: offset + size]


def get_overlay_bytes(pe_bytes: bytes, pe: Any | None) -> bytes:
    if pe is None:
        return b""

    try:
        offset = pe.get_overlay_data_start_offset()
    except Exception:
        offset = None

    if offset is None:
        max_end = 0
        for section in getattr(pe, "sections", []):
            start = int(getattr(section, "PointerToRawData", 0) or 0)
            size = int(getattr(section, "SizeOfRawData", 0) or 0)
            max_end = max(max_end, start + size)
        offset = max_end

    if offset is None or offset < 0 or offset >= len(pe_bytes):
        return b""
    return pe_bytes[offset:]


def get_primary_pe_bytes(pe_bytes: bytes, pe: Any | None) -> bytes:
    overlay = get_overlay_bytes(pe_bytes, pe)
    if not overlay:
        return pe_bytes
    return pe_bytes[: len(pe_bytes) - len(overlay)]


def safe_rva_to_offset(pe: Any | None, rva: int) -> int | None:
    if pe is None:
        return None
    try:
        return int(pe.get_offset_from_rva(rva))
    except Exception:
        return None


def section_flags(section: Any) -> tuple[bool, bool, bool]:
    characteristics = int(getattr(section, "Characteristics", 0) or 0)
    is_executable = bool(characteristics & IMAGE_SCN_MEM_EXECUTE)
    is_readable = bool(characteristics & IMAGE_SCN_MEM_READ)
    is_writable = bool(characteristics & IMAGE_SCN_MEM_WRITE)
    return is_executable, is_readable, is_writable
