"""PE header and entry-point features."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pefile

from src.core.extraction_support import get_authenticode_signature_info
from src.extractors.base import BaseExtractor
from src.extractors.pe_helpers import (
    DLLCHAR_DYNAMIC_BASE,
    DLLCHAR_GUARD_CF,
    DLLCHAR_NX_COMPAT,
    IMAGE_FILE_DLL,
    IMAGE_FILE_EXECUTABLE_IMAGE,
    IMAGE_SUBSYSTEM_NATIVE,
    decode_section_name,
    derive_pe_file_flags,
    get_section_bytes,
    safe_rva_to_offset,
    section_contains_rva,
    section_flags,
)

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class HeaderFeaturesExtractor(BaseExtractor):
    """Extracts PE header fields, security flags, and entry-point heuristics."""

    def __init__(self, registry: "IndicatorRegistry") -> None:
        super().__init__(registry)

    @property
    def feature_names(self) -> list[str]:
        return list(self._defaults().keys())

    def extract(
        self,
        pe_bytes: bytes,
        lief_pe: Any,
        file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        try:
            if lief_pe is None:
                return self._defaults()

            file_header = lief_pe.FILE_HEADER
            optional_header = lief_pe.OPTIONAL_HEADER
            directories = list(getattr(optional_header, "DATA_DIRECTORY", []) or [])
            ep_rva = int(getattr(optional_header, "AddressOfEntryPoint", 0) or 0)
            checksum = int(getattr(optional_header, "CheckSum", 0) or 0)
            characteristics = int(getattr(file_header, "Characteristics", 0) or 0)
            dll_characteristics = int(getattr(optional_header, "DllCharacteristics", 0) or 0)
            now_utc = int(datetime.now(timezone.utc).timestamp())

            ep_section = self._find_ep_section(lief_pe, ep_rva)
            ep_section_name = decode_section_name(ep_section) if ep_section is not None else ""
            ep_offset = safe_rva_to_offset(lief_pe, ep_rva)
            ep_first_byte = pe_bytes[ep_offset] if ep_offset is not None and 0 <= ep_offset < len(pe_bytes) else 0
            ep_head = pe_bytes[ep_offset: ep_offset + 64] if ep_offset is not None and ep_offset >= 0 else b""

            ep_section_bytes = get_section_bytes(pe_bytes, ep_section) if ep_section is not None else b""
            ep_exec = False
            ep_write = False
            ep_virt_raw_ratio = 0.0
            if ep_section is not None:
                ep_exec, _, ep_write = section_flags(ep_section)
                raw_size = int(getattr(ep_section, "SizeOfRawData", 0) or 0)
                virt_size = int(getattr(ep_section, "Misc_VirtualSize", 0) or 0)
                ep_virt_raw_ratio = round(virt_size / max(raw_size, 1), 4)

            signature_info = (
                get_authenticode_signature_info(file_path)
                if file_path is not None
                else {"is_signed": False, "is_valid": False, "status": "unknown"}
            )

            has_signature = self._directory_has_data(
                directories,
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"],
            )
            if signature_info["is_signed"]:
                has_signature = True
            has_tls = self._directory_has_data(
                directories,
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_TLS"],
            )
            has_debug = self._directory_has_data(
                directories,
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DEBUG"],
            )
            has_resources = self._directory_has_data(
                directories,
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
            )
            has_relocations = self._directory_has_data(
                directories,
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_BASERELOC"],
            )
            is_dotnet = self._directory_has_data(
                directories,
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"],
            )

            rich_header_info = self._rich_header_info(pe_bytes, lief_pe)
            image_base = int(getattr(optional_header, "ImageBase", 0) or 0)
            machine_type = int(getattr(file_header, "Machine", 0) or 0)
            magic = int(getattr(optional_header, "Magic", 0) or 0)
            pe_bitness = 64 if magic == 0x20B else 32 if magic == 0x10B else 0
            flags = derive_pe_file_flags(characteristics, int(getattr(optional_header, "Subsystem", 0) or 0), file_path=file_path)
            is_dll = flags["is_dll"]
            is_driver = flags["is_driver"]
            expected_image_base = 0x10000000 if is_dll else 0x400000
            imagebase_is_nonstandard = False if is_driver else image_base not in {0, expected_image_base}

            return {
                "num_sections": len(getattr(lief_pe, "sections", [])),
                "machine_type": machine_type,
                "pe_bitness": pe_bitness,
                "size_of_image": int(getattr(optional_header, "SizeOfImage", 0) or 0),
                "size_of_headers": int(getattr(optional_header, "SizeOfHeaders", 0) or 0),
                "subsystem": int(getattr(optional_header, "Subsystem", 0) or 0),
                "dll_characteristics": dll_characteristics,
                "entrypoint_rva": ep_rva,
                "timestamp_raw": int(getattr(file_header, "TimeDateStamp", 0) or 0),
                "timestamp_is_zero": int(getattr(file_header, "TimeDateStamp", 0) or 0) == 0,
                "timestamp_is_future": int(getattr(file_header, "TimeDateStamp", 0) or 0) == 0 or int(getattr(file_header, "TimeDateStamp", 0) or 0) > now_utc,
                "checksum_is_zero": checksum == 0,
                "checksum_is_valid": self._checksum_is_valid(lief_pe, checksum),
                "imagebase": image_base,
                "imagebase_is_nonstandard": imagebase_is_nonstandard,
                "has_tls": has_tls,
                "has_debug": has_debug,
                "has_resources": has_resources,
                "has_relocations": has_relocations,
                "is_dotnet": is_dotnet,
                "has_signature": has_signature,
                "signature_is_valid": signature_info["is_valid"],
                "has_aslr": bool(dll_characteristics & DLLCHAR_DYNAMIC_BASE),
                "has_dep": bool(dll_characteristics & DLLCHAR_NX_COMPAT),
                "has_cfg": bool(dll_characteristics & DLLCHAR_GUARD_CF),
                "has_rich_header": rich_header_info["has_rich_header"],
                "num_rich_entries": rich_header_info["num_rich_entries"],
                "rich_header_is_zeroed": rich_header_info["rich_header_is_zeroed"],
                "dos_stub_nonstandard": self._dos_stub_nonstandard(pe_bytes, lief_pe),
                "ep_section_name": ep_section_name,
                "ep_is_in_last_section": ep_section is not None and ep_section == lief_pe.sections[-1],
                "ep_outside_sections": ep_section is None,
                "ep_starts_with_pushad": ep_first_byte == 0x60 or 0x60 in ep_head[:16],
                "ep_starts_with_jmp_near": ep_first_byte == 0xE9,
                "ep_starts_with_nop": ep_first_byte == 0x90,
                "ep_first_byte": ep_first_byte,
                "ep_first64_entropy": round(self._safe_entropy(ep_head), 4),
                "ep_section_entropy": round(self._safe_entropy(ep_section_bytes), 4),
                "ep_section_is_wx": ep_exec and ep_write,
                "ep_section_is_writable": ep_write,
                "ep_section_virt_raw_ratio": ep_virt_raw_ratio,
            }
        except Exception:
            return self._defaults()

    def _defaults(self) -> dict[str, Any]:
        return {
            "num_sections": 0,
            "machine_type": 0,
            "pe_bitness": 0,
            "size_of_image": 0,
            "size_of_headers": 0,
            "subsystem": 0,
            "dll_characteristics": 0,
            "entrypoint_rva": 0,
            "timestamp_raw": 0,
            "timestamp_is_zero": False,
            "timestamp_is_future": False,
            "checksum_is_zero": False,
            "checksum_is_valid": False,
            "imagebase": 0,
            "imagebase_is_nonstandard": False,
            "has_tls": False,
            "has_debug": False,
            "has_resources": False,
            "has_relocations": False,
            "is_dotnet": False,
            "has_signature": False,
            "signature_is_valid": False,
            "has_aslr": False,
            "has_dep": False,
            "has_cfg": False,
            "has_rich_header": False,
            "num_rich_entries": 0,
            "rich_header_is_zeroed": False,
            "dos_stub_nonstandard": False,
            "ep_section_name": "",
            "ep_is_in_last_section": False,
            "ep_outside_sections": False,
            "ep_starts_with_pushad": False,
            "ep_starts_with_jmp_near": False,
            "ep_starts_with_nop": False,
            "ep_first_byte": 0,
            "ep_first64_entropy": 0.0,
            "ep_section_entropy": 0.0,
            "ep_section_is_wx": False,
            "ep_section_is_writable": False,
            "ep_section_virt_raw_ratio": 0.0,
        }

    @staticmethod
    def _directory_has_data(directories: list[Any], index: int) -> bool:
        if index >= len(directories):
            return False
        directory = directories[index]
        return bool(getattr(directory, "VirtualAddress", 0) or getattr(directory, "Size", 0))

    @staticmethod
    def _find_ep_section(pe: Any, ep_rva: int) -> Any | None:
        for section in getattr(pe, "sections", []):
            if section_contains_rva(section, ep_rva):
                return section
        return None

    @staticmethod
    def _checksum_is_valid(pe: Any, checksum: int) -> bool:
        if checksum == 0:
            return False
        try:
            return int(pe.generate_checksum()) == checksum
        except Exception:
            return False

    @staticmethod
    def _dos_stub_nonstandard(pe_bytes: bytes, pe: Any) -> bool:
        e_lfanew = int(getattr(pe.DOS_HEADER, "e_lfanew", 0) or 0)
        if e_lfanew <= 64 or e_lfanew > len(pe_bytes):
            return False
        stub = pe_bytes[64:e_lfanew]
        if not stub:
            return False
        return b"This program cannot be run in DOS mode" not in stub

    @staticmethod
    def _rich_header_info(pe_bytes: bytes, pe: Any) -> dict[str, Any]:
        e_lfanew = int(getattr(pe.DOS_HEADER, "e_lfanew", 0) or 0)
        rich_region = pe_bytes[64:e_lfanew] if e_lfanew > 64 else b""
        info = {
            "has_rich_header": b"Rich" in rich_region,
            "num_rich_entries": 0,
            "rich_header_is_zeroed": False,
        }

        parse_rich = getattr(pe, "parse_rich_header", None)
        if not callable(parse_rich):
            return info

        try:
            rich_data = parse_rich() or {}
        except Exception:
            return info

        if not isinstance(rich_data, dict) or not rich_data:
            return info

        info["has_rich_header"] = True

        values = rich_data.get("values")
        if isinstance(values, list):
            info["num_rich_entries"] = len(values) // 2 if values and isinstance(values[0], int) else len(values)

        clear_data = rich_data.get("clear_data")
        if isinstance(clear_data, list) and clear_data:
            info["rich_header_is_zeroed"] = all(int(item) == 0 for item in clear_data)

        return info
