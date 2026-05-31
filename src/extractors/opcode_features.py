"""Opcode-based static features extracted from PE bytes."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from src.extractors.base import BaseExtractor
from src.core.disassembler import Disassembler
from src.extractors.pe_helpers import decode_section_name, get_section_bytes, safe_rva_to_offset

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


# ---------------------------------------------------------------------------
# Opcode categories (lowercase mnemonics, as returned by capstone)
# ---------------------------------------------------------------------------

_OPCODE_CATEGORIES: dict[str, frozenset[str]] = {
    "nop": frozenset({"nop"}),
    "call": frozenset({"call", "lcall"}),
    "jmp_uncond": frozenset({"jmp", "ljmp", "jmpf"}),
    "jmp_cond": frozenset({
        "je", "jne", "jz", "jnz", "jg", "jge", "jl", "jle",
        "ja", "jae", "jb", "jbe", "js", "jns", "jc", "jnc",
        "jo", "jno", "jp", "jnp", "jpe", "jpo", "jcxz", "jecxz", "jrcxz",
    }),
    "ret": frozenset({"ret", "retn", "retf", "lret", "iret", "iretd", "iretq"}),
    "push": frozenset({"push", "pusha", "pushad", "pushf", "pushfd", "pushfq"}),
    "pop": frozenset({"pop", "popa", "popad", "popf", "popfd", "popfq"}),
    "mov": frozenset({
        "mov", "movsx", "movzx", "movs", "movsb", "movsw", "movsd",
        "movsq", "movabs", "movbe", "lea", "xchg",
    }),
    "arith": frozenset({
        "add", "sub", "mul", "imul", "div", "idiv", "inc", "dec",
        "adc", "sbb", "neg", "cbw", "cwd", "cdq", "cqo",
    }),
    "logic": frozenset({
        "and", "or", "not", "xor", "shl", "shr", "sar", "sal",
        "rol", "ror", "rcl", "rcr", "test", "bt", "bts", "btr",
        "btc", "bsf", "bsr",
    }),
    "xor": frozenset({"xor"}),
    "int_": frozenset({"int", "int3", "into", "syscall", "sysenter", "sysexit"}),
    "loop": frozenset({"loop", "loopne", "loope", "loopnz", "loopz"}),
}

# Code section names, in search priority order
_TEXT_SECTION_NAMES = {".text", "CODE", "code", ".code", "text"}

# Entry-point region size in bytes for packed PE files
_EP_REGION_SIZE = 512

# Maximum number of bytes from .text to disassemble
_MAX_TEXT_BYTES = 524_288  # 512 KB


class OpcodeExtractor(BaseExtractor):
    """Extracts opcode statistics from the EP region or .text section."""

    def __init__(self, registry: "IndicatorRegistry") -> None:
        super().__init__(registry)
        self._disasm_32 = Disassembler(is_64bit=False)
        self._disasm_64 = Disassembler(is_64bit=True)

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return list(self._defaults().keys())

    def extract(
        self,
        pe_bytes: bytes,
        lief_pe: object,
        file_path: str | Path | None = None,
    ) -> dict[str, float | int | bool]:
        try:
            return self._extract_impl(pe_bytes, lief_pe)
        except Exception:
            return self._defaults()

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _extract_impl(self, pe_bytes: bytes, pe: object | None) -> dict:
        if pe is None:
            return self._defaults()

        magic = int(getattr(pe.OPTIONAL_HEADER, "Magic", 0) or 0)
        is_64bit = magic == 0x20B
        disasm = self._disasm_64 if is_64bit else self._disasm_32

        raw_bytes = self._select_bytes(pe_bytes, pe)
        if not raw_bytes:
            return self._defaults()

        mnemonics, sizes = disasm.disasm_with_sizes(raw_bytes)
        details = disasm.disasm_with_offsets(raw_bytes)

        invalid_bytes = disasm.count_invalid_bytes(raw_bytes, sizes)
        total_raw = min(len(raw_bytes), Disassembler.MAX_BYTES)

        features: dict = {}
        features.update(self._basic_stats(mnemonics, total_raw, invalid_bytes))
        features.update(self._category_ratios(mnemonics))
        features.update(self._packer_signals(mnemonics, details))
        features.update(self._ngram_features(mnemonics))

        # Override asm_has_pushad_at_ep and asm_pushad_popad_count using the raw EP bytes.
        # capstone x64 skipdata treats 0x60/0x61 as .byte, so inspect the bytes directly.
        ep_rva = int(getattr(pe.OPTIONAL_HEADER, "AddressOfEntryPoint", 0) or 0)
        ep_offset = safe_rva_to_offset(pe, ep_rva)
        if ep_offset is not None and 0 <= ep_offset < len(pe_bytes):
            ep_region = pe_bytes[ep_offset: ep_offset + 16]
            features["asm_has_pushad_at_ep"] = 0x60 in ep_region
            raw_pushad_count = ep_region.count(0x60) + ep_region.count(0x61)
            if raw_pushad_count > features.get("asm_pushad_popad_count", 0):
                features["asm_pushad_popad_count"] = raw_pushad_count

        return features

    # ------------------------------------------------------------------
    # Selecting bytes for disassembly
    # ------------------------------------------------------------------

    def _select_bytes(self, pe_bytes: bytes, pe: object) -> bytes:
        section_names = [decode_section_name(section) for section in getattr(pe, "sections", [])]
        is_packed_by_name = any(
            self._registry.rule_matches(name, rule)
            for name in section_names
            for rule in self._registry.section_indicators
        )

        ep_rva = int(getattr(pe.OPTIONAL_HEADER, "AddressOfEntryPoint", 0) or 0)
        ep_offset = safe_rva_to_offset(pe, ep_rva)
        if is_packed_by_name and ep_offset is not None:
            # Use the EP region for signature detection but prefer .text for ratio analysis.
            # First check whether a .text section has enough data.
            for section in getattr(pe, "sections", []):
                if decode_section_name(section).strip("\x00") in _TEXT_SECTION_NAMES:
                    text_bytes = get_section_bytes(pe_bytes, section)[:_MAX_TEXT_BYTES]
                    if len(text_bytes) > 64:
                        return text_bytes
            return pe_bytes[ep_offset: ep_offset + _EP_REGION_SIZE]

        for section in getattr(pe, "sections", []):
            if decode_section_name(section).strip("\x00") in _TEXT_SECTION_NAMES:
                return get_section_bytes(pe_bytes, section)[:_MAX_TEXT_BYTES]

        if ep_offset is None:
            return b""
        return pe_bytes[ep_offset: ep_offset + _EP_REGION_SIZE]

    # ------------------------------------------------------------------
    # Basic statistics (5 features)
    # ------------------------------------------------------------------

    def _basic_stats(
        self,
        insns: list[str],
        total_raw_bytes: int,
        invalid_bytes: int,
    ) -> dict:
        n = len(insns)
        invalid_ratio = invalid_bytes / max(total_raw_bytes, 1)
        invalid_ratio = min(1.0, max(0.0, invalid_ratio))

        # Shannon entropy of the opcode distribution
        entropy = 0.0
        if n > 0:
            counts = Counter(insns)
            total = sum(counts.values())
            entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in counts.values()
                if c > 0
            )

        return {
            "asm_total_instructions": n,
            "asm_unique_opcodes": len(set(insns)),
            "asm_opcode_entropy": round(entropy, 4),
            "asm_invalid_ratio": round(invalid_ratio, 4),
            "asm_coverage_ratio": round(1.0 - invalid_ratio, 4),
        }

    # ------------------------------------------------------------------
    # Category frequencies (12 features)
    # ------------------------------------------------------------------

    def _category_ratios(self, insns: list[str]) -> dict:
        n = max(len(insns), 1)
        result = {}
        category_keys = [
            ("nop",       "asm_ratio_nop"),
            ("call",      "asm_ratio_call"),
            ("jmp_uncond","asm_ratio_jmp_uncond"),
            ("jmp_cond",  "asm_ratio_jmp_cond"),
            ("ret",       "asm_ratio_ret"),
            ("push",      "asm_ratio_push"),
            ("pop",       "asm_ratio_pop"),
            ("mov",       "asm_ratio_mov"),
            ("arith",     "asm_ratio_arith"),
            ("logic",     "asm_ratio_logic"),
            ("xor",       "asm_ratio_xor"),
            ("int_",      "asm_ratio_int"),
        ]
        for cat_key, feat_key in category_keys:
            cat_set = _OPCODE_CATEGORIES[cat_key]
            count = sum(1 for m in insns if m in cat_set)
            result[feat_key] = round(count / n, 4)
        return result

    # ------------------------------------------------------------------
    # Packer / obfuscation signals (7 features)
    # ------------------------------------------------------------------

    def _packer_signals(
        self,
        insns: list[str],
        details: list[tuple[int, str, str]],
    ) -> dict:
        n = max(len(insns), 1)

        # PUSHAD count
        pushad_count = sum(1 for m in insns if m == "pushad")

        # Maximum NOP sled length
        max_nop = 0
        cur_nop = 0
        for m in insns:
            if m == "nop":
                cur_nop += 1
                max_nop = max(max_nop, cur_nop)
            else:
                cur_nop = 0

        call_count = sum(1 for m in insns if m in _OPCODE_CATEGORIES["call"])
        indirect_calls = sum(
            1 for _, mnemonic, op_str in details
            if mnemonic == "call" and "[" in op_str
        )
        indirect_call_ratio = round(indirect_calls / max(call_count, 1), 4)

        # PUSHAD at the entry point (first 20 instructions)
        has_pushad_at_ep = "pushad" in insns[:20]

        # EP stub entropy (first 30 instructions)
        ep_stub = insns[:30]
        ep_entropy = 0.0
        if ep_stub:
            counts = Counter(ep_stub)
            total = sum(counts.values())
            ep_entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in counts.values() if c > 0
            )

        # CALL followed by POP
        call_pop = sum(
            1 for a, b in zip(insns, insns[1:])
            if a == "call" and b == "pop"
        )

        # Loop density
        loop_set = _OPCODE_CATEGORIES["loop"]
        loop_count = sum(1 for m in insns if m in loop_set)
        loop_density = round(loop_count / n, 4)

        return {
            "asm_pushad_popad_count": pushad_count,
            "asm_nop_sled_max_len": max_nop,
            "asm_indirect_call_ratio": indirect_call_ratio,
            "asm_has_pushad_at_ep": bool(has_pushad_at_ep),
            "asm_ep_stub_entropy": round(ep_entropy, 4),
            "asm_call_followed_by_pop": call_pop,
            "asm_loop_density": loop_density,
        }

    # ------------------------------------------------------------------
    # N-gram features (6 features)
    # ------------------------------------------------------------------

    def _ngram_features(self, insns: list[str]) -> dict:
        if len(insns) < 2:
            return {
                "asm_bigram_push_call": 0,
                "asm_bigram_xor_xor": 0,
                "asm_bigram_mov_push": 0,
                "asm_bigram_jmp_push": 0,
                "asm_top10_bigram_entropy": 0.0,
                "asm_unique_bigrams": 0,
            }

        bigrams = Counter(zip(insns, insns[1:]))
        total = sum(bigrams.values()) or 1

        # Top-10 bigram entropy
        top10 = bigrams.most_common(10)
        top10_probs = [v / total for _, v in top10]
        top10_entropy = -sum(
            p * math.log2(p) for p in top10_probs if p > 0
        )

        return {
            "asm_bigram_push_call": bigrams.get(("push", "call"), 0),
            "asm_bigram_xor_xor": bigrams.get(("xor", "xor"), 0),
            "asm_bigram_mov_push": bigrams.get(("mov", "push"), 0),
            "asm_bigram_jmp_push": bigrams.get(("jmp", "push"), 0),
            "asm_top10_bigram_entropy": round(top10_entropy, 4),
            "asm_unique_bigrams": len(bigrams),
        }

    # ------------------------------------------------------------------
    # Default values on failure
    # ------------------------------------------------------------------

    def _defaults(self) -> dict[str, float | int | bool]:
        return {
            # Basic statistics
            "asm_total_instructions": 0,
            "asm_unique_opcodes": 0,
            "asm_opcode_entropy": 0.0,
            "asm_invalid_ratio": 0.0,
            "asm_coverage_ratio": 0.0,
            # Category frequencies
            "asm_ratio_nop": 0.0,
            "asm_ratio_call": 0.0,
            "asm_ratio_jmp_uncond": 0.0,
            "asm_ratio_jmp_cond": 0.0,
            "asm_ratio_ret": 0.0,
            "asm_ratio_push": 0.0,
            "asm_ratio_pop": 0.0,
            "asm_ratio_mov": 0.0,
            "asm_ratio_arith": 0.0,
            "asm_ratio_logic": 0.0,
            "asm_ratio_xor": 0.0,
            "asm_ratio_int": 0.0,
            # Packer signals
            "asm_pushad_popad_count": 0,
            "asm_nop_sled_max_len": 0,
            "asm_indirect_call_ratio": 0.0,
            "asm_has_pushad_at_ep": False,
            "asm_ep_stub_entropy": 0.0,
            "asm_call_followed_by_pop": 0,
            "asm_loop_density": 0.0,
            # N-grams
            "asm_bigram_push_call": 0,
            "asm_bigram_xor_xor": 0,
            "asm_bigram_mov_push": 0,
            "asm_bigram_jmp_push": 0,
            "asm_top10_bigram_entropy": 0.0,
            "asm_unique_bigrams": 0,
        }
