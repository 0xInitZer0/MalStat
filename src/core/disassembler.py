"""
src/core/disassembler.py

Thin wrapper around the capstone library.
It hides Cs initialization, error handling, and mode selection (x86-32 / x86-64).
It contains no feature business logic, only "give bytes, get mnemonics".

Dependencies:
    capstone >= 5.0.1  (pip install capstone)

Usage:
    from src.core.disassembler import Disassembler

    d = Disassembler(is_64bit=False)
    mnemonics = d.disasm(raw_bytes)          # ['push', 'mov', 'call', ...]
    details   = d.disasm_with_offsets(raw_bytes)  # [(addr, mnem, ops), ...]
"""

from __future__ import annotations

try:
    from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64, CS_OPT_SKIPDATA
    _CAPSTONE_AVAILABLE = True
except ImportError:
    _CAPSTONE_AVAILABLE = False


class Disassembler:
    """
    Wrapper around capstone for linear disassembly of x86/x64 bytes.

    Never raises exceptions to the caller. On any error,
    it returns an empty list.
    """

    MAX_BYTES = 524_288  # 512 KB limit for a single disasm() call

    def __init__(self, is_64bit: bool = False) -> None:
        """
        Args:
            is_64bit: True for CS_MODE_64, False for CS_MODE_32.
        """
        self._is_64bit = is_64bit
        self._cs: object | None = None  # initialized lazily on first use

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def disasm(self, raw_bytes: bytes, offset: int = 0) -> list[str]:
        """
        Disassemble raw_bytes linearly.

        Args:
            raw_bytes: bytes to disassemble, up to MAX_BYTES.
            offset: starting virtual address for capstone. Used only for
                    displayed addresses and does not affect the returned mnemonics.

        Returns:
            List of lowercase mnemonics, for example ['push', 'mov', 'xor'].
            Returns [] on error or when capstone is unavailable.
        """
        if not _CAPSTONE_AVAILABLE or not raw_bytes:
            return []
        try:
            cs = self._get_cs()
            chunk = raw_bytes[: self.MAX_BYTES]
            return [insn.mnemonic for insn in cs.disasm(chunk, offset)]
        except Exception:
            return []

    def disasm_with_offsets(
        self, raw_bytes: bytes, offset: int = 0
    ) -> list[tuple[int, str, str]]:
        """
        Return a list of tuples in the form (address, mnemonic, operands).

        Used for detailed entry-point pattern analysis, for example,
        to precisely detect PUSHAD -> ... -> POPAD -> JMP.

        Returns:
            [(addr, mnemonic, op_str), ...] or [] on error.
        """
        if not _CAPSTONE_AVAILABLE or not raw_bytes:
            return []
        try:
            cs = self._get_cs()
            chunk = raw_bytes[: self.MAX_BYTES]
            return [
                (insn.address, insn.mnemonic, insn.op_str)
                for insn in cs.disasm(chunk, offset)
            ]
        except Exception:
            return []

    def count_invalid_bytes(
        self,
        raw_bytes: bytes,
        instruction_sizes: list[int],
    ) -> int:
        """
        Count bytes that were not part of any valid instruction.

        Args:
            raw_bytes: original bytes.
            instruction_sizes: list of sizes in bytes for each disassembled
                               instruction (insn.size).

        Returns:
            Number of unmatched "junk" bytes (>= 0).
        """
        valid_bytes = sum(instruction_sizes)
        total = min(len(raw_bytes), self.MAX_BYTES)
        return max(0, total - valid_bytes)

    def disasm_with_sizes(
        self, raw_bytes: bytes, offset: int = 0
    ) -> tuple[list[str], list[int]]:
        """
        Disassemble and return instruction sizes at the same time.

        Returns:
            (mnemonics: list[str], sizes: list[int])
            Both lists have the same length.
        """
        if not _CAPSTONE_AVAILABLE or not raw_bytes:
            return [], []
        try:
            cs = self._get_cs()
            chunk = raw_bytes[: self.MAX_BYTES]
            mnemonics: list[str] = []
            sizes: list[int] = []
            for insn in cs.disasm(chunk, offset):
                mnemonics.append(insn.mnemonic)
                sizes.append(insn.size)
            return mnemonics, sizes
        except Exception:
            return [], []

    @property
    def is_available(self) -> bool:
        """True if capstone is installed and available."""
        return _CAPSTONE_AVAILABLE

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _get_cs(self) -> "Cs":
        """Lazily initialize the capstone Cs object."""
        if self._cs is None:
            mode = CS_MODE_64 if self._is_64bit else CS_MODE_32
            self._cs = Cs(CS_ARCH_X86, mode)
            self._cs.detail = False  # faster without full operand details
            self._cs.skipdata = True  # skip invalid bytes instead of stopping
        return self._cs
