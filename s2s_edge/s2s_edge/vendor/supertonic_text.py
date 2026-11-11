# Vendored from supertonic 1.3.1 (MIT License, (c) Supertone Inc.): text preprocessing only (no onnxruntime).
"""Core TTS engine and text processing components.

This module contains the main Supertonic TTS engine, text processor,
and supporting utilities for audio synthesis.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional
from unicodedata import normalize

import numpy as np

AVAILABLE_LANGUAGES = ['en', 'ko', 'ja', 'ar', 'bg', 'cs', 'da', 'de', 'el', 'es', 'et', 'fi', 'fr', 'hi', 'hr', 'hu', 'id', 'it', 'lt', 'lv', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'tr', 'uk', 'vi', 'na']
MAX_SPEED = 2.0
MIN_SPEED = 0.5

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for text processing performance optimization.
# These patterns are compiled once at module load time to avoid repeated compilation
# during text preprocessing operations.
_EMOJI_PATTERN = re.compile(
    "[\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)

_SYMBOL_REPLACEMENTS = {
    "\u2013": "-",  # EN DASH (–)
    "\u2011": "-",  # NON-BREAKING HYPHEN (‑)
    "\u2014": "-",  # EM DASH (—)
    "\u00af": " ",  # MACRON (¯)
    "_": " ",
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK (")
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK (")
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK (')
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK (')
    "\u00b4": "'",  # ACUTE ACCENT (´)
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}

_SPECIAL_SYMBOLS_PATTERN = re.compile(r"[♥☆♡©\\]")

_PUNCTUATION_SPACING_PATTERNS = [
    (re.compile(r" ,"), ","),
    (re.compile(r" \."), "."),
    (re.compile(r" !"), "!"),
    (re.compile(r" \?"), "?"),
    (re.compile(r" ;"), ";"),
    (re.compile(r" :"), ":"),
    (re.compile(r" '"), "'"),
]

_DUPLICATE_QUOTES_PATTERN = re.compile(r'(["\'\`])\1+')

_WHITESPACE_PATTERN = re.compile(r"\s+")

_ENDING_PUNCTUATION_PATTERN = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """
    Convert lengths to binary mask.

    Args:
        lengths: (B,)
        max_len: int

    Returns:
        mask: (B, 1, max_len)
    """
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    """Generate mask for latent representations."""
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask


class UnicodeProcessor:
    """Processes text into unicode indices for the TTS model.

    This class handles text preprocessing, normalization, and conversion to
    numeric indices that the TTS model can understand.

    Args:
        unicode_indexer_path: Path to the unicode indexer JSON file
    """

    def __init__(self, unicode_indexer_path: str):
        self.indexer = self._load_indexer(unicode_indexer_path)
        self.supported_chars = self._make_supported_characters()

    def _load_indexer(self, unicode_indexer_path: str) -> list:
        try:
            with open(unicode_indexer_path, "r") as f:
                indexer = json.load(f)

                # Validate indexer format
                if not isinstance(indexer, list):
                    raise ValueError(
                        f"Unicode indexer must be a list, got {type(indexer).__name__}"
                    )

                if len(indexer) == 0:
                    raise ValueError("Unicode indexer is empty")

                logger.info(
                    f"Loaded unicode indexer from {unicode_indexer_path} "
                    f"({len(indexer)} entries)"
                )
        except FileNotFoundError:
            logger.error(f"Unicode indexer not found: {unicode_indexer_path}")
            raise FileNotFoundError(
                f"Unicode indexer file not found at {unicode_indexer_path}. "
                f"Please ensure the model is properly downloaded."
            )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid unicode indexer format: {e}")
            raise ValueError(
                f"Unicode indexer file is malformed at {unicode_indexer_path}. "
                f"Please re-download the model."
            ) from e
        return indexer

    def _make_supported_characters(self) -> set[str]:
        supported = set()
        for unicode_value, char_dict_idx in enumerate(self.indexer):
            if char_dict_idx == -1:
                # Not supported by the model
                continue
            else:
                char = chr(unicode_value)
                supported.add(char)
        return supported

    @property
    def supported_character_set(self) -> set[str]:
        return self.supported_chars

    def _remove_emojis(self, text: str) -> str:
        """Remove emoji characters from text."""
        text = _EMOJI_PATTERN.sub("", text)
        return text

    def _normalize_symbols(self, text: str) -> str:
        """Normalize various punctuation marks and symbols to standard forms."""
        for old, new in _SYMBOL_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text

    def _remove_special_chars(self, text: str) -> str:
        """Remove unsupported decorative symbols while preserving diacritics."""
        text = _SPECIAL_SYMBOLS_PATTERN.sub("", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations and expressions to full text."""
        expr_replacements = {
            "@": " at ",
            "e.g.,": "for example, ",
            "i.e.,": "that is, ",
        }
        for k, v in expr_replacements.items():
            text = text.replace(k, v)
        return text

    def _fix_punctuation_spacing(self, text: str) -> str:
        """Fix spacing around punctuation marks."""
        # Fix spacing around punctuation using pre-compiled patterns
        for pattern, replacement in _PUNCTUATION_SPACING_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _remove_duplicate_quotes(self, text: str) -> str:
        """Remove duplicate quotation marks."""
        # Use pre-compiled regex to remove all consecutive duplicate quotes in one pass
        text = _DUPLICATE_QUOTES_PATTERN.sub(r"\1", text)
        return text

    def _clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace."""
        # Remove extra spaces using pre-compiled pattern
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()
        return text

    def _add_period_if_needed(self, text: str) -> str:
        # If text doesn't end with punctuation, quotes, or closing brackets, add a period
        if not _ENDING_PUNCTUATION_PATTERN.search(text):
            text += "."
        return text

    def _add_language_token(self, text: str, lang: str) -> str:
        """Add language tokens to text for multilingual model.

        Args:
            text: Preprocessed text
            lang: Language code from ``AVAILABLE_LANGUAGES`` (or ``"na"``
                for unknown / unsupported languages)

        Returns:
            Text wrapped with language tokens: ``<lang>text</lang>``
            (e.g., ``<en>...</en>`` or ``<na>...</na>``)
        """
        if lang not in AVAILABLE_LANGUAGES:
            raise ValueError(
                f"Invalid language: '{lang}'. "
                f"Supported languages: {', '.join(AVAILABLE_LANGUAGES)}"
            )
        return f"<{lang}>{text}</{lang}>"

    def _preprocess_text(self, text: str, lang: Optional[str] = None) -> str:
        """Preprocess text by normalizing, cleaning, and standardizing format.

        This method applies a series of text transformations in sequence:
        1. Unicode normalization (NFKD)
        2. Emoji removal
        3. Symbol normalization
        4. Decorative symbol removal
        5. Abbreviation expansion
        6. Punctuation spacing fixes
        7. Duplicate quote removal
        8. Whitespace cleaning
        9. Add period if needed
        10. Add language tokens (for multilingual models)

        Args:
            text: Raw input text
            lang: Language code for multilingual support. Supertonic-3
                supports 31 ISO codes plus the ``"na"`` fallback for
                unknown languages; Supertonic-2 supports
                ``en``, ``ko``, ``es``, ``pt``, ``fr``.
                If None, no language tokens are added (v1 compatibility).

        Returns:
            Preprocessed and normalized text
        """
        # Apply NFKD normalization with exception handling
        try:
            text = normalize("NFKD", text)
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}. Continuing without normalization.")

        text = self._remove_emojis(text)
        text = self._normalize_symbols(text)
        text = self._remove_special_chars(text)
        text = self._expand_abbreviations(text)
        text = self._fix_punctuation_spacing(text)
        text = self._remove_duplicate_quotes(text)
        text = self._clean_whitespace(text)
        text = self._add_period_if_needed(text)

        # Add language tokens for multilingual models (supertonic-2 / supertonic-3)
        if lang is not None:
            text = self._add_language_token(text, lang)

        return text

    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        text_mask = length_to_mask(text_ids_lengths)
        return text_mask

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        unicode_values = np.array([ord(char) for char in text], dtype=np.uint16)  # 2 bytes
        return unicode_values

    def validate_text(self, text: str) -> tuple[bool, list[str]]:
        """Validate if text can be processed by the model.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, unsupported_chars):
                - is_valid: True if text can be processed
                - unsupported_chars: List of unsupported characters (empty if valid)

        Example:
            ```python
            processor = UnicodeProcessor("unicode_indexer.json")
            is_valid, unsupported = processor.validate_text("Hello world")
            if not is_valid:
                print(f"Cannot process: {unsupported}")
            ```
        """
        input_chars = set(text)
        unsupported_chars = set()
        for input_char in input_chars:
            p_chars = set(self._preprocess_text(input_char))
            us_chars = p_chars - self.supported_character_set
            if len(us_chars) > 0:
                unsupported_chars.update(input_char)
        return len(unsupported_chars) == 0, sorted(list(unsupported_chars))

    def validate_text_list(self, text_list: list[str]) -> tuple[bool, list[str]]:
        """Validate a list of texts."""
        text_cat = "".join(text_list)
        return self.validate_text(text_cat)

    def __call__(
        self, text_list: list[str], lang: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a list of texts into model inputs.

        Args:
            text_list: List of text strings to process
            lang: Language code for multilingual support. Use any code from
                ``AVAILABLE_LANGUAGES`` (or ``"na"`` for unknown languages).
                If None, no language tokens are added (v1 compatibility).

        Returns:
            Tuple of (text_ids, text_mask):
                - text_ids: Array of shape (batch_size, max_length) with unicode indices
                - text_mask: Array of shape (batch_size, 1, max_length) with attention mask
        """
        preprocessed_texts = [self._preprocess_text(t, lang) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in preprocessed_texts], dtype=np.int64)
        text_ids = np.zeros((len(preprocessed_texts), text_ids_lengths.max()), dtype=np.int64)
        for i, text in enumerate(preprocessed_texts):
            unicode_vals = self._text_to_unicode_values(text)
            text_ids[i, : len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask

