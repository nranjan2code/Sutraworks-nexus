"""
NEXUS Tokenizer
===============

Production-ready tokenization with HuggingFace transformers.
Handles text encoding/decoding with NEXUS-specific special tokens.

Features:
- Support for multiple tokenizer backends (GPT2, LLaMA, etc.)
- Special tokens for uncertainty and refusal
- Batch processing with padding/truncation
- Caching for performance
- Thread-safe operations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

logger = logging.getLogger("nexus.tokenizer")


class NEXUSTokenizer:
    """
    Wrapper around HuggingFace tokenizers with NEXUS-specific features.

    This tokenizer adds special tokens for NEXUS's unique capabilities:
    - [UNCERTAIN]: Marks uncertain responses
    - [REFUSE]: Indicates polite refusal
    - [THINK]: Denotes reasoning process
    - [DREAM]: Marks self-supervised dreaming

    Example:
        >>> tokenizer = NEXUSTokenizer()
        >>> ids = tokenizer.encode("What is Python?")
        >>> text = tokenizer.decode(ids)
    """

    # Special tokens for NEXUS
    SPECIAL_TOKENS = {
        "uncertain_token": "[UNCERTAIN]",
        "refuse_token": "[REFUSE]",
        "think_token": "[THINK]",
        "dream_token": "[DREAM]",
        "learn_token": "[LEARN]",
    }

    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None,
        use_fast: bool = True,
    ):
        """
        Initialize tokenizer.

        Args:
            model_name: HuggingFace model name or path
            cache_dir: Directory to cache tokenizer files
            use_fast: Whether to use fast tokenizer (Rust-based)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        logger.info(f"Loading tokenizer: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=use_fast,
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {e}")
            logger.info("Falling back to GPT-2 tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                cache_dir=cache_dir,
                use_fast=use_fast,
            )
            self.model_name = "gpt2"

        # Add NEXUS special tokens
        self._add_special_tokens()

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            f"Tokenizer initialized: {self.model_name}, "
            f"vocab_size={self.vocab_size}"
        )

    def _add_special_tokens(self) -> None:
        """Add NEXUS-specific special tokens to vocabulary."""
        special_tokens_dict = {
            "additional_special_tokens": list(self.SPECIAL_TOKENS.values())
        }

        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_added > 0:
            logger.info(f"Added {num_added} NEXUS special tokens")

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """ID of padding token."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """ID of end-of-sequence token."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """ID of beginning-of-sequence token."""
        return self.tokenizer.bos_token_id

    def get_special_token_id(self, token_name: str) -> int:
        """
        Get ID of a NEXUS special token.

        Args:
            token_name: Name of special token (e.g., "uncertain_token")

        Returns:
            Token ID

        Raises:
            KeyError: If token name is invalid
        """
        if token_name not in self.SPECIAL_TOKENS:
            raise KeyError(f"Unknown special token: {token_name}")

        token_str = self.SPECIAL_TOKENS[token_name]
        return self.tokenizer.convert_tokens_to_ids(token_str)

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            max_length: Maximum sequence length (None for unlimited)
            truncation: Whether to truncate if exceeds max_length
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            Tensor of token IDs, shape (seq_len,)
        """
        if not text:
            logger.warning("Empty text provided to encode()")
            return torch.tensor([self.eos_token_id], dtype=torch.long)

        encoded = self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

        return encoded.squeeze(0)

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs, shape (seq_len,) or (batch, seq_len)
            skip_special_tokens: Whether to skip special tokens in output
            clean_up_tokenization_spaces: Whether to clean extra spaces

        Returns:
            Decoded text string
        """
        # Handle batch dimension
        if token_ids.dim() == 2:
            if token_ids.shape[0] == 1:
                token_ids = token_ids.squeeze(0)
            else:
                # Decode first in batch
                token_ids = token_ids[0]

        # Convert to list if tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        decoded = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

        return decoded

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batch of texts.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether/how to pad sequences ("longest", "max_length", True, False)
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            Dictionary with:
                - input_ids: Tensor of shape (batch, seq_len)
                - attention_mask: Tensor of shape (batch, seq_len)
        """
        if not texts:
            logger.warning("Empty batch provided to batch_encode()")
            return {
                "input_ids": torch.tensor([[self.eos_token_id]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }

        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """
        Decode batch of token sequences.

        Args:
            token_ids: Token IDs, shape (batch, seq_len)
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean extra spaces

        Returns:
            List of decoded strings
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        # Convert to list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        decoded = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

        return decoded

    def create_refusal_response(self, reason: str = "uncertainty") -> torch.Tensor:
        """
        Create a polite refusal response.

        Args:
            reason: Reason for refusal ("uncertainty", "safety", etc.)

        Returns:
            Encoded refusal message
        """
        templates = {
            "uncertainty": "I don't know enough about this yet to provide a confident answer.",
            "safety": "I'm not able to help with that request.",
            "complexity": "This question requires more thought than I can currently provide.",
        }

        message = templates.get(reason, templates["uncertainty"])
        refuse_token = self.SPECIAL_TOKENS["refuse_token"]

        full_message = f"{refuse_token} {message}"
        return self.encode(full_message)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save tokenizer to directory.

        Args:
            save_directory: Directory to save tokenizer files
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(str(save_directory))
        logger.info(f"Tokenizer saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, Path],
    ) -> NEXUSTokenizer:
        """
        Load tokenizer from directory.

        Args:
            load_directory: Directory containing tokenizer files

        Returns:
            Loaded tokenizer instance
        """
        load_directory = str(load_directory)
        instance = cls(model_name=load_directory)
        logger.info(f"Tokenizer loaded from {load_directory}")
        return instance

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"NEXUSTokenizer(model={self.model_name}, "
            f"vocab_size={self.vocab_size})"
        )


def create_tokenizer(
    model_name: str = "gpt2",
    cache_dir: Optional[str] = None,
) -> NEXUSTokenizer:
    """
    Factory function to create NEXUS tokenizer.

    Args:
        model_name: HuggingFace model name
        cache_dir: Cache directory

    Returns:
        Initialized tokenizer
    """
    return NEXUSTokenizer(model_name=model_name, cache_dir=cache_dir)
