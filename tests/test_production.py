"""
Integration Tests for Production NEXUS
=======================================

Tests for production-ready features:
- Tokenization
- Checkpointing
- Metrics
- Error recovery
- Memory management
"""

import os
import tempfile
import time
from pathlib import Path

import pytest
import torch

from nexus.core.living import create_living_nexus
from nexus.core.tokenizer import NEXUSTokenizer
from nexus.service.checkpoint import CheckpointManager, CheckpointMetadata
from nexus.service.memory_manager import MemoryManager
from nexus.service.metrics import HealthCheck, MetricsCollector
from nexus.service.resilience import CircuitBreaker, RetryWithBackoff


class TestTokenization:
    """Test real tokenization."""

    def test_tokenizer_initialization(self):
        """Test tokenizer loads successfully."""
        tokenizer = NEXUSTokenizer(model_name="gpt2")

        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token_id is not None
        assert tokenizer.eos_token_id is not None

    def test_encode_decode(self):
        """Test encoding and decoding text."""
        tokenizer = NEXUSTokenizer(model_name="gpt2")

        text = "Hello, NEXUS!"
        encoded = tokenizer.encode(text)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.dim() == 1
        assert len(encoded) > 0

        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_batch_encode_decode(self):
        """Test batch encoding."""
        tokenizer = NEXUSTokenizer(model_name="gpt2")

        texts = ["Hello", "World", "NEXUS"]
        batch = tokenizer.batch_encode(texts, padding=True)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 3

        decoded = tokenizer.batch_decode(batch["input_ids"])
        assert len(decoded) == 3

    def test_special_tokens(self):
        """Test NEXUS special tokens."""
        tokenizer = NEXUSTokenizer(model_name="gpt2")

        # Check special tokens exist
        uncertain_id = tokenizer.get_special_token_id("uncertain_token")
        refuse_id = tokenizer.get_special_token_id("refuse_token")

        assert uncertain_id is not None
        assert refuse_id is not None

    def test_refusal_response(self):
        """Test refusal response generation."""
        tokenizer = NEXUSTokenizer(model_name="gpt2")

        refusal = tokenizer.create_refusal_response("uncertainty")

        assert isinstance(refusal, torch.Tensor)
        assert len(refusal) > 0


class TestCheckpointing:
    """Test checkpoint persistence."""

    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manager
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            # Create model
            nexus = create_living_nexus(size="small", architecture="flowing")

            # Save checkpoint
            path = manager.save_checkpoint(nexus)

            assert path.exists()
            assert path.suffix == ".pt"

            # Load checkpoint
            loaded = manager.load_checkpoint(path)

            assert "model_state_dict" in loaded
            assert "lifecycle_state" in loaded
            assert "metadata" in loaded

    def test_checkpoint_rotation(self):
        """Test automatic checkpoint rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir, max_checkpoints=3)

            nexus = create_living_nexus(size="small", architecture="flowing")

            # Save multiple checkpoints
            for i in range(5):
                manager.save_checkpoint(nexus)
                time.sleep(0.1)  # Ensure different timestamps

            # Should only keep 3
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) <= 3

    def test_checkpoint_metadata(self):
        """Test checkpoint metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            nexus = create_living_nexus(size="small", architecture="flowing")

            metadata = CheckpointMetadata(
                checkpoint_id="test_001",
                timestamp=time.time(),
                total_interactions=100,
                architecture="flowing",
            )

            path = manager.save_checkpoint(nexus, metadata)
            loaded = manager.load_checkpoint(path)

            assert loaded["metadata"]["total_interactions"] == 100
            assert loaded["metadata"]["architecture"] == "flowing"


class TestMetrics:
    """Test metrics collection."""

    def test_metrics_initialization(self):
        """Test metrics collector initializes."""
        metrics = MetricsCollector()

        assert metrics.total_requests == 0
        assert metrics.total_responses == 0

    def test_record_request(self):
        """Test recording requests."""
        metrics = MetricsCollector()

        metrics.record_request(
            latency=0.15,
            responded=True,
            confidence=0.8,
            flow_depth=12,
            converged=True,
        )

        assert metrics.total_requests == 1
        assert metrics.total_responses == 1

        summary = metrics.get_summary()
        assert summary["requests"]["total"] == 1

    def test_latency_percentiles(self):
        """Test latency percentile calculation."""
        metrics = MetricsCollector()

        # Record 100 requests with varying latencies
        for i in range(100):
            metrics.record_request(latency=i / 1000.0, responded=True)

        summary = metrics.get_summary()

        assert "latency" in summary
        assert summary["latency"]["request"]["p50_ms"] > 0
        assert summary["latency"]["request"]["p95_ms"] > 0

    def test_health_check(self):
        """Test health check logic."""
        metrics = MetricsCollector()
        health_check = HealthCheck()

        # Record successful requests
        for _ in range(10):
            metrics.record_request(latency=0.1, responded=True, error=False)

        health = health_check.check_health(metrics)

        assert health["healthy"] is True
        assert health["status"] == "healthy"


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    def test_circuit_breaker_closed(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker("test")

        # Should allow calls
        result = breaker.call(lambda: "success")
        assert result == "success"

        status = breaker.get_status()
        assert status["state"] == "closed"

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failures."""
        breaker = CircuitBreaker("test")

        # Cause failures
        for i in range(6):
            try:
                breaker.call(lambda: 1 / 0)  # Division by zero
            except ZeroDivisionError:
                pass

        # Should be open now
        status = breaker.get_status()
        assert status["state"] == "open"

        # Should reject calls
        from nexus.service.resilience import CircuitBreakerOpenError

        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(lambda: "should fail")

    def test_retry_with_backoff(self):
        """Test retry mechanism."""
        retry = RetryWithBackoff()

        call_count = 0

        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Fail")
            return "Success"

        result = retry.execute(flaky_function)

        assert result == "Success"
        assert call_count == 3  # Failed twice, succeeded third time


class TestMemoryManagement:
    """Test memory management."""

    def test_memory_manager_initialization(self):
        """Test memory manager initializes."""
        manager = MemoryManager()

        assert manager.config is not None
        assert manager.process is not None

    def test_periodic_cleanup(self):
        """Test periodic cleanup runs."""
        manager = MemoryManager()

        stats = manager.periodic_cleanup()

        assert isinstance(stats, dict)

    def test_replay_buffer_cleanup(self):
        """Test replay buffer cleanup."""
        manager = MemoryManager()

        # Create large buffer
        buffer = list(range(3000))

        stats = manager.cleanup_replay_buffer(buffer)

        assert stats["cleaned"] > 0
        assert len(buffer) < 3000

    def test_trim_history(self):
        """Test history trimming."""
        manager = MemoryManager()

        history = list(range(200))

        removed = manager.trim_history(history, max_size=50)

        assert removed == 150
        assert len(history) == 50


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_request_cycle(self):
        """Test complete request cycle with all components."""
        # Create tokenizer
        tokenizer = NEXUSTokenizer(model_name="gpt2")

        # Create model
        nexus = create_living_nexus(size="small", architecture="flowing")

        # Create metrics
        metrics = MetricsCollector()

        # Process request
        prompt = "What is Python?"
        input_ids = tokenizer.encode(prompt, max_length=32)
        batch = {"input_ids": input_ids.unsqueeze(0)}

        start_time = time.time()
        result = nexus.interact(batch, learn=False)
        latency = time.time() - start_time

        # Record metrics
        metrics.record_request(
            latency=latency,
            responded=result.responded,
            confidence=result.confidence,
            flow_depth=result.flow_depth,
        )

        # Verify
        assert result.logits is not None
        assert metrics.total_requests == 1

    def test_checkpoint_recovery(self):
        """Test saving and recovering from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and use model
            nexus1 = create_living_nexus(size="small", architecture="flowing")

            tokenizer = NEXUSTokenizer(model_name="gpt2")
            prompt = "Test prompt"
            input_ids = tokenizer.encode(prompt)
            batch = {"input_ids": input_ids.unsqueeze(0)}

            # Do some interactions
            for _ in range(10):
                nexus1.interact(batch, learn=True)

            status_before = nexus1.get_status()

            # Save checkpoint
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            path = manager.save_checkpoint(nexus1)

            # Load into new model
            checkpoint = manager.load_checkpoint(path)

            nexus2 = create_living_nexus(size="small", architecture="flowing")

            # Restore state
            nexus2.model.load_state_dict(checkpoint["model_state_dict"])
            nexus2.uncertainty_gate.load_state_dict(
                checkpoint["uncertainty_gate_state_dict"]
            )
            nexus2.refusal_generator.load_state_dict(
                checkpoint["refusal_generator_state_dict"]
            )
            nexus2.lifecycle.load_state(checkpoint["lifecycle_state"])

            status_after = nexus2.get_status()

            # Verify experience was preserved
            assert (
                status_after["total_interactions"]
                >= status_before["total_interactions"]
            )

    def test_memory_stability(self):
        """Test memory doesn't grow unbounded."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        nexus = create_living_nexus(size="small", architecture="flowing")
        tokenizer = NEXUSTokenizer(model_name="gpt2")
        manager = MemoryManager()

        # Simulate many requests
        for i in range(100):
            prompt = f"Test prompt {i}"
            input_ids = tokenizer.encode(prompt, max_length=16)
            batch = {"input_ids": input_ids.unsqueeze(0)}

            nexus.interact(batch, learn=True)

            # Periodic cleanup
            if i % 10 == 0:
                manager.periodic_cleanup()

        final_memory = process.memory_info().rss / 1024 / 1024

        # Memory shouldn't grow more than 200%
        memory_growth = (final_memory - initial_memory) / initial_memory

        assert memory_growth < 2.0, f"Memory grew {memory_growth:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
