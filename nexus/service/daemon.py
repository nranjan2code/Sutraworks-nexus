"""
Nexus Daemon - Production Version
==================================

Production-grade daemon for NEXUS Continuum.
Manages Living NEXUS with full production features:
- Real tokenization
- Checkpoint persistence
- Comprehensive metrics
- Error recovery with circuit breakers
- Memory management
- Resource governance

This is the heartbeat of ever-running, ever-evolving NEXUS.
"""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from nexus.core.living import LivingNEXUS, create_living_nexus
from nexus.core.flowing import DynamicsDivergenceError
from nexus.core.tokenizer import NEXUSTokenizer
from nexus.service.checkpoint import CheckpointManager, CheckpointMetadata
from nexus.service.memory_manager import MemoryConfig, MemoryManager
from nexus.service.metrics import HealthCheck, MetricsCollector
from nexus.service.resilience import CircuitBreaker, GracefulDegradation
from nexus.service.resource import ResourceGovernor, ResourceExhaustedError, ThermalThrottlingError
from nexus.training.teacher import GeminiTeacher

# Use centralized logging and memory utils
from nexus.service.logging_config import get_logger
from nexus.service.memory_utils import cleanup_gpu_memory, memory_cleanup_context

logger = get_logger("daemon")


class NexusDaemon:
    """
    Production-grade NEXUS daemon.

    Features:
    - Continuous operation with checkpointing
    - Real text processing with tokenization
    - Comprehensive monitoring and metrics
    - Error recovery and graceful degradation
    - Memory management for long-running processes
    - Resource governance

    Example:
        >>> daemon = NexusDaemon()
        >>> daemon.startup()
        >>> response = daemon.submit_request("What is Python?")
        >>> daemon.shutdown()
    """

    def __init__(
        self,
        checkpoint_dir: str = "./nexus_checkpoints",
        check_interval: float = 1.0,  # 1 second for Pi-friendly operation
        tokenizer_model: str = "gpt2",
    ):
        """
        Initialize daemon.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            check_interval: Loop interval in seconds
            tokenizer_model: HuggingFace tokenizer model name
        """
        self.check_interval = check_interval
        self.checkpoint_dir = Path(checkpoint_dir)

        # State
        self.running = False
        self.paused = False

        # Core components
        self.nexus: Optional[LivingNEXUS] = None
        self.tokenizer: Optional[NEXUSTokenizer] = None

        # Infrastructure
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=10,
        )
        self.metrics = MetricsCollector(window_size=1000)
        self.health_check = HealthCheck()
        self.memory_manager = MemoryManager()
        self.resource_governor = ResourceGovernor()

        # Circuit breakers
        self.inference_breaker = CircuitBreaker("inference")
        self.learning_breaker = CircuitBreaker("learning")

        # Teacher (optional)
        self.teacher = GeminiTeacher()
        self.training_mode = False
        self.training_topic: Optional[str] = None

        # Communication
        self.request_queue = queue.Queue()
        self.latest_thoughts: List[str] = []

        # Timing
        self.last_checkpoint_time = 0.0
        self.checkpoint_interval = 300.0  # 5 minutes

        # Threading
        self.thread = threading.Thread(target=self._daemon_loop, daemon=True)

        logger.info("NexusDaemon initialized")

    def startup(self) -> None:
        """Initialize and start the daemon."""
        logger.info("=" * 60)
        logger.info("Starting NEXUS Continuum Daemon")
        logger.info("=" * 60)

        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = NEXUSTokenizer(model_name="gpt2")

        # Try to load from checkpoint
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()

        if checkpoint:
            logger.info("Resuming from checkpoint...")
            self.nexus = self._load_from_checkpoint(checkpoint)
        else:
            logger.info("No checkpoint found. Starting fresh...")
            self.nexus = create_living_nexus(
                size="small",
                architecture="flowing",
                start_fresh=True,
                vocab_size=self.tokenizer.vocab_size,
            )

        # Start daemon thread
        self.running = True
        self.thread.start()

        logger.info("NEXUS Daemon started successfully")
        logger.info("Status: RUNNING")
        logger.info("=" * 60)

    def shutdown(self) -> None:
        """Stop the daemon gracefully."""
        logger.info("=" * 60)
        logger.info("Shutting down NEXUS Daemon...")
        logger.info("=" * 60)

        self.running = False

        # Save final checkpoint
        if self.nexus:
            logger.info("Saving final checkpoint...")
            try:
                self._save_checkpoint()
            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")

        # Wait for thread
        if self.thread.is_alive():
            logger.info("Waiting for daemon thread...")
            self.thread.join(timeout=10.0)

        logger.info("NEXUS Daemon stopped")
        logger.info("=" * 60)

    def pause(self) -> None:
        """Pause background learning (keep serving requests)."""
        logger.info("Pausing background evolution")
        self.paused = True

    def resume(self) -> None:
        """Resume background learning."""
        logger.info("Resuming background evolution")
        self.paused = False

    def submit_request(self, prompt: str, timeout: float = 30.0) -> str:
        """
        Submit a request to NEXUS.

        Args:
            prompt: User prompt text
            timeout: Maximum wait time for response

        Returns:
            Response text

        Raises:
            TimeoutError: If request times out
        """
        req = {"prompt": prompt, "result_queue": queue.Queue()}
        self.request_queue.put(req)

        try:
            result = req["result_queue"].get(timeout=timeout)
            return result
        except queue.Empty:
            raise TimeoutError(f"Request timed out after {timeout}s")

    def set_training_mode(self, enabled: bool, topic: Optional[str] = None) -> None:
        """
        Enable/disable focused training mode.

        Args:
            enabled: Whether to enable training mode
            topic: Optional topic to focus on
        """
        self.training_mode = enabled
        self.training_topic = topic if enabled else None

        if topic:
            logger.info(f"Training mode: {enabled} (Topic: {topic})")
        else:
            logger.info(f"Training mode: {enabled}")

    def reload_teacher(self) -> None:
        """Reload teacher configuration from environment."""
        logger.info("Reloading Gemini Teacher configuration...")
        self.teacher = GeminiTeacher()
        logger.info(f"Teacher reloaded: {self.teacher.model}")

    def _daemon_loop(self) -> None:
        """Main event loop."""
        logger.info("Daemon loop started")

        while self.running:
            try:
                # 1. Resource governance
                self.resource_governor.check_and_throttle()

                # 2. Memory management
                self.memory_manager.periodic_cleanup()
                self._trim_histories()

                # 3. Leak detection (periodic)
                if self.metrics.total_requests % 100 == 0:
                    leak_info = self.memory_manager.detect_memory_leak()
                    if leak_info and leak_info["detected"]:
                        logger.warning(f"Memory leak detected: {leak_info}")

                # 4. Handle user requests (high priority)
                try:
                    request = self.request_queue.get_nowait()
                    self._handle_request(request)
                    continue  # Process next request immediately
                except queue.Empty:
                    pass

                # 5. Checkpoint (periodic)
                if time.time() - self.last_checkpoint_time > self.checkpoint_interval:
                    self._save_checkpoint()

                # 6. Background learning (low priority)
                if not self.paused:
                    self.resource_governor.set_mode("idle")  # Use idle limits during background learning
                    self._dream_and_learn()
                    time.sleep(self.check_interval)  # Rate limit the loop
                else:
                    time.sleep(self.check_interval)

            except ThermalThrottlingError as e:
                logger.critical(f"THERMAL CRITICAL: {e}. Pausing for cooldown...")
                time.sleep(30.0)  # Wait 30 seconds for temperature to drop
                continue
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}", exc_info=True)
                time.sleep(1.0)  # Back off on errors

    def _handle_request(self, request: Dict[str, Any]) -> None:
        """Handle user request with full production features."""
        self.resource_governor.set_mode("active")

        prompt = request["prompt"]
        result_queue = request["result_queue"]

        start_time = time.time()

        try:
            logger.info(f"Processing request: {prompt[:50]}...")

            # Tokenize input (REAL tokenization!)
            input_ids = self.tokenizer.encode(prompt, max_length=512)
            batch = {"input_ids": input_ids.unsqueeze(0)}

            # Tokenize input (REAL tokenization!)
            input_ids = self.tokenizer.encode(prompt, max_length=512)
            batch = {"input_ids": input_ids.unsqueeze(0)}

            # Disable gradient for Flowing architecture (no backward pass yet)
            # This prevents memory leak from graph building
            use_grad = self.nexus.architecture != "flowing"

            # Process through circuit breaker
            def inference():
                with torch.set_grad_enabled(use_grad):
                    return self.nexus.interact(
                        batch,
                        learn=use_grad,  # Only learn if we have a backward pass mechanism
                        domain="user_interaction",
                        step_callback=lambda: self.resource_governor.check_and_throttle(),
                    )

            result = self.inference_breaker.call(inference)

            # Decode response
            if result.responded:
                # Get most likely tokens
                predicted_ids = result.logits.argmax(dim=-1)
                response_text = self.tokenizer.decode(predicted_ids)
            else:
                response_text = self.tokenizer.create_refusal_response("uncertainty")
                response_text = self.tokenizer.decode(response_text)

            # Record metrics
            latency = time.time() - start_time
            self.metrics.record_request(
                latency=latency,
                responded=result.responded,
                confidence=result.confidence,
                flow_depth=result.flow_depth,
                converged=result.converged,
                flow_energy=result.flow_energy,
                domain="user_interaction",
            )

            # Log thought
            self._log_thought(
                f"Q: {prompt[:40]}... | "
                f"Confidence: {result.confidence:.2f} | "
                f"Depth: {result.flow_depth} | "
                f"Latency: {latency*1000:.0f}ms"
            )

            result_queue.put(response_text)

        except DynamicsDivergenceError as e:
            logger.critical(f"MODEL DIVERGENCE DETECTED: {e}")
            logger.critical("Circuit breaker should open to prevent further bad inference.")

            # Record error (fail hard)
            latency = time.time() - start_time
            self.metrics.record_request(latency=latency, responded=False, error=True)

            # Graceful degradation with specific message
            fallback = GracefulDegradation.fallback_response("system_unavailable")
            result_queue.put("System Error: Neural dynamics diverged. Protection system activated.")

        except ResourceExhaustedError as e:
            logger.critical(f"RESOURCE EXHAUSTED: {e}")
            self.metrics.record_request(
                latency=time.time() - start_time, responded=False, error=True
            )
            result_queue.put(
                "System Error: Resource limits exceeded. Operation aborted for safety."
            )

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)

            # Record error
            latency = time.time() - start_time
            self.metrics.record_request(latency=latency, responded=False, error=True)

            # Graceful degradation
            fallback = GracefulDegradation.fallback_response("system_unavailable")
            result_queue.put(fallback["message"])

    def _dream_and_learn(self) -> None:
        """Background learning with teacher or self-supervised."""
        time.sleep(0.5)  # Throttle

        try:
            # Teacher-student learning
            if self.training_mode or (torch.rand(1).item() < 0.3 and self.teacher.available):
                self._learn_from_teacher(self.training_topic)
                return

            # Self-supervised dreaming
            dream_text = "[DREAM] Exploring latent space..."
            dream_ids = self.tokenizer.encode(dream_text, max_length=16)
            # Self-supervised dreaming
            dream_text = "[DREAM] Exploring latent space..."
            dream_ids = self.tokenizer.encode(dream_text, max_length=16)
            batch = {"input_ids": dream_ids.unsqueeze(0)}

            # Disable gradient for Flowing architecture
            use_grad = self.nexus.architecture != "flowing"

            start_time = time.time()

            def learning():
                with torch.set_grad_enabled(use_grad):
                    return self.nexus.interact(batch, learn=use_grad, domain="dreaming")

            result = self.learning_breaker.call(learning)

            if result.learned:
                latency = time.time() - start_time
                self.metrics.record_learning_cycle(
                    latency=latency,
                    num_samples=1,
                    loss=result.learning_metrics.get("total_loss"),
                )

                self._log_thought(
                    f"Dream: depth={result.flow_depth}, " f"energy={result.flow_energy:.4f}"
                )

        except Exception as e:
            logger.error(f"Error in dreaming: {e}")

    def _learn_from_teacher(self, topic: Optional[str] = None) -> None:
        """Learn from teacher model."""
        example = self.teacher.generate_synthetic_example(topic=topic)

        if not example:
            return

        prompt = example["prompt"]
        target = example["response"]

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, max_length=256)
        batch = {"input_ids": input_ids.unsqueeze(0)}

        start_time = time.time()

        try:
            result = self.nexus.interact(batch, learn=True, domain="teacher_distillation")

            latency = time.time() - start_time
            self.metrics.record_learning_cycle(latency=latency, num_samples=1)

            log_msg = f"Teacher: '{prompt[:40]}...'"
            if topic:
                log_msg = f"[{topic}] {log_msg}"

            self._log_thought(log_msg)

        except Exception as e:
            logger.error(f"Error learning from teacher: {e}")

    def _save_checkpoint(self) -> None:
        """Save checkpoint with metadata."""
        try:
            logger.info("Saving checkpoint...")

            metadata = CheckpointMetadata(
                checkpoint_id="",  # Will be set by manager
                timestamp=time.time(),
                architecture=self.nexus.architecture,
                total_interactions=self.metrics.total_requests,
                total_learning_steps=self.metrics.total_learning_cycles,
                total_responses=self.metrics.total_responses,
                total_refusals=self.metrics.total_refusals,
            )

            path = self.checkpoint_manager.save_checkpoint(self.nexus, metadata)

            self.last_checkpoint_time = time.time()

            # Clean up GPU memory after checkpoint (serialization can allocate temp buffers)
            cleanup_gpu_memory(verbose=False)

            logger.info(f"Checkpoint saved: {path.name}")

        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}", exc_info=True)

    def _load_from_checkpoint(self, checkpoint: Dict[str, Any]) -> LivingNEXUS:
        """Load NEXUS from checkpoint."""
        # Extract metadata
        metadata = checkpoint.get("metadata", {})

        logger.info(f"Loading checkpoint: {metadata.get('checkpoint_id')}")
        logger.info(f"  Interactions: {metadata.get('total_interactions', 0)}")
        logger.info(f"  Architecture: {metadata.get('architecture', 'unknown')}")

        # Create fresh model with correct vocab size
        nexus = create_living_nexus(
            size="small",
            architecture=metadata.get("architecture", "flowing"),
            start_fresh=True,
            vocab_size=self.tokenizer.vocab_size,
        )

        # Load states
        nexus.model.load_state_dict(checkpoint["model_state_dict"])
        nexus.uncertainty_gate.load_state_dict(checkpoint["uncertainty_gate_state_dict"])
        nexus.refusal_generator.load_state_dict(checkpoint["refusal_generator_state_dict"])
        nexus.lifecycle.load_state(checkpoint["lifecycle_state"])

        if checkpoint.get("learner_state") and nexus.learner:
            nexus.learner.optimizer.load_state_dict(
                checkpoint["learner_state"]["optimizer_state_dict"]
            )
            nexus.learner.update_step = checkpoint["learner_state"]["update_step"]

        logger.info("Checkpoint loaded successfully")

        return nexus

    def _trim_histories(self) -> None:
        """Trim history lists to prevent unbounded growth."""
        # Trim thoughts
        if len(self.latest_thoughts) > 100:
            removed = self.memory_manager.trim_history(self.latest_thoughts, 50)
            if removed > 0:
                logger.debug(f"Trimmed {removed} thoughts")

        # Trim replay buffer (if using layered architecture)
        if self.nexus.learner:
            stats = self.memory_manager.cleanup_replay_buffer(self.nexus.learner.replay_buffer)
            if stats["cleaned"] > 0:
                logger.info(f"Cleaned replay buffer: {stats['cleaned']} entries removed")

    def _log_thought(self, thought: str) -> None:
        """Log a thought to history."""
        timestamp = time.strftime("%H:%M:%S")
        self.latest_thoughts.append(f"[{timestamp}] {thought}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Get health status
        health = self.health_check.check_health(self.metrics)

        return {
            # Daemon status
            "daemon": {
                "running": self.running,
                "paused": self.paused,
                "uptime_seconds": self.metrics.get_summary()["uptime_seconds"],
            },
            # Health
            "health": health,
            # Model status
            "model": self.nexus.get_status() if self.nexus else {},
            # Metrics summary
            "metrics": self.metrics.get_summary(),
            # Memory stats
            "memory": self.memory_manager.get_memory_stats(),
            # Resources
            "resources": self.resource_governor.get_stats(),
            # Circuit breakers
            "circuit_breakers": {
                "inference": self.inference_breaker.get_status(),
                "learning": self.learning_breaker.get_status(),
            },
            # Checkpoints
            "checkpoints": self.checkpoint_manager.get_disk_usage(),
            # Recent thoughts
            "recent_thoughts": self.latest_thoughts[-10:],
        }

    def get_metrics_prometheus(self) -> str:
        """Get metrics in Prometheus format."""
        return self.metrics.get_prometheus_metrics()

    def __repr__(self) -> str:
        status = "running" if self.running else "stopped"
        return f"NexusDaemon(status={status}, requests={self.metrics.total_requests})"
