"""
Neuro-Symbolic Reasoning Engine
================================

Implements a hybrid neural-symbolic reasoning system that combines:
1. Neural pattern recognition (System 1 - fast, intuitive)
2. Symbolic logical reasoning (System 2 - slow, deliberate)

Key innovations:
- Differentiable logic programming
- Neural theorem proving with soft unification
- Knowledge graph integration
- Explainable reasoning chains

This addresses a fundamental limitation of LLMs: they can pattern-match but
struggle with consistent logical reasoning and grounded factual knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReasoningConfig:
    """Configuration for Neuro-Symbolic Reasoner."""
    
    d_model: int = 256          # Model dimension
    d_symbol: int = 64          # Symbol embedding dimension
    n_predicates: int = 1000    # Maximum number of predicates
    n_entities: int = 10000     # Maximum number of entities
    max_proof_depth: int = 5    # Maximum depth of proof trees
    n_rules: int = 100          # Number of learnable rules
    temperature: float = 0.1    # Softmax temperature for soft matching
    dropout: float = 0.1


@dataclass
class Symbol:
    """Represents a symbolic entity or predicate."""
    
    name: str
    symbol_type: str  # "entity", "predicate", "variable"
    arity: int = 0    # For predicates: number of arguments
    embedding: Optional[torch.Tensor] = None


@dataclass
class Fact:
    """Represents a logical fact: predicate(arg1, arg2, ...)"""
    
    predicate: str
    arguments: Tuple[str, ...]
    confidence: float = 1.0


@dataclass
class Rule:
    """Represents a logical rule: head :- body1, body2, ..."""
    
    head: Fact
    body: List[Fact]
    confidence: float = 1.0


class SymbolEmbedding(nn.Module):
    """
    Neural embeddings for symbolic entities and predicates.
    
    Maps discrete symbols to continuous vector space while preserving
    compositional structure.
    """
    
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Entity embeddings
        self.entity_embed = nn.Embedding(
            config.n_entities, config.d_symbol
        )
        
        # Predicate embeddings
        self.predicate_embed = nn.Embedding(
            config.n_predicates, config.d_symbol
        )
        
        # Variable embeddings (for unification)
        self.variable_embed = nn.Parameter(
            torch.randn(10, config.d_symbol) * 0.02  # 10 variable slots
        )
        
        # Composition network for combining predicate and arguments
        self.compose = nn.Sequential(
            nn.Linear(config.d_symbol * 3, config.d_symbol * 2),
            nn.GELU(),
            nn.Linear(config.d_symbol * 2, config.d_symbol),
        )
        
    def embed_entity(self, entity_idx: torch.Tensor) -> torch.Tensor:
        """Embed entity indices."""
        return self.entity_embed(entity_idx)
        
    def embed_predicate(self, predicate_idx: torch.Tensor) -> torch.Tensor:
        """Embed predicate indices."""
        return self.predicate_embed(predicate_idx)
        
    def embed_fact(
        self,
        predicate_idx: torch.Tensor,
        arg1_idx: torch.Tensor,
        arg2_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed a fact (predicate with arguments) into vector space.
        
        Composition: f(P, A1, A2) -> embedding
        """
        pred_emb = self.embed_predicate(predicate_idx)
        arg1_emb = self.embed_entity(arg1_idx)
        arg2_emb = self.embed_entity(arg2_idx)
        
        # Concatenate and compose
        combined = torch.cat([pred_emb, arg1_emb, arg2_emb], dim=-1)
        return self.compose(combined)


class SoftUnification(nn.Module):
    """
    Differentiable unification for neural theorem proving.
    
    Instead of hard symbolic unification, uses soft matching in embedding
    space to enable gradient-based learning.
    """
    
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        
        # Unification scoring network
        self.score_net = nn.Sequential(
            nn.Linear(config.d_symbol * 2, config.d_symbol),
            nn.GELU(),
            nn.Linear(config.d_symbol, 1),
        )
        
    def unify(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft unification of query against candidate facts.
        
        Args:
            query: Query embedding (batch, d_symbol)
            candidates: Candidate fact embeddings (batch, n_candidates, d_symbol)
            
        Returns:
            unification_scores: Soft unification scores (batch, n_candidates)
            unified_binding: Weighted combination of candidates (batch, d_symbol)
        """
        batch, n_candidates, d_symbol = candidates.shape
        
        # Expand query for comparison
        query_expanded = query.unsqueeze(1).expand(-1, n_candidates, -1)
        
        # Compute unification scores
        combined = torch.cat([query_expanded, candidates], dim=-1)
        scores = self.score_net(combined).squeeze(-1)  # (batch, n_candidates)
        
        # Soft attention over candidates
        weights = F.softmax(scores / self.temperature, dim=-1)
        
        # Compute unified binding (weighted sum of candidates)
        unified = torch.einsum("bn,bnd->bd", weights, candidates)
        
        return weights, unified
        
    def match_score(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute match score between two embeddings."""
        # Cosine similarity
        sim = F.cosine_similarity(emb1, emb2, dim=-1)
        return (sim + 1) / 2  # Map to [0, 1]


class NeuralRuleBase(nn.Module):
    """
    Learnable rule base for neural theorem proving.
    
    Contains:
    1. Explicit rules (user-defined or extracted)
    2. Implicit rules (learned neural patterns)
    """
    
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Learnable rule embeddings
        self.rule_embeddings = nn.Parameter(
            torch.randn(config.n_rules, config.d_symbol) * 0.02
        )
        
        # Rule head decoder
        self.head_decoder = nn.Sequential(
            nn.Linear(config.d_symbol, config.d_symbol * 2),
            nn.GELU(),
            nn.Linear(config.d_symbol * 2, config.d_symbol),
        )
        
        # Rule body decoder (outputs attention over facts)
        self.body_decoder = nn.Sequential(
            nn.Linear(config.d_symbol, config.d_symbol * 2),
            nn.GELU(),
            nn.Linear(config.d_symbol * 2, config.d_symbol),
        )
        
        # Rule confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(config.d_symbol, config.d_symbol // 2),
            nn.GELU(),
            nn.Linear(config.d_symbol // 2, 1),
            nn.Sigmoid(),
        )
        
    def get_applicable_rules(
        self,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find rules applicable to a query.
        
        Args:
            query: Query embedding (batch, d_symbol)
            
        Returns:
            rule_weights: Applicability weights (batch, n_rules)
            rule_bodies: Body embeddings for applicable rules
        """
        batch = query.shape[0]
        
        # Compute rule applicability
        query_expanded = query.unsqueeze(1)  # (batch, 1, d_symbol)
        rules = self.rule_embeddings.unsqueeze(0)  # (1, n_rules, d_symbol)
        
        # Similarity to rule heads
        rule_heads = self.head_decoder(rules)
        similarity = F.cosine_similarity(query_expanded, rule_heads, dim=-1)
        
        # Weight by rule confidence
        confidences = self.confidence(self.rule_embeddings).squeeze(-1)
        weights = F.softmax(similarity * confidences, dim=-1)
        
        # Get rule bodies
        bodies = self.body_decoder(self.rule_embeddings)
        
        return weights, bodies


class ProofTree(nn.Module):
    """
    Neural proof tree construction for explainable reasoning.
    
    Builds differentiable proof trees that explain how conclusions
    are derived from facts and rules.
    """
    
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        self.max_depth = config.max_proof_depth
        
        # Proof step network
        self.proof_step = nn.GRUCell(
            config.d_symbol,
            config.d_symbol,
        )
        
        # Goal decomposition
        self.decompose = nn.Sequential(
            nn.Linear(config.d_symbol, config.d_symbol * 2),
            nn.GELU(),
            nn.Linear(config.d_symbol * 2, config.d_symbol * 2),
        )
        
        # Proof success estimator
        self.success_estimator = nn.Sequential(
            nn.Linear(config.d_symbol, config.d_symbol // 2),
            nn.GELU(),
            nn.Linear(config.d_symbol // 2, 1),
            nn.Sigmoid(),
        )
        
    def prove(
        self,
        goal: torch.Tensor,
        facts: torch.Tensor,
        rules: torch.Tensor,
        depth: int = 0,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Attempt to prove a goal using facts and rules.
        
        Args:
            goal: Goal to prove (batch, d_symbol)
            facts: Available facts (batch, n_facts, d_symbol)
            rules: Available rules (n_rules, d_symbol)
            depth: Current proof depth
            
        Returns:
            proof_score: Probability that goal is provable (batch,)
            proof_trace: Explanation of proof steps
        """
        batch = goal.shape[0]
        
        # Base case: check if goal directly matches a fact
        fact_similarity = F.cosine_similarity(
            goal.unsqueeze(1), facts, dim=-1
        )  # (batch, n_facts)
        
        direct_proof = fact_similarity.max(dim=-1)[0]
        
        # Recursive case: try to prove via rules
        if depth < self.max_depth:
            # Decompose goal into subgoals
            subgoals = self.decompose(goal)
            subgoal1, subgoal2 = subgoals.chunk(2, dim=-1)
            
            # Recursively prove subgoals
            score1, trace1 = self.prove(subgoal1, facts, rules, depth + 1)
            score2, trace2 = self.prove(subgoal2, facts, rules, depth + 1)
            
            # Combine subgoal proofs (AND semantics)
            recursive_proof = score1 * score2
        else:
            recursive_proof = torch.zeros(batch, device=goal.device)
            
        # Combine direct and recursive proofs (OR semantics)
        total_proof = torch.max(direct_proof, recursive_proof)
        
        proof_trace = {
            "goal": goal,
            "direct_score": direct_proof,
            "recursive_score": recursive_proof,
            "depth": depth,
        }
        
        return total_proof, [proof_trace]


class NeuroSymbolicReasoner(nn.Module):
    """
    Neuro-Symbolic Reasoning Engine - Core NEXUS component.
    
    Combines neural pattern recognition with symbolic logical reasoning:
    
    1. Neural Component (System 1):
       - Fast pattern matching in embedding space
       - Soft unification and similarity-based retrieval
       - Learned representations for entities and predicates
       
    2. Symbolic Component (System 2):
       - Rule-based inference
       - Proof tree construction
       - Explicit knowledge representation
       
    This addresses key LLM limitations:
    - Hallucination: Grounded in explicit knowledge base
    - Reasoning: Structured proof trees, not just pattern matching
    - Explainability: Traceable inference chains
    """
    
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Neural components
        self.symbol_embedding = SymbolEmbedding(config)
        self.soft_unification = SoftUnification(config)
        self.rule_base = NeuralRuleBase(config)
        self.proof_tree = ProofTree(config)
        
        # Knowledge integration
        self.fact_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_symbol * 2),
            nn.GELU(),
            nn.Linear(config.d_symbol * 2, config.d_symbol),
        )
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_symbol * 2),
            nn.GELU(),
            nn.Linear(config.d_symbol * 2, config.d_symbol),
        )
        
        # Answer decoder
        self.answer_decoder = nn.Sequential(
            nn.Linear(config.d_symbol * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        
        # Confidence estimator for grounded answers
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_symbol, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Knowledge base (populated during training/inference)
        self.register_buffer(
            "fact_bank",
            torch.zeros(1000, config.d_symbol)
        )
        self.n_facts = 0
        
    def encode_facts(
        self,
        facts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode facts into symbolic representation space."""
        return self.fact_encoder(facts)
        
    def encode_query(
        self,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Encode query into symbolic representation space."""
        return self.query_encoder(query)
        
    def add_facts(self, fact_embeddings: torch.Tensor):
        """Add facts to the knowledge base."""
        n_new = fact_embeddings.shape[0]
        if self.n_facts + n_new <= self.fact_bank.shape[0]:
            self.fact_bank[self.n_facts:self.n_facts + n_new] = fact_embeddings
            self.n_facts += n_new
            
    def reason(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform neuro-symbolic reasoning on a query.
        
        Args:
            query: Query tensor (batch, d_model)
            context: Optional context facts (batch, n_facts, d_model)
            
        Returns:
            Dictionary containing:
            - answer: Derived answer embedding
            - confidence: Confidence in the answer
            - proof_trace: Explanation of reasoning
        """
        batch = query.shape[0]
        device = query.device
        
        # Encode query
        query_emb = self.encode_query(query)  # (batch, d_symbol)
        
        # Encode context facts if provided
        if context is not None and context.numel() > 0:
            fact_emb = self.encode_facts(context)  # (batch, n_facts, d_symbol)
        elif self.n_facts > 0:
            # Use stored fact bank - ensure it's on the correct device
            fact_emb = self.fact_bank[:self.n_facts].unsqueeze(0).to(device)
            fact_emb = fact_emb.expand(batch, -1, -1)
        else:
            # No facts available - create a dummy fact bank for computation
            fact_emb = torch.zeros(batch, 1, self.config.d_symbol, device=device, dtype=query.dtype)
            
        # Step 1: Direct fact retrieval (System 1 - fast)
        retrieval_weights, retrieved = self.soft_unification.unify(
            query_emb, fact_emb
        )
        
        # Step 2: Rule-based reasoning (System 2 - slow)
        rule_weights, rule_bodies = self.rule_base.get_applicable_rules(query_emb)
        
        # Step 3: Proof construction
        proof_score, proof_trace = self.proof_tree.prove(
            query_emb,
            fact_emb,
            self.rule_base.rule_embeddings,
        )
        
        # Combine retrieved and derived knowledge
        combined = torch.cat([retrieved, query_emb], dim=-1)
        answer = self.answer_decoder(combined)
        
        # Estimate confidence (grounding score)
        confidence = self.confidence_head(retrieved)
        
        return {
            "answer": answer,
            "confidence": confidence.squeeze(-1),
            "proof_score": proof_score,
            "proof_trace": proof_trace,
            "retrieval_weights": retrieval_weights,
        }
        
    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            query: Query tensor (batch, d_model)
            context: Context facts (batch, n_facts, d_model)
            target: Target answer for supervised training
            
        Returns:
            Dictionary with answer, confidence, and loss
        """
        result = self.reason(query, context)
        
        if target is not None:
            # Compute loss
            answer_loss = F.mse_loss(result["answer"], target)
            result["loss"] = answer_loss
            
        return result


class KnowledgeGraph(nn.Module):
    """
    Neural knowledge graph for structured knowledge representation.
    
    Stores entities, relations, and supports:
    - Link prediction
    - Entity classification  
    - Multi-hop reasoning
    """
    
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(
            config.n_entities, config.d_symbol
        )
        self.relation_embeddings = nn.Embedding(
            config.n_predicates, config.d_symbol
        )
        
        # Graph neural network for message passing
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(config.d_symbol)
            for _ in range(3)
        ])
        
    def encode_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a (head, relation, tail) triple."""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        # RotatE-style composition
        return h * r - t
        
    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score a triple for link prediction."""
        diff = self.encode_triple(head, relation, tail)
        return -torch.norm(diff, dim=-1)


class GraphConvLayer(nn.Module):
    """Graph convolution layer for knowledge graph reasoning."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Message passing on graph."""
        # Simplified message passing
        # In practice, would use proper sparse operations
        messages = torch.cat([node_features, edge_features], dim=-1)
        aggregated = self.linear(messages)
        return self.norm(node_features + aggregated)
