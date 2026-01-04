# Neuro-Symbolic Reasoning

## Explainable Reasoning with Proof Traces

The Neuro-Symbolic Reasoner combines neural network flexibility with symbolic reasoning rigor, enabling NEXUS to produce explainable conclusions with verifiable proof traces.

---

## The Problem with Neural-Only Reasoning

### Black Box Limitations

Standard neural networks:
- Cannot explain their reasoning
- May hallucinate plausible-sounding but false conclusions
- Cannot be verified for logical consistency
- Fail silently without indication

### Why Symbolic Components Help

Symbolic reasoning provides:
- **Explainability**: Step-by-step derivations
- **Verifiability**: Proofs can be checked
- **Compositionality**: Complex reasoning from simple rules
- **Grounding**: Conclusions tied to knowledge

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Neuro-Symbolic Reasoner                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input Representation                                                   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │                  Neural Rule Base                        │           │
│  │                                                         │           │
│  │   Rule 1: IF [condition_emb_1] THEN [conclusion_emb_1]  │           │
│  │   Rule 2: IF [condition_emb_2] THEN [conclusion_emb_2]  │           │
│  │   ...                                                   │           │
│  │   Rule N: IF [condition_emb_N] THEN [conclusion_emb_N]  │           │
│  │                                                         │           │
│  │   (Rules are learnable embedding vectors)               │           │
│  └─────────────────────────────────────────────────────────┘           │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │                 Soft Unification                         │           │
│  │                                                         │           │
│  │   Match query against rule conditions using:            │           │
│  │   score = σ(query · condition / τ)                      │           │
│  │                                                         │           │
│  │   Produces: which rules apply and how strongly          │           │
│  └─────────────────────────────────────────────────────────┘           │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │                Forward Chaining                          │           │
│  │                                                         │           │
│  │   For each applicable rule:                             │           │
│  │     - Derive conclusion with confidence                 │           │
│  │     - Add to working memory                             │           │
│  │     - Record in proof trace                             │           │
│  │   Repeat until fixed point or max depth                 │           │
│  └─────────────────────────────────────────────────────────┘           │
│         │                                                               │
│         ├─────────────────────────────────────────┐                    │
│         │                                         │                    │
│         ▼                                         ▼                    │
│  ┌─────────────────┐                    ┌─────────────────┐            │
│  │ Output + Conf   │                    │  Proof Trace    │            │
│  │                 │                    │                 │            │
│  │ Derived facts   │                    │ Step 1: Rule 3  │            │
│  │ with confidence │                    │ Step 2: Rule 7  │            │
│  │ scores          │                    │ Step 3: Rule 2  │            │
│  │                 │                    │ → Conclusion    │            │
│  └─────────────────┘                    └─────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Neural Rule Base

### Rule Representation

Each rule is represented as learnable embeddings:

```python
class NeuralRule:
    """A single neural rule."""
    
    def __init__(self, hidden_dim):
        # Condition: what must be true for rule to apply
        self.condition = nn.Parameter(torch.randn(hidden_dim))
        
        # Conclusion: what we can derive if rule applies
        self.conclusion = nn.Parameter(torch.randn(hidden_dim))
        
        # Confidence: base confidence of this rule
        self.confidence = nn.Parameter(torch.tensor(0.9))

class NeuralRuleBase(nn.Module):
    """Collection of neural rules."""
    
    def __init__(self, hidden_dim, num_rules):
        super().__init__()
        
        # Rule embeddings
        self.conditions = nn.Parameter(torch.randn(num_rules, hidden_dim))
        self.conclusions = nn.Parameter(torch.randn(num_rules, hidden_dim))
        self.confidences = nn.Parameter(torch.ones(num_rules) * 0.9)
        
        # Rule attention (which rules attend to which)
        self.rule_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
    
    def get_applicable_rules(self, query):
        """Find rules whose conditions match the query.
        
        Args:
            query: [batch, hidden_dim] - current state
            
        Returns:
            scores: [batch, num_rules] - applicability scores
            conclusions: [num_rules, hidden_dim] - rule conclusions
        """
        # Compute similarity between query and conditions
        scores = F.cosine_similarity(
            query.unsqueeze(1),  # [batch, 1, hidden]
            self.conditions.unsqueeze(0),  # [1, rules, hidden]
            dim=-1
        )  # [batch, num_rules]
        
        return scores, self.conclusions
```

### Intuitive Understanding

Think of rules as:
- **Condition**: "This pattern indicates..."
- **Conclusion**: "...therefore this follows"
- **Confidence**: "...with this certainty"

---

## Soft Unification

### Classical vs. Soft Unification

**Classical Unification**:
```
Does "parent(X, Y) ∧ parent(Y, Z)" match "parent(alice, bob) ∧ parent(bob, charlie)"?
Answer: Yes (with X=alice, Y=bob, Z=charlie) or No
```

**Soft Unification**:
```
How well does query embedding match condition embedding?
Answer: 0.87 (continuous score in [0, 1])
```

### Implementation

```python
class SoftUnification(nn.Module):
    """Differentiable unification for neural terms."""
    
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
        # Learnable comparison
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, keys):
        """Compute soft unification scores.
        
        Args:
            query: [batch, hidden_dim] - what we're looking for
            keys: [num_keys, hidden_dim] - what we're matching against
            
        Returns:
            scores: [batch, num_keys] - match scores in [0, 1]
        """
        # Project for comparison
        q = self.query_proj(query)  # [batch, hidden]
        k = self.key_proj(keys)  # [keys, hidden]
        
        # Scaled dot-product similarity
        scores = torch.matmul(q, k.T) / (self.hidden_dim ** 0.5)
        
        # Temperature-scaled sigmoid for smooth [0, 1] output
        scores = torch.sigmoid(scores / self.temperature)
        
        return scores
    
    def unify_structured(self, query_struct, key_struct):
        """Unify structured terms (trees, graphs).
        
        Recursively unifies components and combines scores.
        """
        if isinstance(query_struct, torch.Tensor):
            # Base case: compare embeddings
            return self.forward(query_struct, key_struct)
        
        # Recursive case: unify each component
        scores = []
        for q_comp, k_comp in zip(query_struct, key_struct):
            scores.append(self.unify_structured(q_comp, k_comp))
        
        # Combine with product (logical AND)
        return torch.stack(scores).prod(dim=0)
```

### Why Soft Unification Matters

1. **Differentiability**: Gradients flow through matching
2. **Generalization**: Similar but not identical patterns match
3. **Uncertainty**: Confidence propagates through reasoning

---

## Forward Chaining

### Algorithm

```python
class ForwardChaining(nn.Module):
    """Differentiable forward chaining inference."""
    
    def __init__(self, rule_base, max_depth=10, threshold=0.5):
        super().__init__()
        self.rule_base = rule_base
        self.max_depth = max_depth
        self.threshold = threshold
    
    def forward(self, initial_facts):
        """Perform forward chaining inference.
        
        Args:
            initial_facts: [batch, num_facts, hidden_dim]
            
        Returns:
            derived_facts: [batch, num_derived, hidden_dim]
            proof_trace: List of inference steps
        """
        # Working memory starts with initial facts
        working_memory = initial_facts
        proof_trace = []
        
        for depth in range(self.max_depth):
            # Find applicable rules
            new_facts = []
            
            for fact_idx, fact in enumerate(working_memory.unbind(1)):
                # Check which rules match this fact
                scores, conclusions = self.rule_base.get_applicable_rules(fact)
                
                # For rules above threshold, derive conclusions
                applicable = scores > self.threshold
                
                for rule_idx in applicable.nonzero():
                    # Derive new fact
                    new_fact = conclusions[rule_idx]
                    confidence = scores[:, rule_idx] * self.rule_base.confidences[rule_idx]
                    
                    new_facts.append((new_fact, confidence))
                    
                    # Record in proof trace
                    proof_trace.append({
                        'depth': depth,
                        'rule': rule_idx.item(),
                        'premise': fact_idx,
                        'conclusion': len(new_facts) - 1,
                        'confidence': confidence.mean().item()
                    })
            
            if not new_facts:
                break  # Fixed point reached
            
            # Add new facts to working memory
            new_fact_tensor = torch.stack([f for f, c in new_facts], dim=1)
            working_memory = torch.cat([working_memory, new_fact_tensor], dim=1)
        
        return working_memory, proof_trace
```

### Optimization: Parallel Rule Application

```python
def parallel_forward_chain(self, facts, max_depth=5):
    """Vectorized forward chaining for efficiency."""
    
    batch, num_facts, hidden = facts.shape
    num_rules = self.rule_base.num_rules
    
    # All-pairs matching: which facts match which rules
    fact_flat = facts.reshape(batch * num_facts, hidden)
    match_scores = self.soft_unification(
        fact_flat, 
        self.rule_base.conditions
    ).reshape(batch, num_facts, num_rules)
    
    derived = [facts]
    
    for _ in range(max_depth):
        # Weighted combination of conclusions based on match scores
        # [batch, num_facts, num_rules] x [num_rules, hidden]
        new_facts = torch.einsum(
            'bfr,rh->bfh',
            match_scores,
            self.rule_base.conclusions
        )
        
        # Only keep high-confidence derivations
        confidence = match_scores.max(dim=-1).values
        new_facts = new_facts * (confidence > self.threshold).float().unsqueeze(-1)
        
        derived.append(new_facts)
    
    return torch.cat(derived, dim=1)
```

---

## Proof Trace Generation

### Structure

```python
@dataclass
class ProofStep:
    """A single step in a proof."""
    rule_id: int
    rule_name: str
    premises: List[int]  # Indices of facts used
    conclusion: int  # Index of derived fact
    confidence: float
    explanation: str

@dataclass
class ProofTrace:
    """Complete proof trace."""
    steps: List[ProofStep]
    initial_facts: List[str]
    final_conclusion: str
    total_confidence: float
    
    def to_natural_language(self):
        """Convert proof to readable explanation."""
        lines = ["Given:"]
        for i, fact in enumerate(self.initial_facts):
            lines.append(f"  {i+1}. {fact}")
        
        lines.append("\nReasoning:")
        for step in self.steps:
            lines.append(
                f"  By {step.rule_name} on ({step.premises}): "
                f"{step.explanation} [confidence: {step.confidence:.2f}]"
            )
        
        lines.append(f"\nConclusion: {self.final_conclusion}")
        lines.append(f"Overall confidence: {self.total_confidence:.2f}")
        
        return "\n".join(lines)
```

### Example Output

```
Given:
  1. Socrates is a human
  2. All humans are mortal

Reasoning:
  By Universal Instantiation on (1, 2): 
    Socrates is mortal [confidence: 0.95]

Conclusion: Socrates is mortal
Overall confidence: 0.95
```

---

## Knowledge Graph Integration

### External Grounding

```python
class KnowledgeGraph:
    """External knowledge graph for grounding."""
    
    def __init__(self, entities, relations, triples):
        self.entities = entities  # Dict[str, embedding]
        self.relations = relations  # Dict[str, embedding]
        self.triples = triples  # List[(head, rel, tail)]
        
        # Build index for efficient lookup
        self.entity_index = self._build_index()
    
    def ground(self, query_embedding, top_k=5):
        """Ground a neural query in the knowledge graph.
        
        Args:
            query_embedding: [hidden_dim]
            top_k: Number of matches to return
            
        Returns:
            matches: List of (entity, score) tuples
        """
        scores = {}
        for entity_name, entity_emb in self.entities.items():
            score = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                entity_emb.unsqueeze(0)
            ).item()
            scores[entity_name] = score
        
        # Return top-k matches
        sorted_entities = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_entities[:top_k]
    
    def verify_triple(self, head, relation, tail):
        """Check if a triple exists in the knowledge graph."""
        return (head, relation, tail) in self.triples
```

### Grounded Reasoning

```python
class GroundedReasoner(nn.Module):
    """Reasoner that grounds conclusions in knowledge."""
    
    def __init__(self, reasoner, knowledge_graph):
        super().__init__()
        self.reasoner = reasoner
        self.kg = knowledge_graph
    
    def reason_with_grounding(self, query):
        # Get neural reasoning output
        conclusion, proof = self.reasoner(query)
        
        # Ground conclusion in knowledge graph
        grounding = self.kg.ground(conclusion)
        
        # Adjust confidence based on grounding
        if grounding:
            grounding_score = grounding[0][1]  # Top match score
            adjusted_confidence = proof.total_confidence * grounding_score
        else:
            adjusted_confidence = proof.total_confidence * 0.5  # Penalty
        
        return {
            'conclusion': conclusion,
            'proof': proof,
            'grounding': grounding,
            'confidence': adjusted_confidence
        }
```

---

## Integration with NEXUS

### Reasoning Flow

```
State Space Output
       │
       ▼
┌─────────────────┐
│ Extract Query   │
│ (attention pool)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reasoner      │──────► Proof Trace
│                 │
│  Rule matching  │
│  + Forward chain│
│  + Grounding    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reasoning Loss  │
│ (if training)   │
└─────────────────┘
```

### Reasoning Loss

```python
def reasoning_loss(self, predictions, targets, proofs):
    """Compute reasoning-specific losses.
    
    Args:
        predictions: Model outputs
        targets: Ground truth
        proofs: Generated proof traces
        
    Returns:
        loss: Combined reasoning losses
    """
    losses = {}
    
    # 1. Conclusion accuracy
    losses['conclusion'] = F.cross_entropy(
        predictions['conclusion_logits'],
        targets['conclusion']
    )
    
    # 2. Proof validity (are steps logically sound?)
    if 'proof_validity' in targets:
        losses['validity'] = F.binary_cross_entropy(
            predictions['proof_confidence'],
            targets['proof_validity']
        )
    
    # 3. Grounding consistency
    if 'grounding' in predictions:
        losses['grounding'] = self.grounding_loss(
            predictions['grounding'],
            targets['grounding']
        )
    
    # Combine with weights
    total_loss = (
        losses['conclusion'] +
        0.5 * losses.get('validity', 0) +
        0.3 * losses.get('grounding', 0)
    )
    
    return total_loss, losses
```

---

## Configuration

```python
reasoning_config = {
    # Rule base
    'num_rules': 100,
    'rule_hidden_dim': 256,
    
    # Unification
    'unification_temperature': 0.1,
    'match_threshold': 0.5,
    
    # Forward chaining
    'max_proof_depth': 10,
    'beam_width': 5,
    
    # Knowledge grounding
    'use_knowledge_graph': True,
    'kg_embedding_dim': 256,
    'grounding_weight': 0.3,
    
    # Training
    'reasoning_loss_weight': 0.1,
}
```

---

## Benefits and Limitations

### Benefits

| Benefit | Description |
|---------|-------------|
| Explainability | Step-by-step proofs |
| Verifiability | Proofs can be checked |
| Compositionality | Complex from simple |
| Grounding | Tied to knowledge |

### Current Limitations

| Limitation | Mitigation |
|------------|------------|
| Fixed rule set | Expandable rule base |
| Soft matching errors | Threshold tuning |
| Computational cost | Parallel implementation |
| Limited expressivity | Hybrid with neural |

---

## Future Directions

1. **Rule Learning**: Automatically induce rules from data
2. **Probabilistic Logic**: Integration with probabilistic programming
3. **Meta-Reasoning**: Reasoning about reasoning strategies
4. **Interactive Proofs**: Human-in-the-loop verification

---

## Further Reading

- [Neural Theorem Provers](https://arxiv.org/abs/1705.11040)
- [Neural Logic Machines](https://arxiv.org/abs/1904.11694)
- [Neuro-Symbolic AI Survey](https://arxiv.org/abs/2305.00813)
- [Architecture Overview](overview.md)
- [Causal Engine](causal.md)

---

*Reasoning is not magic—it's mechanism. NEXUS makes the mechanism visible.*
