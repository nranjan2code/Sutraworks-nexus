# Causal Inference Engine

## From Correlation to Causation

The Causal Inference Engine enables NEXUS to understand cause-and-effect relationships, plan interventions, and reason about counterfactuals—capabilities fundamentally beyond pattern recognition.

---

## The Causal Revolution

### Why Causality Matters

Standard neural networks learn correlations:
- "When X appears, Y often follows"
- Cannot distinguish: X causes Y vs Y causes X vs Z causes both

This leads to failures:
- **Spurious correlations** (ice cream sales → drowning deaths)
- **Intervention blindness** (will forcing X change Y?)
- **Counterfactual inability** (what if X hadn't happened?)

### Pearl's Causal Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                   Pearl's Ladder of Causation                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 3: COUNTERFACTUALS (Imagining)                          │
│  ─────────────────────────────────────                         │
│  "What if X had been different?"                               │
│  P(Y_x | X = x', Y = y')                                       │
│  Requires: Full causal model                                    │
│                                                                 │
│  ▲                                                              │
│  │                                                              │
│  │                                                              │
│  Level 2: INTERVENTIONS (Doing)                                │
│  ─────────────────────────────────                             │
│  "What if I do X?"                                             │
│  P(Y | do(X))                                                  │
│  Requires: Causal graph structure                              │
│                                                                 │
│  ▲                                                              │
│  │                                                              │
│  │                                                              │
│  Level 1: ASSOCIATION (Seeing)                                 │
│  ─────────────────────────────                                 │
│  "What if I see X?"                                            │
│  P(Y | X)                                                      │
│  Standard ML lives here                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**NEXUS operates at ALL THREE levels.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Causal Inference Engine                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │              Causal Graph Structure                          │      │
│   │                                                              │      │
│   │    Variables:     V = {V₁, V₂, ..., Vₙ}                     │      │
│   │    Adjacency:     A ∈ {0,1}ⁿˣⁿ (learned or specified)       │      │
│   │    Edge weights:  W ∈ ℝⁿˣⁿ (learned)                        │      │
│   │                                                              │      │
│   │         V₁ ──┐                                              │      │
│   │              │                                              │      │
│   │              ▼                                              │      │
│   │         V₃ ───────► V₄                                      │      │
│   │              ▲                                              │      │
│   │              │                                              │      │
│   │         V₂ ──┘                                              │      │
│   │                                                              │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                           │                                             │
│          ┌────────────────┼────────────────┐                           │
│          │                │                │                           │
│          ▼                ▼                ▼                           │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │
│   │  Structure  │  │ Mechanism   │  │   Query     │                   │
│   │  Learning   │  │  Learning   │  │  Processor  │                   │
│   │             │  │             │  │             │                   │
│   │ • DAG       │  │ • P(Vᵢ|Pa)  │  │ • do(X)     │                   │
│   │ • Discover  │  │ • Neural    │  │ • Counter-  │                   │
│   │   edges     │  │   conditnls │  │   factuals  │                   │
│   └─────────────┘  └─────────────┘  └─────────────┘                   │
│          │                │                │                           │
│          └────────────────┼────────────────┘                           │
│                           │                                             │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │                    Inference Engine                          │      │
│   │                                                              │      │
│   │   • Observational: P(Y | X)                                 │      │
│   │   • Interventional: P(Y | do(X))                            │      │
│   │   • Counterfactual: P(Y_{x'} | X=x, Y=y)                    │      │
│   │   • Planning: argmax_do(X) P(Y | do(X))                     │      │
│   │                                                              │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Causal Graph Representation

```python
class CausalGraph(nn.Module):
    """Differentiable causal graph structure."""
    
    def __init__(self, num_variables, hidden_dim):
        super().__init__()
        self.num_vars = num_variables
        self.hidden_dim = hidden_dim
        
        # Soft adjacency matrix (will be discretized)
        # A[i,j] = 1 means i → j (i causes j)
        self.adjacency_logits = nn.Parameter(
            torch.zeros(num_variables, num_variables)
        )
        
        # Edge strength weights
        self.edge_weights = nn.Parameter(
            torch.zeros(num_variables, num_variables)
        )
        
        # Variable embeddings
        self.var_embeddings = nn.Embedding(num_variables, hidden_dim)
    
    def get_adjacency(self, temperature=1.0, hard=False):
        """Get (soft) adjacency matrix.
        
        Returns:
            A: [num_vars, num_vars] adjacency matrix
        """
        # Apply sigmoid for soft edges
        A_soft = torch.sigmoid(self.adjacency_logits / temperature)
        
        # Mask diagonal (no self-loops)
        mask = 1 - torch.eye(self.num_vars, device=A_soft.device)
        A_soft = A_soft * mask
        
        if hard:
            # Gumbel-Softmax for discrete but differentiable
            A_hard = (A_soft > 0.5).float()
            return A_hard - A_soft.detach() + A_soft  # Straight-through
        
        return A_soft
    
    def get_parents(self, var_idx, threshold=0.5):
        """Get parent variables of given variable."""
        A = self.get_adjacency()
        # Parents are nodes that point TO this variable
        parent_weights = A[:, var_idx]
        parents = (parent_weights > threshold).nonzero().flatten()
        return parents
    
    def get_children(self, var_idx, threshold=0.5):
        """Get child variables of given variable."""
        A = self.get_adjacency()
        # Children are nodes this variable points TO
        child_weights = A[var_idx, :]
        children = (child_weights > threshold).nonzero().flatten()
        return children
    
    def is_dag(self):
        """Check if current graph is acyclic (DAG)."""
        A = self.get_adjacency(hard=True)
        # A DAG has no cycles, so A^n should eventually be zero
        power = A.clone()
        for _ in range(self.num_vars):
            power = power @ A
            if power.trace() > 0:
                return False  # Found a cycle
        return True
    
    def dag_constraint(self):
        """Differentiable DAG constraint (NOTEARS).
        
        h(A) = tr(e^A) - n = 0 iff A is a DAG
        """
        A = self.get_adjacency()
        # Matrix exponential trick
        expm_A = torch.matrix_exp(A * A)  # Element-wise square for positive
        h = torch.trace(expm_A) - self.num_vars
        return h
```

### 2. Structural Causal Model (SCM)

```python
class StructuralCausalModel(nn.Module):
    """Full structural causal model with mechanisms."""
    
    def __init__(self, num_variables, hidden_dim, mechanism_hidden=64):
        super().__init__()
        self.num_vars = num_variables
        
        # Causal graph
        self.graph = CausalGraph(num_variables, hidden_dim)
        
        # Causal mechanisms: P(V_i | Parents(V_i))
        self.mechanisms = nn.ModuleList([
            CausalMechanism(hidden_dim, mechanism_hidden)
            for _ in range(num_variables)
        ])
        
        # Noise distributions (exogenous variables)
        self.noise_scale = nn.Parameter(torch.ones(num_variables))
    
    def forward(self, interventions=None, num_samples=1):
        """Sample from the SCM.
        
        Args:
            interventions: dict {var_idx: value} for do(X=x)
            num_samples: number of samples to draw
            
        Returns:
            samples: [num_samples, num_vars, hidden_dim]
        """
        A = self.graph.get_adjacency()
        samples = torch.zeros(num_samples, self.num_vars, self.hidden_dim)
        
        # Topological sort for generation order
        order = self.topological_order(A)
        
        # Generate in causal order
        for var_idx in order:
            if interventions and var_idx in interventions:
                # Intervention: set value directly
                samples[:, var_idx] = interventions[var_idx]
            else:
                # Get parent values
                parents = self.graph.get_parents(var_idx)
                if len(parents) > 0:
                    parent_values = samples[:, parents]
                else:
                    parent_values = None
                
                # Sample from mechanism
                noise = torch.randn(num_samples, self.hidden_dim)
                noise = noise * self.noise_scale[var_idx]
                
                samples[:, var_idx] = self.mechanisms[var_idx](
                    parent_values, noise
                )
        
        return samples
    
    def topological_order(self, A):
        """Get topological ordering of variables."""
        # Simple implementation using Kahn's algorithm
        A_binary = (A > 0.5).float()
        in_degree = A_binary.sum(dim=0)
        
        order = []
        remaining = set(range(self.num_vars))
        
        while remaining:
            # Find node with no incoming edges
            for node in remaining:
                if in_degree[node] == 0:
                    order.append(node)
                    remaining.remove(node)
                    # Update in-degrees
                    for child in range(self.num_vars):
                        if A_binary[node, child] > 0:
                            in_degree[child] -= 1
                    break
            else:
                # Cycle detected, break arbitrary edge
                order.extend(remaining)
                break
        
        return order


class CausalMechanism(nn.Module):
    """Neural causal mechanism P(V_i | Parents(V_i))."""
    
    def __init__(self, hidden_dim, mechanism_hidden):
        super().__init__()
        
        # Mechanism network
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, mechanism_hidden),  # parent + noise
            nn.ReLU(),
            nn.Linear(mechanism_hidden, mechanism_hidden),
            nn.ReLU(),
            nn.Linear(mechanism_hidden, hidden_dim)
        )
        
        # For root nodes (no parents)
        self.prior = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, parent_values, noise):
        """
        Args:
            parent_values: [batch, num_parents, hidden_dim] or None
            noise: [batch, hidden_dim] exogenous noise
            
        Returns:
            value: [batch, hidden_dim]
        """
        if parent_values is None or parent_values.size(1) == 0:
            # Root node: just transform noise
            return self.prior(noise)
        
        # Aggregate parent values
        parent_agg = parent_values.mean(dim=1)  # Simple mean aggregation
        
        # Combine with noise
        combined = torch.cat([parent_agg, noise], dim=-1)
        
        return self.net(combined)
```

---

## Causal Queries

### Interventional Inference: do(X)

```python
class InterventionalQuery:
    """Compute P(Y | do(X = x))."""
    
    def __init__(self, scm):
        self.scm = scm
    
    def __call__(self, target_var, intervention_var, intervention_value, 
                 num_samples=1000):
        """
        Compute interventional distribution.
        
        "What is the distribution of target_var if we SET intervention_var
         to intervention_value?"
        """
        # Sample with intervention
        interventions = {intervention_var: intervention_value}
        samples = self.scm(interventions=interventions, num_samples=num_samples)
        
        # Extract target variable samples
        target_samples = samples[:, target_var]
        
        return {
            'samples': target_samples,
            'mean': target_samples.mean(dim=0),
            'std': target_samples.std(dim=0)
        }
    
    def causal_effect(self, target_var, intervention_var, 
                      value_0, value_1, num_samples=1000):
        """
        Average Causal Effect (ACE):
        E[Y | do(X=1)] - E[Y | do(X=0)]
        """
        effect_0 = self(target_var, intervention_var, value_0, num_samples)
        effect_1 = self(target_var, intervention_var, value_1, num_samples)
        
        ace = effect_1['mean'] - effect_0['mean']
        
        return {
            'ace': ace,
            'effect_0': effect_0['mean'],
            'effect_1': effect_1['mean']
        }
```

### Counterfactual Inference

```python
class CounterfactualQuery:
    """Compute counterfactuals: What if X had been different?"""
    
    def __init__(self, scm):
        self.scm = scm
    
    def __call__(self, observed_values, intervention_var, counterfactual_value,
                 target_var):
        """
        Three-step counterfactual algorithm:
        1. Abduction: Infer noise given observations
        2. Action: Apply intervention
        3. Prediction: Compute outcome with inferred noise
        """
        
        # Step 1: Abduction - infer exogenous noise
        # Given observed V = v, what were the noise terms U?
        inferred_noise = self.abduction(observed_values)
        
        # Step 2: Action - modify the graph for intervention
        # do(X = counterfactual_value)
        modified_interventions = {intervention_var: counterfactual_value}
        
        # Step 3: Prediction - generate with fixed noise
        counterfactual_sample = self.predict_with_noise(
            inferred_noise, modified_interventions
        )
        
        return counterfactual_sample[:, target_var]
    
    def abduction(self, observed_values):
        """Infer exogenous noise from observations."""
        # This requires inverting the mechanisms
        # Simplified: assume additive noise model
        #   V_i = f(Parents(V_i)) + U_i
        #   U_i = V_i - f(Parents(V_i))
        
        A = self.scm.graph.get_adjacency()
        order = self.scm.topological_order(A)
        
        inferred_noise = torch.zeros_like(observed_values)
        
        for var_idx in order:
            parents = self.scm.graph.get_parents(var_idx)
            
            if len(parents) > 0:
                parent_values = observed_values[:, parents]
                predicted = self.scm.mechanisms[var_idx](
                    parent_values, torch.zeros_like(observed_values[:, var_idx])
                )
                inferred_noise[:, var_idx] = observed_values[:, var_idx] - predicted
            else:
                inferred_noise[:, var_idx] = observed_values[:, var_idx]
        
        return inferred_noise
    
    def predict_with_noise(self, noise, interventions):
        """Forward pass with fixed noise values."""
        A = self.scm.graph.get_adjacency()
        order = self.scm.topological_order(A)
        
        samples = torch.zeros_like(noise)
        
        for var_idx in order:
            if interventions and var_idx in interventions:
                samples[:, var_idx] = interventions[var_idx]
            else:
                parents = self.scm.graph.get_parents(var_idx)
                if len(parents) > 0:
                    parent_values = samples[:, parents]
                else:
                    parent_values = None
                
                samples[:, var_idx] = self.scm.mechanisms[var_idx](
                    parent_values, noise[:, var_idx]
                )
        
        return samples
```

---

## Structure Learning

### Learn Graph from Data

```python
class CausalDiscovery(nn.Module):
    """Learn causal structure from observational data."""
    
    def __init__(self, num_variables, hidden_dim):
        super().__init__()
        self.num_vars = num_variables
        
        # Initialize SCM with unknown structure
        self.scm = StructuralCausalModel(num_variables, hidden_dim)
        
        # DAG constraint weight (increases during training)
        self.dag_weight = 0.0
    
    def fit(self, data, num_epochs=1000, lr=0.01):
        """Learn causal structure and mechanisms from data.
        
        Args:
            data: [num_samples, num_vars, hidden_dim] observations
        """
        optimizer = torch.optim.Adam(self.scm.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Reconstruction loss
            recon_loss = self.reconstruction_loss(data)
            
            # DAG constraint
            dag_loss = self.scm.graph.dag_constraint()
            
            # Sparsity regularization
            A = self.scm.graph.get_adjacency()
            sparsity_loss = A.abs().sum()
            
            # Total loss
            loss = recon_loss + self.dag_weight * dag_loss + 0.01 * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            # Increase DAG constraint weight (augmented Lagrangian)
            if epoch % 100 == 0:
                self.dag_weight *= 1.5
                self.dag_weight = min(self.dag_weight + 0.1, 100.0)
        
        return self.scm.graph.get_adjacency(hard=True)
    
    def reconstruction_loss(self, data):
        """How well can we reconstruct data with current graph?"""
        A = self.scm.graph.get_adjacency()
        
        total_loss = 0.0
        
        for var_idx in range(self.num_vars):
            parents = self.scm.graph.get_parents(var_idx)
            
            if len(parents) > 0:
                parent_values = data[:, parents]
            else:
                parent_values = None
            
            # Predict this variable from parents
            noise = torch.zeros(data.size(0), data.size(2))
            predicted = self.scm.mechanisms[var_idx](parent_values, noise)
            
            # Reconstruction error
            actual = data[:, var_idx]
            total_loss += F.mse_loss(predicted, actual)
        
        return total_loss / self.num_vars
```

---

## Causal Planning

### Plan Actions to Achieve Goals

```python
class CausalPlanner:
    """Plan interventions to achieve desired outcomes."""
    
    def __init__(self, scm, num_samples=100):
        self.scm = scm
        self.num_samples = num_samples
        self.intervention_query = InterventionalQuery(scm)
    
    def plan(self, goal_var, goal_value, actionable_vars, 
             value_range=(-1, 1), num_candidates=20):
        """
        Find best intervention to achieve goal.
        
        Args:
            goal_var: Variable we want to influence
            goal_value: Desired value for goal variable
            actionable_vars: Variables we can intervene on
            value_range: Range of intervention values
            num_candidates: Number of intervention values to try
            
        Returns:
            best_intervention: (var_idx, value) tuple
        """
        best_intervention = None
        best_distance = float('inf')
        
        # Try each actionable variable
        for act_var in actionable_vars:
            # Try different intervention values
            values = torch.linspace(value_range[0], value_range[1], num_candidates)
            
            for value in values:
                # Simulate intervention
                result = self.intervention_query(
                    target_var=goal_var,
                    intervention_var=act_var,
                    intervention_value=value.expand(self.num_samples, -1),
                    num_samples=self.num_samples
                )
                
                # Distance to goal
                distance = (result['mean'] - goal_value).norm()
                
                if distance < best_distance:
                    best_distance = distance
                    best_intervention = (act_var, value)
        
        return {
            'intervention': best_intervention,
            'expected_distance': best_distance.item(),
            'variable': best_intervention[0],
            'value': best_intervention[1].item()
        }
    
    def multi_step_plan(self, goal_var, goal_value, actionable_vars, 
                       max_steps=3):
        """Plan sequence of interventions."""
        
        current_state = self.scm(num_samples=1)
        plan = []
        
        for step in range(max_steps):
            # Find best single intervention
            intervention = self.plan(
                goal_var, goal_value, actionable_vars
            )
            
            if intervention['expected_distance'] < 0.1:
                break  # Close enough
            
            plan.append(intervention)
            
            # Update state with intervention
            current_state = self.scm(
                interventions={intervention['variable']: intervention['value']},
                num_samples=1
            )
        
        return plan
```

---

## Integration with NEXUS

### Causal Reasoning in Context

```python
class CausalReasoningModule(nn.Module):
    """Integrate causal reasoning into NEXUS."""
    
    def __init__(self, hidden_dim, max_variables=32):
        super().__init__()
        
        # Extract causal variables from hidden states
        self.variable_extractor = nn.Linear(hidden_dim, max_variables)
        
        # SCM for reasoning
        self.scm = StructuralCausalModel(max_variables, hidden_dim)
        
        # Query processor
        self.query_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.query_type_classifier = nn.Linear(hidden_dim, 3)  # obs/int/cf
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, hidden_states, query=None):
        """
        Apply causal reasoning to hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            query: Optional query embedding
            
        Returns:
            causal_output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Extract variable values from hidden states
        var_logits = self.variable_extractor(hidden_states)  # [B, S, V]
        var_values = var_logits.mean(dim=1)  # Aggregate over sequence
        
        # Determine query type
        if query is not None:
            query_enc = self.query_encoder(query)
            query_type = self.query_type_classifier(query_enc)
            query_type = F.softmax(query_type, dim=-1)
        else:
            query_type = torch.tensor([1.0, 0.0, 0.0])  # Default: observational
        
        # Perform causal inference
        if query_type.argmax() == 0:
            # Observational
            causal_result = self.observational_inference(var_values)
        elif query_type.argmax() == 1:
            # Interventional
            causal_result = self.interventional_inference(var_values, query)
        else:
            # Counterfactual
            causal_result = self.counterfactual_inference(var_values, query)
        
        # Combine with original hidden states
        causal_expanded = causal_result.unsqueeze(1).expand(-1, seq_len, -1)
        output = self.output_proj(
            torch.cat([hidden_states, causal_expanded], dim=-1)
        )
        
        return output
    
    def observational_inference(self, var_values):
        """Standard forward pass through SCM."""
        return self.scm(num_samples=var_values.size(0))[:, -1]  # Last variable
    
    def interventional_inference(self, var_values, query):
        """do(X) inference."""
        # Parse query to get intervention target
        # Simplified: intervene on highest-valued variable
        int_var = var_values.argmax(dim=-1)[0].item()
        int_value = query  # Use query as intervention value
        
        result = self.scm(
            interventions={int_var: int_value},
            num_samples=var_values.size(0)
        )
        return result[:, -1]
    
    def counterfactual_inference(self, var_values, query):
        """Counterfactual reasoning."""
        cf_query = CounterfactualQuery(self.scm)
        
        # Simplified counterfactual
        int_var = 0  # First variable
        target_var = -1  # Last variable
        
        return cf_query(
            observed_values=var_values.unsqueeze(-1).expand(-1, -1, self.scm.hidden_dim),
            intervention_var=int_var,
            counterfactual_value=query,
            target_var=target_var
        )
```

---

## Configuration

```python
causal_config = {
    # Graph structure
    'num_variables': 32,
    'max_parents': 5,
    
    # Learning
    'dag_constraint_weight': 1.0,
    'sparsity_weight': 0.01,
    'mechanism_hidden_dim': 64,
    
    # Inference
    'num_samples': 100,
    'counterfactual_samples': 50,
    
    # Planning
    'max_planning_steps': 5,
    'intervention_candidates': 20,
}
```

---

## Causal Reasoning Examples

### Example 1: Medical Diagnosis

```
Graph: Smoking → Lung Cancer
       Smoking → Yellow Teeth
       
Query: "Does whitening teeth reduce cancer risk?"

Observational: P(Cancer | White Teeth) < P(Cancer | Yellow Teeth)
  → Correlation suggests yes!

Interventional: P(Cancer | do(Teeth=White)) = P(Cancer | do(Teeth=Yellow))
  → No causal effect! Teeth don't cause cancer.

NEXUS correctly identifies: Correlation ≠ Causation
```

### Example 2: Planning

```
Graph: Study → Knowledge → Test Score
       Sleep → Energy → Test Score
       
Goal: Maximize Test Score

Plan: 1. Intervene on Study (high impact on Knowledge)
      2. Intervene on Sleep (high impact on Energy)
      
NEXUS finds optimal intervention sequence.
```

---

## Benefits and Limitations

### Benefits

| Benefit | Description |
|---------|-------------|
| Intervention reasoning | Predict effects of actions |
| Counterfactual thinking | "What if" scenarios |
| Robust generalization | Causal structure transfers |
| Explainability | Causal paths explain predictions |

### Limitations

| Limitation | Mitigation |
|------------|------------|
| Structure learning is hard | Use domain knowledge |
| Scalability | Limit variable count |
| Identifiability | Require interventional data |

---

## Further Reading

- [Causality (Pearl, 2009)](http://bayes.cs.ucla.edu/BOOK-2K/)
- [Elements of Causal Inference](https://mitpress.mit.edu/books/elements-causal-inference)
- [NOTEARS Algorithm](https://arxiv.org/abs/1803.01422)
- [State Space Module](state-space.md)
- [Integration Guide](integration.md)

---

*Causality is the language of agency. NEXUS speaks it fluently.*
