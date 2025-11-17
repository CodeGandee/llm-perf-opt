# Template: Neural Network Module Documentation

This template defines the standard structure for documenting PyTorch `nn.Module` classes. Use this format to create comprehensive, performance-focused documentation for any neural network module.

---

## Document Structure

Each module documentation file should be named `op-<ModuleName>.md` and contain the following sections in order:

### 1. Title (H1)
```markdown
# ModuleName
```
- Use the exact Python class name
- No extra formatting or prefixes

---

### 2. What It Is (H2)

**Purpose**: Provide a clear, high-level description of the module's role and functionality.

**Content**:
- **First sentence**: One-line summary of what the module does
- **Architecture context**: Where it fits in the overall model (e.g., "used in transformer decoder", "part of vision encoder")
- **Key features**: Highlight 2-4 distinctive characteristics or innovations
- **Comparison**: If applicable, contrast with standard implementations (e.g., "Unlike standard LayerNorm, RMSNorm omits mean centering")

**Example**:
```markdown
## What It Is
`DeepseekV2RMSNorm` (Root Mean Square Layer Normalization) is a normalization module used throughout DeepSeek-OCR's transformer layers. It's a variant of LayerNorm that normalizes activations using only the root mean square (RMS) statistic, without centering (no mean subtraction). This simplifies computation while maintaining training stability.

RMSNorm is equivalent to T5's LayerNorm implementation and is commonly used in modern large language models (LLaMA, Mistral, etc.) for efficiency.
```

**Writing tips**:
- Write for readers who may not be experts in the specific technique
- Explain acronyms on first use
- Include context about why this design choice was made
- Length: 2-4 paragraphs

---

### 3. Definition (H2)

**Purpose**: Show the actual Python class implementation.

**Content**:
- Complete class definition with key methods
- Include docstrings if they add clarity
- Show constructor signature and forward method
- Highlight important implementation details with comments
- Use proper Python syntax highlighting

**Example**:
```markdown
## Definition
\`\`\`python
class ModuleName(nn.Module):
    def __init__(self, param1: int, param2: float = 1.0):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        # Key components
        self.layer1 = nn.Linear(param1, param1 * 2)
        self.layer2 = nn.Linear(param1 * 2, param1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main computation
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x
\`\`\`
```

**Writing tips**:
- Include type hints
- Show only the essential methods (init, forward, and 1-2 others if critical)
- Omit lengthy helper methods unless they're conceptually important
- Add inline comments for non-obvious operations
- Length: 10-30 lines of code

---

### 4. Constructor Information (H2)

**Purpose**: Document the module's initialization parameters and created submodules.

**Content**:
```markdown
## Constructor Information
**Location**: `path/to/file.py:<line_start>-<line_end>`

**Signature**:
\`\`\`python
def __init__(self, param1: type, param2: type = default, ...)
\`\`\`

**Parameters**:
- `param1`: Description of param1 (e.g., typical values: 64, 128, 256)
- `param2`: Description of param2 (default: X, affects Y)
- `param3`: Description of param3 (optional, used when Z)

**Created Components**:
- `self.component1`: Type (shape, purpose)
- `self.component2`: Type (shape, purpose)
- Total parameters: X (at dtype)

**Key Behavior** (optional):
- Special initialization logic
- Conditional component creation
- Device/dtype handling
```

**Writing tips**:
- Always include file location with line numbers for traceability
- Describe parameters with realistic example values
- Explain what each parameter controls
- Calculate and document total parameter count
- Note any non-standard initialization patterns
- Length: 1-2 pages

---

### 5. Module Internals (H2)

**Purpose**: Visualize the data flow through the module using a sequence diagram.

**Content**:
```markdown
## Module Internals

\`\`\`mermaid
sequenceDiagram
    participant Input as Input Tensor<br/>(shape, dtype)
    participant Module as ModuleName
    participant Sub1 as Submodule1
    participant Sub2 as Submodule2
    participant Output as Output Tensor<br/>(shape)

    Input->>Module: forward(x)
    Module->>Sub1: operation1(x)
    Sub1-->>Module: intermediate

    Note over Module: Key transformation:<br/>Details here

    Module->>Sub2: operation2(intermediate)
    Sub2-->>Module: result
    Module-->>Output: Return processed tensor
\`\`\`
```

**Diagram guidelines**:
- Use `sequenceDiagram` for control flow
- Label participants with tensor shapes/dtypes
- Show key transformations with `Note over`
- Use `alt`/`else` blocks for conditional paths
- Keep it focused on the main forward path
- Include shapes to show dimension changes
- Use line breaks (`<br/>`) for multi-line labels

**Example patterns**:
```mermaid
# Conditional logic
alt condition
    Module->>Path1: do_something()
else
    Module->>Path2: do_other()
end

# Loops
loop for each layer
    Module->>Layer: forward(x)
end

# Parallel operations
par Parallel computation
    Module->>Branch1: compute1()
and
    Module->>Branch2: compute2()
end
```

**Writing tips**:
- Focus on conceptual flow, not implementation details
- Show the "happy path" (most common execution)
- Use notes to explain non-obvious transformations
- Include shapes to help readers track dimensions
- Length: 15-30 lines of Mermaid code

---

### 6. Key Pseudo Code (H2)

**Purpose**: Explain the forward pass logic with simplified, annotated code.

**Content**:
```markdown
## Key Pseudo Code

\`\`\`python
def forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    Args:
        input: Input tensor of shape (batch, seq_len, dim)
               or any shape ending with dim

    Returns:
        Output tensor of same shape as input
    """
    # 1. Preprocessing step
    x = self.preprocess(input)  # (batch, seq_len, dim)

    # 2. Main computation
    # Detailed explanation of what happens here
    intermediate = self.layer1(x)  # (batch, seq_len, dim*2)
    activated = F.activation(intermediate)

    # 3. Postprocessing
    output = self.layer2(activated)  # (batch, seq_len, dim)

    # 4. Optional residual connection
    output = output + input

    return output
\`\`\`

**Mathematical Formulation** (if applicable):
\`\`\`
Output = f(Input)

where:
  f(x) = layer2(activation(layer1(x))) + x
  activation: GELU, SiLU, etc.
\`\`\`

**Implementation Notes** (optional):
- Specific optimizations used
- Trade-offs in this implementation
- Differences from standard approaches
```

**Writing tips**:
- Include docstring with Args/Returns
- Annotate with tensor shapes after each operation
- Number key steps for clarity
- Explain "why" not just "what"
- Add mathematical formulation for complex operations
- Show concrete examples of tensor shapes
- Length: 20-50 lines

---

### 7. FLOP Count and Memory Usage Impact (H2)

**Purpose**: Provide detailed computational and memory cost analysis.

This is the **most critical section** for performance optimization work.

#### 7.1 FLOPs Subsection (H3)

**Content structure**:
```markdown
### FLOPs (per forward pass)

Assume:
- `B` = batch size (e.g., 1)
- `S` = sequence length (e.g., 8192)
- `d` = hidden dimension (e.g., 1280)
- [Other relevant dimensions]

**Operation Breakdown**:

1. **Operation 1 (e.g., Linear projection)**:
   \`\`\`
   FLOPs = 2 × B × S × d_in × d_out
   \`\`\`
   Explanation: Matrix multiplication (d_in × d_out) for each of B×S elements.

2. **Operation 2 (e.g., Activation)**:
   \`\`\`
   FLOPs = B × S × d (element-wise operation)
   \`\`\`

3. **Operation 3**:
   \`\`\`
   [Detailed formula]
   \`\`\`

**Total FLOPs**:
\`\`\`
FLOPs_total = Operation1 + Operation2 + Operation3
            = [simplified formula]
            ≈ [dominant term approximation]

Example (B=1, S=8192, d=1280):
FLOPs ≈ [concrete numerical result] MFLOPs / GFLOPs / TFLOPs
\`\`\`

**Comparison** (optional):
\`\`\`
This module:    X FLOPs
Standard impl:  Y FLOPs
Speedup:        Y/X
\`\`\`
```

**FLOP calculation guidelines**:

| Operation | FLOP Formula | Notes |
|-----------|--------------|-------|
| Matrix multiply (M×K) @ (K×N) | `2×M×K×N` | Factor of 2: multiply + add |
| Element-wise ops (+, *, /) | `N` | One op per element |
| Softmax | `~5×N` | exp + sum + divide |
| LayerNorm | `~5×N` | mean + var + normalize + scale + shift |
| RMSNorm | `~3×N` | var + normalize + scale |
| GELU | `~3×N` | Approximate (tanh, exp, etc.) |
| Attention (full) | `4×S²×d + 2×B×N²×d` | QKV proj + attention + out proj |

**Writing tips**:
- Start with dimension assumptions using realistic values
- Break down into individual operations
- Provide formula for each operation with explanation
- Sum to total FLOPs
- Simplify to dominant term (e.g., "≈ 2×B×S×d² dominates")
- Give concrete numerical example
- Compare to alternatives if relevant
- Length: 1-2 pages

#### 7.2 Memory Usage Subsection (H3)

**Content structure**:
```markdown
### Memory Usage

#### Parameters:
\`\`\`
component1: shape × dtype = calculation
component2: shape × dtype = calculation

Total parameters: X million parameters
At dtype: X × sizeof(dtype) = Y MB
\`\`\`

**Calculation example**:
- Weight: 1280 × 3840 = 4.9M parameters
- At bf16: 4.9M × 2 bytes = 9.8 MB

#### Activations (per forward pass):

**Intermediate tensors** (worst case, no in-place ops):
1. `tensor_name`: B × S × d × sizeof(dtype) = X MB
2. `tensor_name2`: [calculation]

**Example (B=1, S=8192, d=1280, bf16)**:
\`\`\`
tensor1: 1 × 8192 × 1280 × 2 = 21 MB
tensor2: [calculation]
...

Peak activation memory: ~X MB
\`\`\`

**Memory Optimization** (optional):
- In-place operations reduce peak by Y MB
- Gradient checkpointing can save Z MB
- Specific optimization techniques

#### Gradient Memory (training):

\`\`\`
Parameter gradients: same as parameter memory (X MB)
Activation gradients: same as activation memory (Y MB)

Optimizer states (AdamW):
- momentum: params × sizeof(fp32) = X × 4 bytes
- variance: params × sizeof(fp32) = X × 4 bytes

Total training memory: params + gradients + activations + optimizer
                     = [calculation]
\`\`\`

#### Special Considerations (if applicable):
- KV cache for attention: formula and size
- Dynamic allocation patterns
- Peak vs steady-state memory
- Multi-GPU memory distribution
```

**Memory calculation guidelines**:
- Always specify dtype (bf16=2 bytes, fp32=4 bytes, fp16=2 bytes)
- Calculate both parameter and activation memory
- Consider worst-case peak memory
- Note in-place optimization opportunities
- For training: include gradients and optimizer states
- Use realistic dimensions (B=1, S=8192 is common for inference)
- Show calculations step-by-step
- Length: 1-2 pages

---

### 8. Related Modules (H2)

**Purpose**: Document dependencies and relationships with other modules.

**Content**:
```markdown
## Related Modules
- **Used by**:
  - `ModuleA`: How this module is used
  - `ModuleB`: Another usage context

- **Inherits from**:
  - `ParentModule`: What is inherited

- **Uses / Depends on**:
  - `SubmoduleC`: How it's used internally
  - `HelperFunction`: Utility functions called

- **Alternative implementations**:
  - `VariantModule`: Different approach to same problem

- **Related patterns**:
  - `SimilarModule`: Shares architectural patterns
```

**Writing tips**:
- Link to other op-*.md files when possible
- Explain the relationship, not just list names
- Include both parent and child dependencies
- Note alternatives for comparison
- Length: 5-10 bullet points

---

### 9. Usage Pattern (H2)

**Purpose**: Show practical code examples from actual usage.

**Content**:
```markdown
## Usage Pattern

\`\`\`python
from modeling_module import ModuleName

# Initialization with typical parameters
module = ModuleName(
    param1=1280,
    param2=0.1,
    param3=True
)

# Forward pass example
input_tensor = torch.randn(1, 8192, 1280, dtype=torch.bfloat16, device='cuda')
output = module(input_tensor)
# output shape: (1, 8192, 1280)

# Integration example (how it's used in the model)
class ParentModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.module = ModuleName(config.param1, config.param2)

    def forward(self, x):
        x = self.module(x)
        return x
\`\`\`

**Configuration Example** (if applicable):
\`\`\`json
{
  "param1": 1280,
  "param2": 0.1,
  "param3": true
}
\`\`\`
```

**Writing tips**:
- Show complete, runnable examples
- Use realistic tensor shapes and dtypes
- Include both standalone and integrated usage
- Show configuration if loaded from config file
- Comment the output shapes
- Length: 15-40 lines of code

---

### 10. Key Performance Characteristics (H2) [Optional]

**Purpose**: Highlight performance implications and optimization opportunities.

**Content**:
```markdown
## Key Performance Characteristics

1. **Computational Complexity**: O(n²) / O(n) / etc.
   - Dominant term: [explanation]
   - Scaling behavior: [how it grows with inputs]

2. **Memory Characteristics**:
   - Peak memory: [when it occurs]
   - Steady-state memory: [typical usage]
   - Memory growth: [how it scales]

3. **Numerical Stability**:
   - Precision requirements: fp32 / bf16 / fp16
   - Overflow/underflow risks: [if any]
   - Mitigation strategies: [if implemented]

4. **Hardware Utilization**:
   - Memory bandwidth bound / compute bound
   - GPU occupancy considerations
   - CPU vs GPU efficiency

5. **Trade-offs**:
   - ✅ Advantages: [list 2-3 key benefits]
   - ❌ Limitations: [list 2-3 drawbacks]
   - When to use: [usage recommendations]

6. **Optimization Opportunities**:
   - Potential speedup techniques
   - Memory reduction strategies
   - Implementation variants (Flash Attention, etc.)
```

**Writing tips**:
- Focus on actionable performance insights
- Explain bottlenecks and why they occur
- Suggest concrete optimization strategies
- Compare with alternatives when relevant
- Length: 1-2 pages

---

### 11. References (H2) [Optional]

**Purpose**: Cite papers, documentation, and related resources.

**Content**:
```markdown
## References
- Original paper: "Paper Title" (Authors, Year)
  - Key contribution: [brief description]
  - Link: [URL if available]

- Implementation reference: [GitHub repo, docs]
- Related techniques: [other papers/methods]
- Used in: [list of models using this technique]

## See Also
- `op-RelatedModule1.md`: Similar functionality
- `op-RelatedModule2.md`: Extended version
```

**Writing tips**:
- Prioritize original papers and official docs
- Include brief descriptions, not just citations
- Link to GitHub repositories when available
- Note adoption in popular models
- Length: 5-10 references

---

## Complete Example Template

Save this as `op-YourModule.md`:

```markdown
# YourModule

## What It Is
[2-4 paragraphs explaining the module's purpose, context, and key features]

## Definition
\`\`\`python
class YourModule(nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, x):
        ...
\`\`\`

## Constructor Information
**Location**: `path/to/file.py:X-Y`

**Signature**: ...
**Parameters**: ...
**Created Components**: ...

## Module Internals
\`\`\`mermaid
sequenceDiagram
    ...
\`\`\`

## Key Pseudo Code
\`\`\`python
def forward(self, input):
    """Docstring"""
    # Step 1
    # Step 2
    ...
\`\`\`

## FLOP Count and Memory Usage Impact

### FLOPs
[Detailed breakdown with formulas and examples]

### Memory Usage
[Parameters, activations, gradients analysis]

## Related Modules
- **Used by**: ...
- **Uses**: ...

## Usage Pattern
\`\`\`python
# Example code
\`\`\`

## Key Performance Characteristics
1. ...
2. ...

## References
- ...
```

---

## Quality Checklist

Before finalizing a module documentation file, verify:

### Completeness
- [ ] All sections present (minimum: 1-9)
- [ ] File location with line numbers provided
- [ ] Constructor parameters documented with types and defaults
- [ ] Forward method explained with input/output shapes
- [ ] Mermaid diagram shows data flow clearly
- [ ] FLOP formula provided with concrete example
- [ ] Memory analysis includes parameters + activations
- [ ] At least one usage example provided

### Accuracy
- [ ] Code snippets are syntactically correct
- [ ] Tensor shapes are accurate throughout
- [ ] FLOP calculations verified (sum of operations)
- [ ] Memory calculations verified (shape × dtype size)
- [ ] Cross-references point to existing files
- [ ] Line numbers are current (not outdated)

### Clarity
- [ ] Technical terms explained on first use
- [ ] Formulas include variable definitions
- [ ] Examples use realistic dimensions
- [ ] Comparisons help contextualize the module
- [ ] Optimization opportunities are actionable

### Consistency
- [ ] Follows naming convention (`op-ModuleName.md`)
- [ ] Uses same section headers as template
- [ ] Mermaid diagram style matches other docs
- [ ] Code formatting is consistent (4-space indents)
- [ ] Dimensions use same variable names (B, S, d, etc.)

### Performance Focus
- [ ] Dominant FLOP term identified
- [ ] Peak memory point noted
- [ ] Bottlenecks explained
- [ ] Optimization opportunities listed
- [ ] Comparison with alternatives (if applicable)

---

## Tips for Efficient Documentation

### When to Use This Template

**Always use for**:
- Core neural network modules (attention, MLP, normalization, etc.)
- Custom architectures or novel techniques
- Modules with significant compute/memory costs
- Modules used in performance-critical paths

**Can abbreviate for**:
- Simple wrapper classes
- Utility functions (not nn.Module)
- Deprecated or rarely-used modules

### Common Patterns

**For Transformer Blocks**:
- Emphasize pre-norm vs post-norm architecture
- Document attention mask handling
- Explain residual connection patterns
- Show how KV cache is managed

**For Attention Mechanisms**:
- Break down into Q/K/V projection + attention + output projection
- Calculate attention matrix FLOPs: `2×B×H×S²×d`
- Document KV cache growth with sequence length
- Note Flash Attention variants and benefits

**For MLP/FFN Layers**:
- Show gate/up/down projection pattern (if SwiGLU)
- Calculate: `3 linear layers × 2×B×S×d×d_ff`
- Note activation function overhead
- Explain intermediate dimension expansion

**For Normalization**:
- Upcast to fp32 for stability (if applicable)
- Document running statistics (BatchNorm) vs no statistics (LayerNorm)
- Calculate `~5N` FLOPs for standard normalization

**For Vision Modules**:
- Include both 2D (NCHW) and sequence (BLC) perspectives
- Document patch size and spatial dimensions
- Show how spatial structure is preserved/lost
- Explain position embedding schemes

### FLOP Calculation Reference

Quick reference for common operations:

```
# Linear layer (input: d_in, output: d_out)
FLOPs = 2 × d_in × d_out × num_tokens

# Multi-head attention (full)
QKV projection:  3 × 2 × B × S × d × d    = 6×B×S×d²
Attention:       4 × B × H × S² × d       = 4×B×H×S²×d
Output proj:     2 × B × S × d × d        = 2×B×S×d²
Total:           6×B×S×d² + 4×B×H×S²×d + 2×B×S×d²

# Normalization
LayerNorm: ~5N (mean + var + normalize + scale + shift)
RMSNorm:   ~3N (var + normalize + scale)

# Activations (approximate)
GELU:      ~3N
SiLU:      ~2N
ReLU:      N

# Softmax
~5N (exp + sum + divide)

# Typical module sizes (B=1, S=8192, d=1280, d_ff=3584)
Single attention: ~100 GFLOPs
Single MLP:       ~120 GFLOPs
Full decoder layer: ~220 GFLOPs
40 layers:        ~8.8 TFLOPs (prefill)
```

### Memory Calculation Reference

```
# Parameter memory
params × sizeof(dtype)

Common dtypes:
- fp32 / float32: 4 bytes
- bf16 / bfloat16: 2 bytes
- fp16 / float16: 2 bytes
- int8: 1 byte

# Activation memory (inference)
All intermediate tensors (no gradients)

# Training memory
params + activations + param_gradients + activation_gradients + optimizer_states

Optimizer states (AdamW):
- momentum: params × 4 bytes (fp32)
- variance: params × 4 bytes (fp32)
Total: params × (param_dtype + 4 + 4)

# KV cache (attention)
2 × num_layers × batch × seq_len × num_kv_heads × head_dim × dtype
```

---

## Version History

- **v1.0** (2025-01-17): Initial template based on 34 DeepSeek-OCR module docs
- Created from: op-DeepseekV2RMSNorm.md, op-DeepseekV2Attention.md, op-DeepseekV2MoE.md, etc.
- Codifies best practices from comprehensive module documentation project

---

## Questions?

When in doubt:
1. Look at existing `op-*.md` files for examples
2. Prioritize FLOP/memory analysis (critical for optimization)
3. Include concrete numerical examples (B=1, S=8192, d=1280)
4. Cross-reference related modules
5. Explain "why" not just "what"

The goal is to create documentation that enables **performance optimization**, **architecture understanding**, and **computational budgeting** for neural network modules.
