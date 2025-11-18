# How to Use Hydra, OmegaConf, and fvcore Registry for Dynamic Object Construction

This guide demonstrates the recommended implementation pattern for dynamically constructing Python objects using Hydra configuration, OmegaConf's `DictConfig`, and fvcore's registry system.

## Overview

The pattern combines three powerful tools:
- **Hydra**: Configuration framework with `instantiate()` for object creation
- **OmegaConf**: Type-safe configuration with `DictConfig` and structured configs
- **fvcore Registry**: Decorator-based registration system for component discovery

This approach is widely used in projects like Detectron2 and enables flexible, configuration-driven object construction with strong typing.

## Pattern Components

### 1. Registry Setup with fvcore

Create a registry for your component type using fvcore:

```python
from fvcore.common.registry import Registry

# Create a registry for attention layers
ATTENTION_REGISTRY = Registry("ATTENTION")
ATTENTION_REGISTRY.__doc__ = """
Registry for attention layer implementations.
"""
```

### 2. Component Registration

Register your classes using the decorator pattern. There are two common patterns:

#### Pattern A: Parameterized `__init__()` (Traditional)

Direct instantiation via `__init__` with all parameters:

```python
from fvcore.common.registry import Registry
from omegaconf import DictConfig

ATTENTION_REGISTRY = Registry("ATTENTION")

@ATTENTION_REGISTRY.register()
class SimpleAttention:
    """Attention with direct __init__ parameters."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_area: int,
        num_windows: int,
        use_rel_pos: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.window_area = window_area
        self.num_windows = num_windows
        self.use_rel_pos = use_rel_pos
        self.use_flash_attention = use_flash_attention
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SimpleAttention":
        """Optional: Provide config-based factory for convenience."""
        return cls(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            window_area=cfg.window_area,
            num_windows=cfg.num_windows,
            use_rel_pos=cfg.get("use_rel_pos", True),
            use_flash_attention=cfg.get("use_flash_attention", True),
        )
```

**Pros**: Works directly with Hydra's `instantiate()`, simple and straightforward.  
**Cons**: Single construction pattern, less flexible for complex initialization.

#### Pattern B: Factory Constructor Pattern (Flexible)

Empty `__init__()` with factory classmethods for construction:

```python
from typing import Optional

@ATTENTION_REGISTRY.register()
class Attention:
    """Attention using factory constructor pattern."""
    
    def __init__(self) -> None:
        """Initialize empty instance. Use factory constructors to configure."""
        self.m_dim: Optional[int] = None
        self.m_num_heads: Optional[int] = None
        self.m_window_area: Optional[int] = None
        self.m_num_windows: Optional[int] = None
        self.m_use_rel_pos: bool = True
        self.m_use_flash_attention: bool = True
    
    def set_config(
        self,
        *,
        dim: int,
        num_heads: int,
        window_area: int,
        num_windows: int,
        use_rel_pos: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        """Configure the attention layer after construction."""
        self.m_dim = dim
        self.m_num_heads = num_heads
        self.m_window_area = window_area
        self.m_num_windows = num_windows
        self.m_use_rel_pos = use_rel_pos
        self.m_use_flash_attention = use_flash_attention
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Attention":
        """Factory constructor from Hydra/OmegaConf config."""
        instance = cls()
        instance.set_config(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            window_area=cfg.window_area,
            num_windows=cfg.num_windows,
            use_rel_pos=cfg.get("use_rel_pos", True),
            use_flash_attention=cfg.get("use_flash_attention", True),
        )
        return instance
    
    @classmethod
    def from_global_config(
        cls,
        *,
        dim: int,
        num_heads: int,
        height: int,
        width: int,
        use_rel_pos: bool = True,
        use_flash_attention: bool = True,
    ) -> "Attention":
        """Factory for global attention (single window over full H×W)."""
        instance = cls()
        instance.set_config(
            dim=dim,
            num_heads=num_heads,
            window_area=height * width,
            num_windows=1,
            use_rel_pos=use_rel_pos,
            use_flash_attention=use_flash_attention,
        )
        return instance
```

**Pros**: Multiple factory methods for different construction scenarios, clean separation of construction vs. configuration, easier testing.  
**Cons**: Requires custom builder functions for Hydra integration, slightly more verbose.

#### Explicit Name Registration

For both patterns, you can specify custom registry names:

```python
@ATTENTION_REGISTRY.register("custom_attention")
class MyCustomAttention:
    pass
```

### 3. Structured Config with OmegaConf

Define typed configuration using dataclasses:

```python
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class AttentionConfig:
    """Structured config for attention layers."""
    _target_: str = MISSING  # Required for Hydra instantiate
    dim: int = MISSING
    num_heads: int = MISSING
    window_area: int = MISSING
    num_windows: int = MISSING
    use_rel_pos: bool = True
    use_flash_attention: bool = True


@dataclass
class ModelConfig:
    """Parent config with nested attention config."""
    name: str = "vision_model"
    attention: AttentionConfig = field(default_factory=AttentionConfig)
```

### 4. YAML Configuration Files

Create Hydra config groups for different attention variants. The approach depends on which class pattern you use.

#### For Pattern A: Parameterized `__init__()`

Direct instantiation with `_target_` pointing to the class:

**conf/attention/simple_standard.yaml**
```yaml
_target_: myproject.layers.attention.SimpleAttention
dim: 768
num_heads: 12
window_area: 196
num_windows: 25
use_rel_pos: true
use_flash_attention: false
```

This works directly with `hydra.utils.instantiate()` - Hydra passes all parameters to `__init__()`.

#### For Pattern B: Factory Constructor Pattern

You have three options:

**Option B1: Target the factory classmethod directly**

**conf/attention/standard.yaml**
```yaml
_target_: myproject.layers.attention.Attention.from_config
cfg:
  dim: 768
  num_heads: 12
  window_area: 196
  num_windows: 25
  use_rel_pos: true
  use_flash_attention: false
```

**Option B2: Use a builder function (Recommended)**

**conf/attention/standard.yaml**
```yaml
_target_: myproject.builders.build_attention
name: Attention
dim: 768
num_heads: 12
window_area: 196
num_windows: 25
use_rel_pos: true
use_flash_attention: false
```

**Option B3: Use alternative factory method**

**conf/attention/global.yaml**
```yaml
_target_: myproject.layers.attention.Attention.from_global_config
dim: 768
num_heads: 12
height: 64
width: 64
use_rel_pos: true
use_flash_attention: true
```

**conf/model/base.yaml**
```yaml
name: vision_encoder
attention:
  _target_: myproject.layers.attention.Attention
  dim: 768
  num_heads: 12
  window_area: 4096
  num_windows: 1
```

### 5. Object Instantiation with Hydra

#### Method A: Using `hydra.utils.instantiate()` (For Pattern A & B)

The most common approach - use Hydra's built-in instantiation:

```python
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Pattern A: Direct instantiation (calls __init__ with params)
    # Works when _target_: myproject.layers.attention.SimpleAttention
    simple_attention = instantiate(cfg.simple_attention)
    print(f"Created: {simple_attention}")
    
    # Pattern B: Factory method instantiation
    # Works when _target_: myproject.layers.attention.Attention.from_config
    # or when _target_: myproject.builders.build_attention
    factory_attention = instantiate(cfg.attention)
    
    # Override parameters at runtime (Pattern A only)
    attention_custom = instantiate(cfg.simple_attention, num_heads=16)
    
    # For Pattern B, use OmegaConf.merge for overrides (see Method B below)
    
    # Recursive instantiation for nested configs (works for both patterns)
    model = instantiate(cfg.model)  # Automatically instantiates nested attention
```

#### Method B: Registry + Pattern-Aware Builder

Combine registry lookup with support for both patterns:

```python
from omegaconf import DictConfig
from hydra.utils import instantiate
import inspect

def build_attention_from_registry(cfg: DictConfig):
    """Build attention using registry, supporting both construction patterns.
    
    Automatically detects whether to use __init__ or factory constructor.
    """
    # Option 1: Direct instantiate if _target_ is specified
    if "_target_" in cfg:
        return instantiate(cfg)
    
    # Option 2: Use registry name and detect pattern
    attention_name = cfg.get("name", "Attention")
    attention_cls = ATTENTION_REGISTRY.get(attention_name)
    
    # Detect which pattern the class uses
    init_sig = inspect.signature(attention_cls.__init__)
    has_params = len(init_sig.parameters) > 1  # More than just 'self'
    
    if has_params:
        # Pattern A: Parameterized __init__
        params = {k: v for k, v in cfg.items() 
                  if not k.startswith("_") and k != "name"}
        return attention_cls(**params)
    else:
        # Pattern B: Factory constructor
        if hasattr(attention_cls, "from_config"):
            return attention_cls.from_config(cfg)
        else:
            raise ValueError(f"{attention_cls.__name__} has no-arg __init__ "
                           f"but no from_config factory method")
```

#### Method C: Unified Builder Function (Recommended)

Create a builder function that handles both patterns seamlessly:

```python
from omegaconf import DictConfig
from typing import Optional
import inspect

def build_attention(cfg: DictConfig, registry: Optional[Registry] = None):
    """Build attention layer from config supporting both construction patterns.
    
    This builder works for:
    - Pattern A: Classes with parameterized __init__()
    - Pattern B: Classes with factory constructors (from_config, etc.)
    
    Args:
        cfg: Configuration with either _target_ or name field
        registry: Optional registry override (defaults to ATTENTION_REGISTRY)
    
    Returns:
        Instantiated attention layer
    """
    if registry is None:
        registry = ATTENTION_REGISTRY
    
    # Strategy 1: Use _target_ with Hydra instantiate (works for both patterns)
    if "_target_" in cfg:
        return instantiate(cfg)
    
    # Strategy 2: Use registry lookup
    if "name" not in cfg:
        raise ValueError("Config must specify either '_target_' or 'name'")
    
    attention_cls = registry.get(cfg.name)
    
    # Detect which pattern the class uses
    init_sig = inspect.signature(attention_cls.__init__)
    has_params = len(init_sig.parameters) > 1  # More than just 'self'
    
    if has_params:
        # Pattern A: Call __init__ with parameters
        params = {k: v for k, v in cfg.items() 
                  if not k.startswith("_") and k != "name"}
        return attention_cls(**params)
    else:
        # Pattern B: Call factory constructor
        if hasattr(attention_cls, "from_config"):
            return attention_cls.from_config(cfg)
        else:
            raise ValueError(f"{attention_cls.__name__} requires factory constructor")
```

### 6. Repeated Instantiation (Both Patterns)

#### For Pattern A: Use Hydra's `_partial_`

```python
from hydra.utils import instantiate

# Create partial factory - faster for repeated instantiation
factory = instantiate(cfg.simple_attention, _partial_=True)

# Create multiple instances
attention1 = factory()
attention2 = factory(num_heads=16)  # Override specific params
attention3 = factory(num_heads=24, dim=1024)
```

**Note**: Nested objects are reused across factory calls for efficiency.

#### For Pattern B: Custom Factory Function

```python
from omegaconf import OmegaConf
from functools import partial

def make_attention_factory(base_cfg: DictConfig):
    """Return a factory function for creating attention instances."""
    cls = ATTENTION_REGISTRY.get(base_cfg.get("name", "Attention"))
    
    def factory(**overrides):
        """Create attention with optional parameter overrides."""
        if overrides:
            cfg = OmegaConf.merge(base_cfg, overrides)
        else:
            cfg = base_cfg
        return cls.from_config(cfg)
    
    return factory

# Usage
factory = make_attention_factory(cfg.attention)
attn1 = factory()
attn2 = factory(num_heads=16)
attn3 = factory(num_heads=24, dim=1024)
```

#### Pattern Comparison for Factories

| Aspect | Pattern A | Pattern B |
|--------|-----------|-----------|
| Factory creation | `instantiate(cfg, _partial_=True)` | Custom factory function |
| Override syntax | `factory(param=value)` | `factory(param=value)` with merge |
| Complexity | Simpler | More control |
| Performance | Faster (Hydra optimized) | Good (manual merge) |

### 7. Complete Integration Example

Here's a complete example integrating all components:

```python
# myproject/registry.py
from fvcore.common.registry import Registry

ATTENTION_REGISTRY = Registry("ATTENTION")
LAYER_REGISTRY = Registry("LAYER")

# myproject/layers/attention.py
from dataclasses import dataclass
from omegaconf import MISSING
from myproject.registry import ATTENTION_REGISTRY

@dataclass
class AttentionConfig:
    _target_: str = "myproject.layers.attention.Attention"
    dim: int = MISSING
    num_heads: int = MISSING
    window_area: int = MISSING
    num_windows: int = MISSING
    use_rel_pos: bool = True
    use_flash_attention: bool = True


@ATTENTION_REGISTRY.register()
class SimpleAttention:
    """Pattern A: Parameterized __init__."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_area: int,
        num_windows: int,
        use_rel_pos: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.window_area = window_area
        self.num_windows = num_windows
        self.use_rel_pos = use_rel_pos
        self.use_flash_attention = use_flash_attention


@ATTENTION_REGISTRY.register()
class Attention:
    """Pattern B: Factory constructor."""
    
    def __init__(self) -> None:
        """Initialize empty instance. Use factory constructors to configure."""
        self.m_dim: Optional[int] = None
        self.m_num_heads: Optional[int] = None
        self.m_window_area: Optional[int] = None
        self.m_num_windows: Optional[int] = None
        self.m_use_rel_pos: bool = True
        self.m_use_flash_attention: bool = True
    
    def set_config(
        self,
        *,
        dim: int,
        num_heads: int,
        window_area: int,
        num_windows: int,
        use_rel_pos: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        """Configure the attention layer."""
        self.m_dim = dim
        self.m_num_heads = num_heads
        self.m_window_area = window_area
        self.m_num_windows = num_windows
        self.m_use_rel_pos = use_rel_pos
        self.m_use_flash_attention = use_flash_attention
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Attention":
        """Factory constructor from Hydra config (recommended pattern)."""
        instance = cls()
        instance.set_config(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            window_area=cfg.window_area,
            num_windows=cfg.num_windows,
            use_rel_pos=cfg.get("use_rel_pos", True),
            use_flash_attention=cfg.get("use_flash_attention", True),
        )
        return instance
    
    @classmethod
    def from_global_config(
        cls,
        *,
        dim: int,
        num_heads: int,
        height: int,
        width: int,
        use_rel_pos: bool = True,
        use_flash_attention: bool = True,
    ) -> "Attention":
        """Factory for global attention (single window over full H×W)."""
        instance = cls()
        instance.set_config(
            dim=dim,
            num_heads=num_heads,
            window_area=height * width,
            num_windows=1,
            use_rel_pos=use_rel_pos,
            use_flash_attention=use_flash_attention,
        )
        return instance


# myproject/builders.py
from omegaconf import DictConfig
from hydra.utils import instantiate
from myproject.registry import ATTENTION_REGISTRY

def build_attention(cfg: DictConfig):
    """Build attention supporting both construction patterns.
    
    Handles:
    - Pattern A: Classes with parameterized __init__()
    - Pattern B: Classes with factory constructors (from_config)
    """
    # Strategy 1: Use _target_ (works for both patterns)
    if "_target_" in cfg:
        return instantiate(cfg)
    
    # Strategy 2: Registry lookup with pattern detection
    if "name" not in cfg:
        raise ValueError("Config must have '_target_' or 'name'")
    
    cls = ATTENTION_REGISTRY.get(cfg.name)
    
    # Check which pattern the class uses
    init_sig = inspect.signature(cls.__init__)
    has_params = len(init_sig.parameters) > 1
    
    if has_params:
        # Pattern A: Call __init__ directly
        params = {k: v for k, v in cfg.items() 
                  if not k.startswith("_") and k != "name"}
        return cls(**params)
    else:
        # Pattern B: Call factory constructor
        return cls.from_config(cfg)


# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
from myproject.builders import build_attention

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))
    
    # Pattern A: Direct instantiate with __init__ params
    simple_attn = instantiate(cfg.simple_attention)
    
    # Pattern B: Instantiate via builder or factory method
    factory_attn = instantiate(cfg.attention)
    
    # Unified builder works for both
    unified_simple = build_attention(cfg.simple_attention)
    unified_factory = build_attention(cfg.attention)
    
    # Overrides
    # Pattern A: Direct parameter override
    simple_override = instantiate(cfg.simple_attention, num_heads=24)
    
    # Pattern B: Config merge before building
    cfg_override = OmegaConf.merge(cfg.attention, {"num_heads": 24})
    factory_override = build_attention(cfg_override)
    
    print(f"Pattern A: {simple_attn}, {unified_simple}, {simple_override}")
    print(f"Pattern B: {factory_attn}, {unified_factory}, {factory_override}")

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Use `_target_` for Hydra Integration
Always include `_target_` in configs that will be used with `instantiate()`:

```yaml
attention:
  _target_: myproject.layers.attention.Attention
  dim: 768
```

### 2. Combine Registry with Hydra (Don't Replace)
Registry is for **discovery** and **organization**; Hydra is for **instantiation**:

```python
# Good: Use both together
@ATTENTION_REGISTRY.register()
class Attention:
    pass

# In YAML
_target_: myproject.layers.Attention  # Hydra finds it
```

### 3. Type Safety with Structured Configs
Use dataclasses for type checking and IDE support:

```python
@dataclass
class AttentionConfig:
    dim: int = MISSING  # Required field
    num_heads: int = 8   # With default
```

### 4. Choosing Between Patterns

#### Use Pattern A (Parameterized `__init__`) When:
- Simple, single-purpose construction
- All parameters available at construction time
- Direct Hydra instantiation is sufficient
- You want minimal boilerplate

```python
class SimpleLayer:
    def __init__(self, dim: int, num_heads: int):
        self.dim = dim
        self.num_heads = num_heads
```

**Pros**: Simple, works directly with Hydra, less code  
**Cons**: Single construction path, harder to add alternative construction methods

#### Use Pattern B (Factory Constructors) When:
- Multiple construction scenarios (e.g., from_config, from_global_config)
- Complex initialization with derived parameters
- Need to separate construction from configuration
- Prefer explicit construction patterns

```python
class ComplexLayer:
    def __init__(self) -> None:
        """Empty init - use factory constructors."""
        self.m_dim: Optional[int] = None
        self.m_num_heads: Optional[int] = None
    
    def set_config(self, *, dim: int, num_heads: int) -> None:
        """Configure after construction."""
        self.m_dim = dim
        self.m_num_heads = num_heads
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ComplexLayer":
        """Build from Hydra config."""
        instance = cls()
        instance.set_config(dim=cfg.dim, num_heads=cfg.num_heads)
        return instance
    
    @classmethod
    def from_dimensions(cls, height: int, width: int) -> "ComplexLayer":
        """Alternative factory with derived parameters."""
        instance = cls()
        dim = height * width
        instance.set_config(dim=dim, num_heads=dim // 64)
        return instance
```

**Pros**: Multiple construction paths, clear separation, easier testing, flexible  
**Cons**: More boilerplate, requires builder functions for Hydra integration

### 5. Runtime Overrides (Both Patterns)

#### For Pattern A (Parameterized `__init__`)

Direct override with Hydra's instantiate:

```bash
# Command line override
python main.py simple_attention.num_heads=16 simple_attention.dim=1024
```

```python
# In code - pass kwargs to instantiate
attention = instantiate(cfg.simple_attention, num_heads=16, dim=1024)
```

#### For Pattern B (Factory Constructor)

Use `OmegaConf.merge()` for overrides:

```bash
# Command line override (works automatically)
python main.py attention.num_heads=16 attention.dim=1024
```

```python
# In code - merge configs before building
from omegaconf import OmegaConf

cfg_override = OmegaConf.merge(cfg.attention, {"num_heads": 16, "dim": 1024})
attention = build_attention(cfg_override)

# Or create new config on-the-fly
override_cfg = OmegaConf.create({
    "name": "Attention",
    "dim": 1024,
    "num_heads": 16,
    "window_area": 196,
    "num_windows": 25,
})
attention = build_attention(override_cfg)
```

**Key Difference**: Pattern A supports `instantiate(cfg, **overrides)`, but Pattern B requires merging configs first.

## Common Patterns

### Pattern 1: Multi-level Registry Hierarchy

```python
# Base registry
MODEL_REGISTRY = Registry("MODEL")

# Component registries
BACKBONE_REGISTRY = Registry("BACKBONE")
ATTENTION_REGISTRY = Registry("ATTENTION")
HEAD_REGISTRY = Registry("HEAD")

@MODEL_REGISTRY.register()
class VisionModel:
    def __init__(self, cfg: DictConfig):
        self.backbone = instantiate(cfg.backbone)
        self.attention = instantiate(cfg.attention)
        self.head = instantiate(cfg.head)
```

### Pattern 2: Conditional Registration

```python
# Register different implementations based on conditions
if torch.cuda.is_available():
    @ATTENTION_REGISTRY.register()
    class CUDAAttention(Attention):
        pass
else:
    @ATTENTION_REGISTRY.register()
    class CPUAttention(Attention):
        pass
```

### Pattern 3: Config Groups with Defaults

```yaml
# conf/config.yaml
defaults:
  - attention: standard
  - _self_

model:
  name: my_model
```

```bash
# Override attention variant
python main.py attention=flash
```

## Advanced: Convert Between Config Formats

### DictConfig to Dict

```python
from omegaconf import OmegaConf

# Convert strategies
cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Resolve interpolations
cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
```

### Structured Config to DictConfig

```python
from omegaconf import OmegaConf, DictConfig

# From dataclass instance
cfg: DictConfig = OmegaConf.structured(AttentionConfig(dim=768, num_heads=12))

# From dataclass type with defaults
cfg: DictConfig = OmegaConf.structured(AttentionConfig)
```

## Troubleshooting

### Issue: "Could not locate target"
**Problem**: Hydra can't find the class specified in `_target_`

**Solution**: Ensure the module is imported before instantiation:
```python
import myproject.layers.attention  # Import to register
attention = instantiate(cfg.attention)
```

### Issue: Type validation errors with MISSING
**Problem**: `MISSING` fields not provided

**Solution**: Use structured config with required fields:
```python
@dataclass
class Config:
    required_field: int = MISSING  # Must be provided
    optional_field: int = 10        # Has default
```

### Issue: Registry not finding registered class
**Problem**: Decorator executed but class not in registry

**Solution**: Import the module containing `@register()` decorated classes:
```python
# In __init__.py or main entry point
import myproject.layers.attention  # Executes decorators
```

## References

- [Hydra Documentation: Instantiating Objects](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
- [OmegaConf Documentation: Structured Configs](https://omegaconf.readthedocs.io/en/latest/structured_config.html)
- [fvcore Registry Documentation](https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/registry.py)
- [Detectron2: Write Models Tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/write-models.html)
- [Registry Pattern in Python](https://charlesreid1.github.io/python-patterns-the-registry.html)

## Related Hints

- `howto-hydra-compose-configs.md` - Composing complex configurations
- `howto-omegaconf-interpolation.md` - Using variable interpolation
- `about-factory-pattern-python.md` - Factory pattern fundamentals
