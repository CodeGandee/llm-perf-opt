"""Contract/domain conversion utilities using `cattrs`.

Provides a shared converter for transforming between domain models and
contract models.
"""

from __future__ import annotations

from cattrs import Converter

# Public converter instance; register hooks as needed.
converter = Converter()

