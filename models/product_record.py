# python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProductRecord:
    product_id: int
    sale_code: str
    alias_name: str
    pic: str