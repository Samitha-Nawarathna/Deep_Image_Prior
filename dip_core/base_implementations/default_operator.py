
from dip_core.abstractions.operator import Operator

class IdentityOperator(Operator):
    def forward(self, x):
        return x
    