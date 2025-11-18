from dip_core.abstractions.operator import Operator

class DefaultOperator(Operator):
    def forward(self, x):
        return x
    