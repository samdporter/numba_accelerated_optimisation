from cil.optimisation.functions import OperatorCompositionFunction

def proximal_ocf(self, x, tau, out=None):
    """
    Proximal operator of the operator composition function
    :param x: input DataContainer
    :param tau: step size
    :param out: optional output DataContainer
    :return: out

    maths: out = x + A^T (prox_{F}(Ax)-Ax)

    """

    if out is None:
        out = x.copy()

    tmp = self.operator.direct(x)
    self.function.proximal(tmp, tau, out=tmp)
    tmp -= self.operator.direct(x)
    self.operator.adjoint(tmp, out=out)
    out += x

    return out