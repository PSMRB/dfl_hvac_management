import pyomo.environ as pyo
import pyomo.mpec as mpec
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from itertools import chain, combinations
from ..layers.full_space import _input_layer_and_block


def bigm_leakyrelu_activation_constraint(net_block, net, layer_block, layer, alpha):
    r"""
    Big-M Leaky ReLU activation formulation.

    Generates the constraints for the Leaky ReLU activation function.

    .. math::

        \begin{align*}
        z_i &= \text{max}(0, \hat{z_i}) && \forall i \in N
        \end{align*}

    The Big-M formulation for the i-th node is given by:

    .. math::

        \begin{align*}
        z_i &\geq \hat{z_i} \\
        z_i &\leq \hat{z_i} - l(1-\sigma) \\
        z_i &\leq u(\sigma) \\
        \sigma &\in \{0, 1\}
        \end{align*}

    where :math:`l` and :math:`u` are, respectively, lower and upper bounds of :math:`\hat{z_i}`.
    """
    # Create the binary variable for the ReLU activation
    layer_block.q_leakyrelu = pyo.Var(layer.output_indexes, within=pyo.Binary)
    # Record alpha as a parameter
    layer_block.alpha = pyo.Param(layer.output_indexes, initialize=alpha, mutable=False)

    # Create the Leaky ReLU activation constraints but do not define them yet
    layer_block._z_lower_bound_leakyrelu = pyo.Constraint(layer.output_indexes)
    layer_block._z_lower_bound_zhat_leakyrelu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_leakyrelu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_zhat_leakyrelu = pyo.Constraint(layer.output_indexes)

    # set dummy parameters here to avoid warning message from Pyomo
    layer_block._big_m_lb_leakyrelu = pyo.Param(
        layer.output_indexes, default=-1e6, mutable=True
    )
    layer_block._big_m_ub_leakyrelu = pyo.Param(
        layer.output_indexes, default=1e6, mutable=True
    )

    for output_index in layer.output_indexes:
        lb, ub = layer_block.zhat[output_index].bounds
        layer_block._big_m_lb_leakyrelu[output_index] = lb
        layer_block.z[output_index].setlb(0)

        layer_block._big_m_ub_leakyrelu[output_index] = ub
        layer_block.z[output_index].setub(max(0, ub))

        layer_block._z_lower_bound_leakyrelu[output_index] = (
                layer_block.z[output_index] >= layer_block.alpha[output_index] * layer_block.zhat[output_index])

        layer_block._z_lower_bound_zhat_leakyrelu[output_index] = (
                layer_block.z[output_index] >= layer_block.zhat[output_index]
        )

        layer_block._z_upper_bound_leakyrelu[output_index] = (
                layer_block.z[output_index]
                <= layer_block._big_m_ub_leakyrelu[output_index] * layer_block.q_leakyrelu[output_index]
                + layer_block.alpha[output_index] * layer_block.zhat[output_index]
        )

        layer_block._z_upper_bound_zhat_leakyrelu[output_index] = layer_block.z[
                                                                 output_index
                                                             ] <= layer_block.zhat[output_index] - \
                                                                  layer_block._big_m_lb_leakyrelu[
                                                                 output_index
                                                             ] * (
                                                                     1.0 - layer_block.q_leakyrelu[output_index]
                                                             )