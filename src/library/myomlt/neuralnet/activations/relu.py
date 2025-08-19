import pyomo.environ as pyo
import pyomo.mpec as mpec
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from itertools import chain, combinations
from ..layers.full_space import _input_layer_and_block


def bigm_relu_activation_constraint(net_block, net, layer_block, layer):
    r"""
    Big-M ReLU activation formulation.

    Generates the constraints for the ReLU activation function.

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
    # Create the binary variable
    layer_block.q_leakyrelu = pyo.Var(layer.output_indexes, within=pyo.Binary)

    # Create the constraints but does not define them yet
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

        layer_block._z_lower_bound_leakyrelu[output_index] = layer_block.z[output_index] >= 0

        layer_block._z_lower_bound_zhat_leakyrelu[output_index] = (
                layer_block.z[output_index] >= layer_block.zhat[output_index]
        )

        layer_block._z_upper_bound_leakyrelu[output_index] = (
                layer_block.z[output_index]
                <= layer_block._big_m_ub_leakyrelu[output_index]
                * layer_block.q_leakyrelu[output_index]
        )

        layer_block._z_upper_bound_zhat_leakyrelu[output_index] = layer_block.z[
                                                                 output_index
                                                             ] <= layer_block.zhat[output_index] - \
                                                                  layer_block._big_m_lb_leakyrelu[
                                                                 output_index
                                                             ] * (
                                                                     1.0 - layer_block.q_leakyrelu[output_index]
                                                             )


def ideal_relu_activation_constraint(net_block, net, layer_block, layer):
    r"""
    Ideal ReLU activation formulation.
    The ideal ReLU activation formulation is given by the Big-M formulation with
    an additional constraints. See paper: Strong Mixed-Integer formulation for trained neural networks
    by Anderson et al. (2020). Final version on Mathematical Programming, 2020.
    Equations 28a is the additional constraint.

    Generates the constraints for the ReLU activation function.

    .. math::

        \begin{align*}
        z_i &= \text{max}(0, \hat{z_i}) && \forall i \in N
        \end{align*}

    The ideal formulation for the i-th node is given by:

    .. math::

        To be replaced by the actual formulation

    where :math:`l` and :math:`u` are, respectively, lower and upper bounds of :math:`\hat{z_i}`.
    """

    # get all subsets of a set
    def all_subsets_and_complements(s):
        all_subsets = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
        complements = [tuple(x for x in s if x not in subset) for subset in all_subsets]
        return all_subsets, complements

    # define a set which has the size of the number of subsets
    subsets, complements = list(all_subsets_and_complements(layer.input_indexes))
    layer_block.subset_indexes = pyo.Set(initialize=range(len(subsets)), ordered=True)

    # import previous layer
    _, previous_layer_block = _input_layer_and_block(net_block, net, layer)

    # define one binary for each relu in the layer
    layer_block.q_leakyrelu = pyo.Var(layer.output_indexes, within=pyo.Binary)

    # define the constraints for each relu in the layer
    layer_block._z_lower_bound_leakyrelu = pyo.Constraint(layer.output_indexes)  # Eq. 28b
    layer_block._z_lower_bound_zhat_leakyrelu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_leakyrelu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_zhat_leakyrelu = pyo.Constraint(layer.output_indexes)
    layer_block._z_scd_upper_bound_relu = pyo.Constraint(layer.output_indexes,
                                                         layer_block.subset_indexes)  # Eq. 28a of the final paper

    # set dummy parameters here to avoid warning message from Pyomo
    layer_block._big_m_lb_leakyrelu = pyo.Param(
        layer.output_indexes, default=-1e6, mutable=True
    )
    layer_block._big_m_ub_leakyrelu = pyo.Param(
        layer.output_indexes, default=1e6, mutable=True
    )
    # one pair of bounds for each weight w_i of the weight matrix
    layer_block.L_pos = pyo.Param(
        layer.input_indexes, layer.output_indexes, default=-1e6, mutable=True
    )
    layer_block.U_pos = pyo.Param(
        layer.input_indexes, layer.output_indexes, default=1e6, mutable=True
    )
    # compute the bounds for the input of the layer DenseLayer + relu (i.e., the output of the previous layer)
    for input_index in layer.input_indexes:
        lb, ub = previous_layer_block.z[input_index].bounds  # z of the previous layer are our x
        for output_index in layer.output_indexes:
            weight = layer.weights[input_index, output_index]
            if weight >= 0:
                layer_block.L_pos[input_index, output_index] = lb
                layer_block.U_pos[input_index, output_index] = ub
            else:
                layer_block.L_pos[input_index, output_index] = ub
                layer_block.U_pos[input_index, output_index] = lb

    # for each relu in the layer, define the constraints
    for output_index in layer.output_indexes:
        # get the bounds of the input and propagate them to the output
        lb, ub = layer_block.zhat[output_index].bounds
        layer_block._big_m_lb_leakyrelu[output_index] = lb
        layer_block.z[output_index].setlb(0)

        layer_block._big_m_ub_leakyrelu[output_index] = ub
        layer_block.z[output_index].setub(max(0, ub))

        layer_block._z_lower_bound_leakyrelu[output_index] = layer_block.z[output_index] >= 0

        layer_block._z_lower_bound_zhat_leakyrelu[output_index] = (
                layer_block.z[output_index] >= layer_block.zhat[output_index]
        )

        layer_block._z_upper_bound_leakyrelu[output_index] = (
                layer_block.z[output_index]
                <= layer_block._big_m_ub_leakyrelu[output_index]
                * layer_block.q_leakyrelu[output_index]
        )

        layer_block._z_upper_bound_zhat_leakyrelu[output_index] = layer_block.z[
                                                                 output_index
                                                             ] <= layer_block.zhat[output_index] - \
                                                                  layer_block._big_m_lb_leakyrelu[
                                                                 output_index
                                                             ] * (
                                                                     1.0 - layer_block.q_leakyrelu[output_index]
                                                             )

        # set the upper bound of the output (z <= sum(w_i * (x_i - L_i * (1 - q_i)) + (b_i + sum(w_i * U_i))*q_i))
        for subset_index in layer_block.subset_indexes:
            subset = subsets[subset_index]
            complement = complements[subset_index]
            layer_block._z_scd_upper_bound_relu[output_index, subset_index] = (
                    layer_block.z[output_index]  # y
                    <= sum(
                layer.weights[input_index, output_index] * (
                        previous_layer_block.z[input_index] - layer_block.L_pos[input_index, output_index] * (
                        1 - layer_block.q_leakyrelu[output_index])
                )
                for input_index in subset
            )
                    + (
                            layer.biases[output_index] + sum(
                        layer.weights[input_index, output_index] * layer_block.U_pos[input_index, output_index]
                        for input_index in complement
                    )
                    ) * layer_block.q_leakyrelu[output_index]
            )


def extended_relu_activation_constraint1(net_block, net, layer_block, layer):
    r"""
    Extended ReLU activation formulation.
    The extended ReLU activation formulation is given by adding two continuous variables and one binary per ReLU.
    See paper: Strong Mixed-Integer formulation for trained neural networks
    by Anderson et al. (2020). Final version on Mathematical Programming, 2020.
    Equations 7a-7f. Note that there is no need for y0. So y = y1 (no need for y1 either)

    Generates the constraints for the ReLU activation function.

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

    # import previous layer
    _, previous_layer_block = _input_layer_and_block(net_block, net, layer)

    # define one binary for each relu in the layer
    layer_block.q_leakyrelu = pyo.Var(layer.output_indexes, within=pyo.Binary)

    # define two variables for each input of the layer block (=DenseLayer + relu)
    layer_block.za = pyo.Var(layer.output_indexes, within=pyo.Reals)

    # TODO: Avoid creating zhat in the dense layer before extended formulation is called
    # layer_block._z_lower_bound_relu = pyo.Constraint(layer.output_indexes)  # z >= 0
    layer_block._z_eq_wxbq_relu = pyo.Constraint(layer.output_indexes)  # z == z_b + b*(1-q), x = z previous layer
    layer_block._leq_wxbq_relu = pyo.Constraint(layer.output_indexes)  # 0 >= z_a + b*(q)
    layer_block._zb_lb_relu = pyo.Constraint(layer.output_indexes)  # [L(1-q) <= z_b] <= U*(1-q)
    layer_block._zb_ub_relu = pyo.Constraint(layer.output_indexes)  # L(1-q) <= [z_b <= U*(1-q)]
    layer_block._za_lb_relu = pyo.Constraint(layer.output_indexes)  # [L*q <= z_a] <= U*q
    layer_block._za_ub_relu = pyo.Constraint(layer.output_indexes)  # L*q <= [z_a <= U*q]
    layer_block._zazbwx_eq_relu = pyo.Constraint(layer.output_indexes)  # z == w*(x-x_0) + b*q, x = z previous layer

    # parameters
    layer_block.L = pyo.Param(layer.output_indexes, default=-1e6, mutable=True)
    layer_block.U = pyo.Param(layer.output_indexes, default=1e6, mutable=True)

    # compute the bounds for the input of the layer DenseLayer + relu (i.e., the output of the previous layer)
    for output_index in layer.output_indexes:
        # get the bounds of the input and propagate them to the output
        # z >= 0
        lb, ub = layer_block.zhat[output_index].bounds
        layer_block.z[output_index].setlb(0)
        layer_block.z[output_index].setub(max(0, ub))
        # Compute bounds without the bias: bounds for za and zb
        layer_block.L[output_index] = lb - layer.biases[output_index]
        layer_block.U[output_index] = ub - layer.biases[output_index]

        # compute the dense layer results
        def dense_layer_bq(x1, x2, q, layer, output_index):
            """
            Function to structure the code and avoid code duplication.
            X is the input variable of the dense layer. In our case, we split z from the input layer (i.e., the
            previous layer) into x0 and x-x0.
            Here, we add a binary condition to the biais.
            :return: w*Xi + b*q
            """
            # dense layers multiply only the last dimension of their inputs
            expr = 0.0
            for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
                w = layer.weights[local_index[-1], output_index[-1]]
                if x2 is None:
                    expr += x1[input_index] * w
                else:
                    expr += (x1[input_index] - x2[input_index]) * w
            # move this at the end to avoid numpy/pyomo var bug
            expr += layer.biases[output_index[-1]] * q

            return expr

        # x*w
        xw = dense_layer_bq(previous_layer_block.z, None, 0, layer, output_index)

        # Split the input variables and constraints them
        # L(1-q) <= zb <= U*(1-q)
        layer_block._zb_lb_relu[output_index] = ((layer_block.L[output_index] * (1 - layer_block.q_leakyrelu[output_index]))
                                                 <= xw - layer_block.za[output_index])
        layer_block._zb_ub_relu[output_index] = (xw - layer_block.za[output_index] <=
                                                 (layer_block.U[output_index] * (1 - layer_block.q_leakyrelu[output_index])))
        # L*q <= za <= U*q
        layer_block._za_lb_relu[output_index] = ((layer_block.L[output_index] * layer_block.q_leakyrelu[output_index]) <=
                                                 (layer_block.za[output_index]))
        layer_block._za_ub_relu[output_index] = ((layer_block.za[output_index]) <=
                                                 (layer_block.U[output_index] * layer_block.q_leakyrelu[output_index]))

        TMP0 = lb * (1 - layer_block.q_leakyrelu[output_index])
        TMP1 = ub * (1 - layer_block.q_leakyrelu[output_index])
        TMP2 = lb * layer_block.q_leakyrelu[output_index]
        TMP3 = ub * layer_block.q_leakyrelu[output_index]

        if lb >= 0:
            stop = 0
            print(lb)
            TMP4 = lb
        elif ub <= 0:
            stop = 0
            print(ub)
            TMP4 = ub

        # z == zb + b*(1-q)
        layer_block._z_eq_wxbq_relu[output_index] = (layer_block.z[output_index] ==
                                                     (xw - layer_block.za[output_index]) +
                                                     layer.biases[output_index] *
                                                     (1 - layer_block.q_leakyrelu[output_index]))
        # 0 >= za + b*(q)
        layer_block._leq_wxbq_relu[output_index] = (
                0 >= layer_block.za[output_index] + layer.biases[output_index] * layer_block.q_leakyrelu[output_index])
        # x*w = za + zb
        layer_block._zazbwx_eq_relu[output_index] = (layer_block.za[output_index] + (xw - layer_block.za[output_index])
                                                     == xw)
        TMP7 = (layer_block.za[output_index] + layer_block.za[output_index] == xw)

        Stop = 0

def extended_relu_activation_constraint2(net_block, net, layer_block, layer):
    r"""
    Extended ReLU activation formulation.
    The extended ReLU activation formulation is given by adding two continuous variables and one binary per ReLU.
    See paper: Strong Mixed-Integer formulation for trained neural networks
    by Anderson et al. (2020). Final version on Mathematical Programming, 2020.
    Equations 7a-7f. Note that there is no need for y0. So y = y1 (no need for y1 either)

    Generates the constraints for the ReLU activation function.

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

    # import previous layer
    _, previous_layer_block = _input_layer_and_block(net_block, net, layer)

    # define one binary for each relu in the layer
    layer_block.q_leakyrelu = pyo.Var(layer.output_indexes, within=pyo.Binary)

    # define two variables for each input of the layer block (=DenseLayer + relu)
    layer_block.x0 = pyo.Var(layer.input_indexes, layer.output_indexes, within=pyo.Reals)

    # TODO: Avoid creating zhat in the dense layer before extended formulation is called
    layer_block._z_eq_wxbq_relu = pyo.Constraint(layer.output_indexes)  # z == w*(x-x_0) + b*q, x = z previous layer
    layer_block._eq_wxbq_relu = pyo.Constraint(layer.output_indexes)  # 0 >= w*x_0 + b*(1-q)
    layer_block._x0_lb_relu = pyo.Constraint(layer.input_indexes, layer.output_indexes)  # [L(1-q) <= x_0] <= U*(1-q)
    layer_block._x0_ub_relu = pyo.Constraint(layer.input_indexes, layer.output_indexes)  # L(1-q) <= [x_0 <= U*(1-q)]
    layer_block._x1_lb_relu = pyo.Constraint(layer.input_indexes, layer.output_indexes)  # [L*q <= x - x_0] <= U*q
    layer_block._x1_ub_relu = pyo.Constraint(layer.input_indexes, layer.output_indexes)  # L*q <= [x - x_0 <= U*q]

    for output_index in layer.output_indexes:
        # compute the bounds for the input of the layer DenseLayer + relu (i.e., the output of the previous layer)
        for input_index in layer.input_indexes:
            lb, ub = previous_layer_block.z[input_index].bounds  # z of the previous layer are our x

            # Split the input variables and constraints them
            # L(1-q) <= x_0 <= U*(1-q)
            layer_block._x0_lb_relu[input_index, output_index] = ((lb * (1 - layer_block.q_leakyrelu[output_index])) <=
                                                                  layer_block.x0[input_index, output_index])
            layer_block._x0_ub_relu[input_index, output_index] = (layer_block.x0[input_index, output_index] <=
                                                                  (ub * (1 - layer_block.q_leakyrelu[output_index])))
            # L*q <= x - x_0 <= U*q
            layer_block._x1_lb_relu[input_index, output_index] = ((lb * layer_block.q_leakyrelu[output_index]) <=
                                                                  (previous_layer_block.z[input_index] -
                                                     layer_block.x0[input_index, output_index]))
            layer_block._x1_ub_relu[input_index, output_index] = (
                        (previous_layer_block.z[input_index] - layer_block.x0[input_index, output_index]) <=
                        (ub * layer_block.q_leakyrelu[output_index]))

            TMP0 = lb * (1 - layer_block.q_leakyrelu[output_index])
            # TMP01 = layer_block.x0[input_index]
            TMP1 = ub * (1 - layer_block.q_leakyrelu[output_index])
            TMP2 = lb * layer_block.q_leakyrelu[input_index]
            # TMP21 = previous_layer_block.z[input_index] - layer_block.x0[input_index]
            TMP3 = ub * layer_block.q_leakyrelu[output_index]
            # TMP40 = ((lb * (1 - layer_block.q_relu[input_index])) <=
            #          layer_block.x0[input_index])
            # TMP41 = (layer_block.x0[input_index] <=
            #          (ub * (1 - layer_block.q_relu[input_index])))
            # TMP42 = ((lb * layer_block.q_relu[input_index]) <=
            #          (previous_layer_block.z[input_index] - layer_block.x0[input_index]))
            # TMP43 = ((previous_layer_block.z[input_index] - layer_block.x0[input_index]) <=
            #          (ub * layer_block.q_relu[input_index]))


        # get the bounds of the input and propagate them to the output
        lb, ub = layer_block.zhat[output_index].bounds
        # z >= 0
        layer_block.z[output_index].setlb(0)
        layer_block.z[output_index].setub(max(0, ub))

        # compute the dense layer results
        def dense_layer_bq(x1, x2, q, layer, output_index):
            """
            Function to structure the code and avoid code duplication.
            X is the input variable of the dense layer. In our case, we split z from the input layer (i.e., the
            previous layer) into x0 and x-x0.
            Here, we add a binary condition to the biais.
            :return: w*Xi + b*q
            """
            # dense layers multiply only the last dimension of their inputs
            expr = 0.0
            for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
                w = layer.weights[local_index[-1], output_index[-1]]
                if x2 is None:
                    expr += x1[input_index, output_index] * w
                else:
                    TMP0 = x2[input_index, output_index]
                    expr += (x1[input_index] - x2[input_index, output_index]) * w
            # move this at the end to avoid numpy/pyomo var bug
            expr += layer.biases[output_index[-1]] * q

            return expr

        # z == w*(x-x_0) + b*q
        layer_block._z_eq_wxbq_relu[output_index] = layer_block.z[output_index] == dense_layer_bq(
            previous_layer_block.z, layer_block.x0, layer_block.q_leakyrelu[output_index], layer, output_index)
        TMP5 = dense_layer_bq(
            previous_layer_block.z, layer_block.x0, layer_block.q_leakyrelu[output_index], layer, output_index)
        # 0 >= w*x_0 + b*(1-q)
        layer_block._eq_wxbq_relu[output_index] = 0 >= dense_layer_bq(layer_block.x0, None,
                                                                      1 - layer_block.q_leakyrelu[output_index], layer,
                                                                      output_index)
        TMP6 = dense_layer_bq(layer_block.x0, None,
                              1 - layer_block.q_leakyrelu[output_index], layer, output_index)
        TMP7 = 0


class ComplementarityReLUActivation:
    r"""
    Complementarity-based ReLU activation forumlation.

    Generates the constraints for the ReLU activation function.

    .. math::

        \begin{align*}
        z_i &= \text{max}(0, \hat{z_i}) && \forall i \in N
        \end{align*}

    The complementarity-based formulation for the i-th node is given by:

    .. math::

        \begin{align*}
        0 &\leq z_i \perp (z-\hat{z_i}) \geq 0
        \end{align*}

    """

    def __init__(self, transform=None):
        if transform is None:
            transform = "mpec.simple_nonlinear"
        self.transform = transform

    def __call__(self, net_block, net, layer_block, layer):
        layer_block._complementarity = mpec.Complementarity(
            layer.output_indexes, rule=_relu_complementarity
        )
        xfrm = pyo.TransformationFactory(self.transform)
        xfrm.apply_to(layer_block)


def _relu_complementarity(b, *output_index):
    return mpec.complements(
        b.z[output_index] - b.zhat[output_index] >= 0, b.z[output_index] >= 0
    )
