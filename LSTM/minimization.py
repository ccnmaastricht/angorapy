import numpy as np




def evaluate_fixedpoint(jac_fun, fixedpoint):
    jacobian_matrix = jac_fun(fixedpoint.x)

    e_val, e_vec = np.linalg.eig(jacobian_matrix)

    is_stable = np.all(np.abs(e_val) < 1.0)
    scale = 0.75

    for i in range(len(e_val)):
        magnitude = np.abs(e_val[i])
        if not is_stable and magnitude > 1.0:
            x_plus = fixedpoint.x + scale*magnitude*e_vec[:, i]
            x_minus = fixedpoint.x - scale*magnitude*e_vec[:, i]

            x_mode = np.vstack((x_minus, fixedpoint.x, x_plus))

            return x_mode

        else:
            return is_stable

