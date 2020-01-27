import numpy as np




def evaluate_fixedpoint(fixedpoint):
    scale = 1.2
    x_modes = []

    for n in range(len(fixedpoint)):
        e_val, e_vec = np.linalg.eig(fixedpoint[n].jac)

        is_stable = np.all(np.abs(e_val) < 1.0)

        det = np.linalg.det(fixedpoint[n].jac)
        for i in range(len(e_val)):
            magnitude = np.abs(e_val[i])
            if det < 0 and magnitude > 1.0:
                x_plus = fixedpoint[n].x + scale*magnitude*e_vec[:, i]
                x_minus = fixedpoint[n].x - scale*magnitude*e_vec[:, i]

                x_mode = np.vstack((x_minus, fixedpoint[n].x, x_plus))
                x_modes.append(np.real(x_mode))
    return x_modes

