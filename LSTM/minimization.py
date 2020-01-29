import numpy as np




def evaluate_fixedpoint(fixedpoint):
    x_modes = []

    for n in range(len(fixedpoint)):
        e_val, e_vec = np.linalg.eig(fixedpoint[n].jac)

        #is_stable = np.all(np.abs(e_val) < 1.0)

        det = np.linalg.det(fixedpoint[n].jac)
        magnitude = np.argmax(np.abs(e_val))
        for i in range(len(e_val)):
            #magnitude = np.abs(e_val[i])
            #if not is_stable and magnitude > 1.0:
             #   x_plus = fixedpoint[n].x + magnitude * e_vec[:, i]
             #   x_minus = fixedpoint[n].x - magnitude * e_vec[:, i]

             #   x_mode = np.vstack((x_minus, fixedpoint[n].x, x_plus))


            if det < 0 :
                x_plus = fixedpoint[n].x + e_val[magnitude[i]]*e_vec[:, magnitude[i]]
                x_minus = fixedpoint[n].x - e_val[magnitude[i]]*e_vec[:, magnitude[i]]

                x_mode = np.vstack((x_minus, fixedpoint[n].x, x_plus))
                x_modes.append(np.real(x_mode))
    return x_modes

def classify_fixedpoint(fixedpoint):
    # scale = 4
    for n in range(len(fixedpoint)):
        e_val, e_vecs = np.linalg.eig(fixedpoint[n].jac)
        trace = np.matrix.trace(fixedpoint[n].jac)
        det = np.linalg.det(fixedpoint[n].jac)

        #is_complex = isinstance(e_val, complex)
        if det < 0:
            print('saddle_point')
            x_modes.append(True)
            #ids = np.argwhere(np.abs(np.real(e_val)) > 1.0)
            #for i in range(len(ids)):
             #   x_plus = fixedpoint[n].x + scale*e_val[ids[i]]*e_vecs[:, ids[i]]
             #   x_minus = fixedpoint[n].x + scale*e_val[ids[i]]*e_vecs[:, ids[i]]
             #   x_mode = np.vstack((x_minus, fixedpoint[n].x, x_plus))
             #   x_modes.append(np.real(x_mode))
        elif det > 0:
            if trace**2-4*det > 0 and trace < 0:
                # print('node was found.')
                print('stable fixed point was found.')
                x_modes.append(False)
            elif trace**2-4*det > 0 and trace > 0:
                    print('unstable fixed point was found')
            else :
                    print('center was found.')
        else:
            print('fixed point manifold was found.')

    return x_modes



