#!/usr/bin/env python
"""TODO Module Docstring."""
import os

from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach

vc = VirtualCoach(environment='local', storage_username='nrpuser')
simulation = vc.launch_experiment("template_new_0")
print(simulation.get_state())

try:
    with open(os.path.dirname(os.path.realpath(__file__)) + "/transferfunctions/test_tf.py", "r") as f:
        tf1 = f.read()

    if simulation.get_state() in ["paused", "stopped"]:
        simulation.start()
        simulation.add_transfer_function(tf1)
        print("\nTransfer functions:\n__________________")
        simulation.print_transfer_functions()
except Exception as e:
    # always stop simulation in the end
    print(e)
    pass

print("\n\nNOW STOPPING SERVER\n----------------------")
simulation.stop()
