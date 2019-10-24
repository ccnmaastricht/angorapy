import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg
@nrp.Neuron2Robot(Topic("/contacts/rh_mf/distal", geometry_msgs.msg.Twist))
def testing_tf(t):
    #log the first timestep (20ms), each couple of seconds
    if t % 2 < 0.02:
        clientLogger.info('Time: ', t)
        return geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(1,2,3), angular=geometry_msgs.msg.Vector3(1,22,5))