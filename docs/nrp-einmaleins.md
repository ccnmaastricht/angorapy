# How to NRP

- docker vs. from source

### Controlling from Python
- gym wrapper style
- python 3

- rospy
- rostopics
- roskinetic (http://wiki.ros.org/rospy)

#### Ongoing Simulation
- constant input? or can simulation wait for timesteps
- sleep um zu warten bis action zuende
- schauen ob illegale werte ros joints ändern aber gelenke gleich bleiben
- nicht schneller als real time
- cube spawning mit ros (spawn_model = rospy.ServiceProxy("gazebo/spawn_model", SpawnModel))
- camera mit config files

#### Parallelization
 - multiple docker containers for multiple simulations?
