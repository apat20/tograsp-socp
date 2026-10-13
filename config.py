import numpy as np

# Specifying gripper tolerances and 
# attributes to compute the location of the reference frame at the flange for the grasp pose and pre-grasp pose:
GRIPPER_WIDTH_TOLERANCE = 0.08
GRIPPER_HEIGHT_TOLERANCE = 0.041
# G_DELTA = 0.0624
# G_DELTA_INTER = 0.0925

G_DELTA = 0.1434
G_DELTA_INTER = 0.1800

# Friction params for SOCP:
F = 30

# Specifying the contact model to compute the Grasp Map. Here the contact model is the soft finger contact model
CONTACT_MODEL = 'sf'

# External force acting on the object. For this application we assume that the external force is the self weight of 
# the object.
# NOTE: Dimension of the array if 1x6 while initializing. It should be 6x1 when solving the optimization problem. 
F_EXTERNAL = np.reshape(np.asarray([0,0,-10,0,0,0]), [6,1])

# Variables specifying the dimensions of the optimization variables in the SOCP:
M = 6
N = 1

# Friction parameters for soft finger contact at the object-robot contact locations:
# Contact location 1:
E11 = 1
E12 = 1
E1R = 0.2
MU1 = 0.4

# Contact location 2:
E21 = 1
E22 = 1
E2R = 0.2
MU2 = 0.4

SIGMA = 0.6

# Friction coefficient for point contact with friction model assumed at the object-environment contact:
MU = 0.6
