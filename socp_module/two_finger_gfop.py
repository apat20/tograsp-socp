# Code for computing the task dependent grasp metric for tasks which don't
# have contact with the environment and the objective is to apply a pure 
# force about an axis. Example: Picking up an object off the ground
# By: Aditya Patankar

import cvxpy as cp
import numpy as np

# Function to compute the skew symmetric matrix of a vector:
def get_skew_symmetric(p):
    if p.shape == (3,1):
        p_hat = np.asarray([[0, float(-p[2]), float(p[1])],
                            [float(p[2]), 0,  float(-p[0])],
                            [float(-p[1]), float(p[0]), 0]])
    else:
        print('Invalid Dimensions')
    return p_hat

# Function to compute the Grasp Map:
def get_grasp_map(R, p_hat, contact_model):
    # Soft-Finger with Elliptic Approximation:
    if contact_model == 'sf':
        I = np.zeros([3,5])
        I[0:3, 0:3] = np.eye(3)
        Z = np.zeros([3,5])
        z_small = np.zeros([3])
        e_z = np.transpose(np.asarray([0,0,1]))
        B_c = np.zeros([6,6])
        B_c[0:3, 0:5] = I
        B_c[0:3, 5] = z_small
        B_c[3:6, 0:5] = Z
        B_c[3:6, 5] = e_z

        adjoint = np.zeros([6,6])
        adjoint[0:3, 0:3] = R
        adjoint[0:3, 3:6] = np.zeros([3,3])
        adjoint[3:6, 0:3] = np.matmul(p_hat,R)
        adjoint[3:6, 3:6] = R

        G = np.matmul(adjoint, B_c)
    # Point Contact with Friction:
    elif contact_model == 'pf':
        B_c = np.zeros([5,3])
        B_c[0:3, 0:3] = np.eye(3)

        adjoint = np.zeros([6,6])
        adjoint[0:3, 0:3] = R
        adjoint[0:3, 3:6] = np.zeros([3,3])
        adjoint[3:6, 0:3] = np.matmul(p_hat,R)
        adjoint[3:6, 3:6] = R
        '''Dimensionality mismatch. Need to review the original formula for point contact with friction'''
        G = np.matmul(adjoint, B_c)
    else:
        print('Invalid Contact Model')
    return G

# Function to compute the Adjoint of a Matrix:
def get_adjoint_wrench(g):
    R = g[0:3, 0:3]
    p = g[0:3, 3]
    p_hat = get_skew_symmetric(np.reshape(p, [3,1]))
    adjoint = np.zeros([6,6])
    adjoint[0:3, 0:3] = R
    adjoint[0:3, 3:6] = np.zeros([3,3])
    adjoint[3:6, 0:3] = np.matmul(p_hat,R)
    adjoint[3:6, 3:6] = R

    return adjoint

# Function to compute the Adjoint of a Matrix using CVX's matrix concatenation function bmat
def get_adjoint_wrench_cvx(g):
    R = g[0:3, 0:3]
    p = g[0:3, 3]
    p_hat = get_skew_symmetric(np.reshape(p, [3,1]))
    adjoint = cp.bmat([[R, np.zeros([3,3])], [cp.matmul(p_hat,R), R]])

    return adjoint

class two_finger_gfop(object):

    def __init__(self):

        # Initial Position and Orientation of the Object Reference Frame {O}:
        self.R_O_initial = None
        self.p_O_initial = None

        # Final Position and Orientation of the Object Reference Frame {O}:
        self.R_O_final = None
        self.p_O_final = None

        # Object-Robot contact reference frames {C1} and {C2} expressed with respect to the object reference frame {O}:
        self.R_OC_1 = None
        self.p_OC_1 = None
        self.normal_C_1 = None

        # Skew Symmetric form of the position vectors:
        self.p_OC_1_hat = None
        self.p_OC_2_hat = None

        self.R_OC_2 = None
        self.p_OC_2 = None
        self.normal_C_2 = None

        # External wrench, in this case the self weight of the object, acting at and expressed in the object reference frame {O}:
        self.F_external = None

        # Maximum force in Newtons that can be exerted in the normal direction at the contact locations {C1} and {C2}: 
        self.F = None

        # Contact Model: 
        # Data type: str
        # Values: 
        # 'sf' :  Soft Finger contact model. 
        self.contact_model = None

        # Contact Model Parameters:
        # Friction parameters for soft finger contact at the object-robot contact locations:
        # Contact location 1:
        self.e11 = None
        self.e12 = None
        self.e1r = None
        self.mu1 = None

        # Contact location 2:
        self.e21 = None
        self.e22 = None
        self.e2r = None
        self.mu2 = None

        self.sigma  = None

        # Friction coefficient for point contact with friction model assumed at the object-environment contact:
        self.mu = None

        # Variables Specifying the dimensions of the optimization parameters in the SOCP:
        self.m = None
        self.n = None

        # Screw Axis:
        self.unit_vector = None
        self.point = None

        # Data structure to store the computed task-dependent grasp metric values:
        self.metric_values = None


    def compute_metric_force(self):
        # Computing the skew symmetric matrices and the corresponding Grasp Map:
        self.p_OC_1_hat = get_skew_symmetric(self.p_OC_1)
        self.p_OC_2_hat = get_skew_symmetric(self.p_OC_2)
        G1 = get_grasp_map(self.R_OC_1, self.p_OC_1_hat, self.contact_model)
        G2 = get_grasp_map(self.R_OC_2, self.p_OC_2_hat, self.contact_model)

        # Using CVXPY Atoms:
        G = cp.hstack([G1, G2])

        '''Defining and solving the Grasping Force Optimization Problem as a SOCP: '''
        fC = cp.Variable(shape = (12, 1))
        tau = cp.Variable(self.n)

        # Extracting the first two components of the object-robot contact forces
        fc1 = fC[0:2]
        fc2 = fC[6:8]

        # Task-Dependent Grasp Metric is Force along the screw axis:
        vect = cp.vstack([self.unit_vector, np.zeros([3,1])])

        # Specifying the second order cone constraints corresponding to the soft finger contact model:
        soc_constraints = [
            # Soft finger contact constraints at the object-robot contacts:
            cp.SOC(self.mu1*fC[2] , cp.hstack([fc1[0]/self.e11, fc1[1]/self.e12, fC[5]/self.e1r])),
            cp.SOC(self.mu2*fC[8] , cp.hstack([fc2[0]/self.e21, fc2[1]/self.e22, fC[11]/self.e2r])),
            cp.SOC(self.sigma*fC[2], fC[5]),
            cp.SOC(self.sigma*fC[8], fC[11]),
        ]

        prob = cp.Problem(cp.Maximize(tau),
                                        # Equilibrium constraint
                    soc_constraints + [G@fC + self.F_external == tau*vect, 
                                    # Non-zero contact force along the normal direction
                                    fC[2] >= 0, fC[8] >= 0,
                                    fC[3] == 0, fC[9] == 0, 
                                    fC[4] == 0, fC[10] == 0,
                                    fC[2] <= self.F, fC[8] <= self.F]
                                    )

        prob.solve()
        return prob.value
    
    def compute_metric_moment(self):
        # Computing the skew symmetric matrices and the corresponding Grasp Map:
        self.p_OC_1_hat = get_skew_symmetric(self.p_OC_1)
        self.p_OC_2_hat = get_skew_symmetric(self.p_OC_2)
        G1 = get_grasp_map(self.R_OC_1, self.p_OC_1_hat, self.contact_model)
        G2 = get_grasp_map(self.R_OC_2, self.p_OC_2_hat, self.contact_model)

        # Using CVXPY Atoms:
        G = cp.hstack([G1, G2])

        '''Defining and solving the Grasping Force Optimization Problem as a SOCP: '''
        fC = cp.Variable(shape = (12, 1))
        tau = cp.Variable(self.n)

        # Extracting the first two components of the object-robot contact forces
        fc1 = fC[0:2]
        fc2 = fC[6:8]

        # Task-Dependent Grasp Metric is Moment along the screw axis:
        vect = cp.vstack([np.zeros([3,1]), self.unit_vector])

        # Specifying the second order cone constraints corresponding to the soft finger contact model:
        soc_constraints = [
            # Soft finger contact constraints at the object-robot contacts:
            cp.SOC(self.mu1*fC[2] , cp.hstack([fc1[0]/self.e11, fc1[1]/self.e12, fC[5]/self.e1r])),
            cp.SOC(self.mu2*fC[8] , cp.hstack([fc2[0]/self.e21, fc2[1]/self.e22, fC[11]/self.e2r])),
            cp.SOC(self.sigma*fC[2], fC[5]),
            cp.SOC(self.sigma*fC[8], fC[11]),
        ]

        prob = cp.Problem(cp.Maximize(tau),
                                        # Equilibrium constraint
                    soc_constraints + [G@fC + self.F_external == tau*vect, 
                                    # Non-zero contact force along the normal direction
                                    fC[2] >= 0, fC[8] >= 0,
                                    fC[3] == 0, fC[9] == 0, 
                                    fC[4] == 0, fC[10] == 0,
                                    fC[2] <= self.F, fC[8] <= self.F]
                                    )

        prob.solve()
        return prob.value