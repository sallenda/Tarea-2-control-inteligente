import numpy as np


class EKF:
    def __init__(self, system, Qx, R_vel, R_pose, Px, x_est, Ts=0.001):
        self.system = system
        self.x = x_est
        self.Ts = Ts

        #### State Estimator ######
        self.Qx = Qx
        self.R_vel = R_vel
        self.R_pose = R_pose
        self.Px = Px
        self.x_est = x_est
        self.x_dim = len(x_est)

    def predict(self, u_k):
        u_k = np.array(u_k).reshape(-1, 1)

        ### 1) State prediction ###+
        x_pred = self.system.f(self.x_est, u=u_k)

        # Get jacobians
        A, B = self.system.evaluate_jacobians(self.x_est, u_k)
        Px_pred = A @ self.Px @ A.T + self.Qx

        self.x_est = x_pred
        self.Px = Px_pred


    def update(self, y_k, measurement_type='vel', angle_indices=[2]):
        y_k = np.array(y_k).reshape(-1, 1)

        if measurement_type == 'vel':
            Rx = self.R_vel
            C = np.array([[0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])
            innovation = (y_k - C @ self.x_est)
        else:
            Rx = self.R_pose
            C = np.array([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0]])
            innovation = (y_k - C @ self.x_est)
            for idx in angle_indices:
                innovation[idx, 0] = np.arctan2(
                    np.sin(innovation[idx, 0]),
                    np.cos(innovation[idx, 0]))

        S = Rx + C @ self.Px @ C.T
        Kx = np.linalg.solve(S.T, (self.Px @ C.T).T).T
        self.x_est = self.x_est + Kx @ innovation
        self.x_est[2, 0] = np.arctan2(np.sin(self.x_est[2, 0]), np.cos(self.x_est[2, 0]))

        I = np.eye(self.x_dim)
        self.Px = ((I - Kx @ C) @ self.Px @ (I - Kx @ C).T + Kx @ Rx @ Kx.T)  # Joseph form

    def get_state_and_cov(self):
        return self.x_est, self.Px