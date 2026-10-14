import time
import numpy as np
import pybullet as p
import pybullet_data

WCMD_MIN, WCMD_MAX = -4.0, 4.0

class HuskyTrajectoryEnv:
    def __init__(
        self,
        gui=True,
        dt=1 / 120,
        pose_rate_hz=1,
        sigma_v=0.03,
        sigma_omega=np.deg2rad(0.5),
        sigma_x=0.1,
        sigma_y=0.1,
        sigma_theta=np.deg2rad(1.0),
        seed=None,
        consider_obstacles=False,
    ):

        self.dt = dt
        self.rng = np.random.default_rng(seed)

        if pose_rate_hz <= 0:
            raise ValueError("pose_rate_hz must be > 0")

        self.pose_rate_hz = float(pose_rate_hz)
        self.pose_period = 1.0 / self.pose_rate_hz

        self.sigma_v = float(sigma_v)
        self.sigma_omega = float(sigma_omega)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.sigma_theta = float(sigma_theta)
        self.consider_obstacles = consider_obstacles

        # These matrices can be passed directly to the EKF.
        self.R_vel = np.diag([
            self.sigma_v**2,
            self.sigma_omega**2,
        ])

        self.R_pose = np.diag([
            self.sigma_x**2,
            self.sigma_y**2,
            self.sigma_theta**2,
        ])

        self.sim_time = 0.0
        self.next_pose_time = 0.0
        self.last_measurements = None

        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)

        # Husky parameters
        self.FL = 2
        self.FR = 3
        self.RL = 4
        self.RR = 5
        self.wheel_joints = [
            self.FL,
            self.FR,
            self.RL,
            self.RR,
        ]

        # Approximate Husky dimensions
        self.wheel_radius = 0.17775
        self.track_width = 0.5708

        # World
        self.plane = p.loadURDF("plane.urdf")
        self.husky = p.loadURDF(
            "husky/husky.urdf",
            basePosition=[0, 0, 0.15],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )

        # Increase friction a little
        for joint in self.wheel_joints:
            p.changeDynamics(
                self.husky,
                joint,
                lateralFriction=1,
            )

        # Obstacles
        # Format: [x, y, half_size_x, half_size_y]
        if self.consider_obstacles:
            self.obstacles = [
                [3.0, 1.0, 0.6, 0.6],
                [5.5, -1.0, 0.7, 0.7],
                [8.0, 1.2, 0.5, 0.8],
                [10.0, -0.8, 0.8, 0.5],
            ]
        else:
            self.obstacles = []
        self.obstacle_ids = []

        for obstacle in self.obstacles:
            x, y, hx, hy = obstacle
            obstacle_id = self.create_box_obstacle(
                position=[x, y, 0.5],
                half_extents=[hx, hy, 0.5],
            )
            self.obstacle_ids.append(obstacle_id)

        # Reference trajectory
        self.reference = self.generate_reference()
        self.draw_reference()

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=12,
                cameraYaw=0,
                cameraPitch=-60,
                cameraTargetPosition=[5, 0, 0],
            )

        # Initial sensor packet: pose is available at t = 0.
        self._generate_measurements(force_pose=True)

        self.Kp_w, self.Ki_w = 1.5, 1.0
        self.integ_w = 0.0

    @staticmethod
    def wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))


    def create_box_obstacle(self, position, half_extents):
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1],
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        return obstacle_id


    def generate_reference(self):

        x = np.linspace(0, 12, 400)
        y = 2.0 * np.sin(0.45 * x)

        dx = np.gradient(x)
        dy = np.gradient(y)
        theta = np.arctan2(dy, dx)

        return np.column_stack([x, y, theta])

    def draw_reference(self):
        for k in range(len(self.reference) - 1):
            p1 = [
                self.reference[k, 0],
                self.reference[k, 1],
                0.05,
            ]
            p2 = [
                self.reference[k + 1, 0],
                self.reference[k + 1, 1],
                0.05,
            ]
            p.addUserDebugLine(
                p1,
                p2,
                lineColorRGB=[0, 0, 1],
                lineWidth=2,
            )

    # Low-level velocity command

    def command(self, v, omega):
        left_velocity = (
            v - omega * self.track_width / 2
        ) / self.wheel_radius

        right_velocity = (
            v + omega * self.track_width / 2
        ) / self.wheel_radius

        wheel_velocities = [
            left_velocity,
            right_velocity,
            left_velocity,
            right_velocity,
        ]

        p.setJointMotorControlArray(
            bodyUniqueId=self.husky,
            jointIndices=self.wheel_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=wheel_velocities,
            forces=[100, 100, 100, 100],
        )

    def get_state(self):
        """
        Return PyBullet ground truth:
            [x, y, theta, v, omega]

        This should be used for evaluation/plots, not as the EKF input.
        """
        position, quaternion = p.getBasePositionAndOrientation(self.husky)
        _, _, yaw = p.getEulerFromQuaternion(quaternion)

        linear_velocity, angular_velocity = p.getBaseVelocity(self.husky)

        # Forward velocity expressed in the robot body direction.
        v = (
            linear_velocity[0] * np.cos(yaw)
            + linear_velocity[1] * np.sin(yaw)
        )
        omega = angular_velocity[2]

        return np.array([
            position[0],
            position[1],
            yaw,
            v,
            omega,
        ], dtype=float)

    # Simulated sensors
    def _generate_measurements(self, force_pose=False):
        """
        Generate one sensor packet from the current PyBullet ground truth.

        Velocity measurement is generated every environment step.
        Pose measurement is generated only at pose_rate_hz (or when forced).
        """
        truth = self.get_state()
        x, y, theta, v, omega = truth

        # Encoder / IMU-like measurement, available every step.
        z_vel = np.array([
            v + self.rng.normal(0.0, self.sigma_v),
            omega + self.rng.normal(0.0, self.sigma_omega)])

        pose_available = force_pose or (
            self.sim_time + 1e-12 >= self.next_pose_time
        )
        z_pose = None

        if pose_available:
            theta_meas = self.wrap_angle(theta + self.rng.normal(0.0, self.sigma_theta))
            z_pose = np.array([x + self.rng.normal(0.0, self.sigma_x),y + self.rng.normal(0.0, self.sigma_y),
                theta_meas])

            # Schedule the next fiducial/pose observation.
            while self.next_pose_time <= self.sim_time + 1e-12:
                self.next_pose_time += self.pose_period

        self.last_measurements = {
            "velocity": z_vel,
            "pose": z_pose,
            "pose_available": pose_available,
            "timestamp": self.sim_time,
        }

        return self.last_measurements

    def get_measurements(self):
        return self.last_measurements



    def collision(self):
        for obstacle in self.obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=self.husky,
                bodyB=obstacle,
            )
            if len(contacts) > 0:
                return True
        return False


    def step(self, action, substeps=4):
        v = float(action[0])
        omega = float(action[1])
        omega_ref = float(action[1])

        # PI here
        omega_real = self.get_state()[-1]
        err_w = omega_ref - omega_real
        wcmd_unsat = omega_ref + self.Kp_w * err_w + self.Ki_w * self.integ_w
        wcmd = float(np.clip(wcmd_unsat, WCMD_MIN, WCMD_MAX))
        saturated_high = wcmd_unsat > WCMD_MAX
        saturated_low = wcmd_unsat < WCMD_MIN
        winding_up = (saturated_high and err_w > 0) or (saturated_low and err_w < 0)
        if not winding_up:
            self.integ_w += err_w * self.dt * substeps

        self.command(v, wcmd)


        for _ in range(substeps):
            p.stepSimulation()

        # Elapsed simulated time for this control step.
        self.sim_time += substeps * self.dt

        state = self.get_state()
        collision = self.collision()

        # Generate exactly one new sensor packet per environment step.
        self._generate_measurements()

        return state, collision

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.husky,
            [0, 0, 0.15],
            p.getQuaternionFromEuler([0, 0, 0]),
        )

        p.resetBaseVelocity(
            self.husky,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
        )

        self.sim_time = 0.0
        self.next_pose_time = 0.0

        state = self.get_state()

        # Assume an initial fiducial/localization fix is available.
        self._generate_measurements(force_pose=True)

        return state

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = HuskyTrajectoryEnv(
        gui=True,
        pose_rate_hz=5.0,
        seed=1,
    )

    env.reset()

    for k in range(600):
        truth, collision = env.step([0.8, 0.15])
        measurements = env.get_measurements()

        if measurements["pose_available"]:
            print(
                f't={measurements["timestamp"]:.2f}s | '
                f'z_vel={measurements["velocity"]} | '
                f'z_pose={measurements["pose"]}'
            )

        if collision:
            print("Collision!")
            break

        time.sleep(env.dt)

    env.close()
