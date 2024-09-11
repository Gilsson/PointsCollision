import os
import shutil
import cv2
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class _MyGymPointsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    max_size = 10

    def __init__(
        self,
        a=10,
        b=10,
        max_size=10,
        verbose=False,
        draw_image=True,
        image_path="",
        changed_velocity_epsilon=0.01,
        changed_angle_epsilon=0.01,
        max_radius=1,
        draw_rectangle=True,
        seed=42,
    ):
        super().__init__()
        self.seed = seed
        self.max_radius = max_radius
        self.draw_rectangle = draw_rectangle
        self.velocity_epsilon = changed_velocity_epsilon
        self.angle_epsilon = changed_angle_epsilon
        self.draw_image = draw_image
        self.verbose = verbose
        self.passed_max_size = max_size
        self.max_size = np.random.randint(1, max_size)
        self.size = (a, b)
        self.step_num = 0
        self.points = self._generate_points()
        self.image_path = image_path
        self.observation_space = spaces.Box(-1.0, 1.0, (self.max_size, 4))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        reward = 0.0
        if self.verbose:
            print(action)
        target = int(action[0] * (self.max_size - 1) / 2)
        change_velocity = action[1] * max(self.size)
        change_angle = action[2] * np.pi
        if self.verbose:
            print(
                f"target: {target}",
                f"change_velocity: {change_velocity}",
                f"change_angle: {change_angle}",
            )
        self.points[target][2] += change_velocity
        self.points[target][2] = np.clip(
            self.points[target][2], -max(self.size), max(self.size)
        )
        self.points[target][3] += change_angle
        self.points[target][3] = np.clip(self.points[target][3], -np.pi, np.pi)

        # observation, reward, terminated, False, info
        self._move_forward()

        terminated = False
        if not (self._check_collisions()):
            reward += 2.0
            if (
                np.abs(change_velocity) < self.angle_epsilon
                and np.abs(change_angle) < self.velocity_epsilon
            ):
                reward += 2.0
        else:
            if self.verbose:
                print("collision!")
            terminated = True
            reward -= 5.0
        if self.verbose:
            print(f"terminated: {terminated}")
            print(f"reward: {reward}")
        if self.draw_image:
            self._save_image(f"{self.step_num}_after_step")
        # self.step_num += 1
        return self._get_obs(), reward, terminated, False, {}

    def _recoil_from_bounds(self):
        # Recoil points if they are about to go out of image bounds
        margin = 1  # Margin to determine when points should recoil

        # Check x-coordinate bounds
        mask_x_lower = self.points[:, 0] < margin
        mask_x_upper = self.points[:, 0] > (self.size[0] - margin)
        self.points[mask_x_lower, 3] += np.pi
        self.points[mask_x_upper, 3] += np.pi

        # Check y-coordinate bounds
        mask_y_lower = self.points[:, 1] < margin
        mask_y_upper = self.points[:, 1] > (self.size[1] - margin)
        self.points[mask_y_lower, 3] += np.pi
        self.points[mask_y_upper, 3] -= np.pi
        self.points[:, 3] = (self.points[:, 3] + np.pi) % (2 * np.pi) - np.pi

    def _move_forward(self, dt=1):
        # Update the positions of points based on their velocity and angle
        # Using Euler integration scheme
        self.points[:, 3] = (self.points[:, 3] + np.pi) % (2 * np.pi) - np.pi
        self.points[:, 0] += (
            self.points[:, 2] * np.cos(self.points[:, 3]) * dt
        )  # Update x positions
        self.points[:, 1] += (
            self.points[:, 2] * np.sin(self.points[:, 3]) * dt
        )  # Update y positions
        self.points[:, 0] = np.round(self.points[:, 0])
        self.points[:, 1] = np.round(self.points[:, 1])

        # Ensure that the updated coordinates are within the bounds of the environment
        self.points[:, 0] = np.clip(self.points[:, 0], 0, self.size[0] - 1)
        self.points[:, 1] = np.clip(self.points[:, 1], 0, self.size[1] - 1)
        # self._recoil_from_bounds()

    def _check_collision(self, point, neighbor):
        # Extract coordinates and radii of the points
        x1, y1, r1 = point[0], point[1], point[4]
        x2, y2, r2 = neighbor[0], neighbor[1], neighbor[4]

        max_x = max(x1, x2)
        if max_x == x1:
            max_y = y1
            max_r = r1
            min_x = x2
            min_y = y2
            min_r = r2
        else:
            max_y = y2
            max_r = r2
            min_x = x1
            min_y = y1
            min_r = r1

        # print(max_x, max_y, max_r, min_x, min_y, min_r)
        if min_x + min_r + 1 >= max_x and (
            min_y - min_r - 1 <= max_y and min_y + max_r + 1 >= max_y
        ):
            # print(min_x, min_y, min_r, max_x, max_y, max_r)
            # print(self.step_num)
            return True
        # if min_x + min_r + 1 >= max_x and min_y + min_r + 1 <= max_y:
        # Calculate the distance between the centers of the points
        # distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate the sum of their radii
        # sum_radii = r1 + r2 + 1
        # if abs(x1 - x2) <= r2 + 1 and abs(y1 - y2) <= r2 + 1:
        #     return True  # New point is too close to an existing point
        # if (x1 + r1 + 1) <= x2 or (y1 - r1 - 1) <= y2:
        # # Check if the distance is less than or equal to the sum of their radii
        # if distance <= sum_radii:
        #     return True  # Collision occurred
        # elif distance <= sum_radii + 1 and (abs(x1 - x2) <= 1 or abs(y1 - y2) <= 1):
        #     return True
        else:
            return False  # No collision

    def _check_collisions(self):
        # Set a threshold for proximity to consider a collision

        # Get the x and y coordinates of the points
        x_coords = self.points[:, 0].astype(int)
        y_coords = self.points[:, 1].astype(int)

        # Create an empty collision matrix
        self.collisions = np.zeros((len(self.points),), dtype=bool)

        # Iterate over each point
        for i in range(len(self.points)):
            point = self.points[i]
            x, y = x_coords[i], y_coords[i]
            radius = int(point[4])

            # Define the square region around the point
            x_min = max(0, x - self.max_radius - radius - 1)
            x_max = min(self.size[0], x + radius + self.max_radius + 1)
            y_min = max(0, y - self.max_radius - radius - 1)
            y_max = min(self.size[1], y + radius + self.max_radius + 1)

            # Extract the neighboring pixels within the square region
            neighbors = self.points[
                (x_coords >= x_min)
                & (x_coords < x_max)
                & (y_coords >= y_min)
                & (y_coords < y_max)
            ]

            # Exclude the current point from neighbors
            neighbors = neighbors[
                ~np.all(neighbors[:, :2] == self.points[i, :2], axis=1)
            ]
            # print(self.max_radius, radius)
            # print(x_min, x_max, y_min, y_max)
            # print(x, y, neighbors)

            # Check for collisions within the square region
            for neighbor in neighbors:
                if self._check_collision(point, neighbor):
                    self.collisions[i] = True
            if np.any(self.collisions):
                return True
        # Return False if no neighbors are found
        return False

    def _get_obs(self):
        return np.array(
            [
                self.points[:, 0] / (self.size[0]),
                self.points[:, 1] / (self.size[1]),
                self.points[:, 2] / (max(self.size)),
                self.points[:, 3] / (np.pi),
            ],
            dtype=np.float32,
        ).T

    def _generate_points(self):

        points = []
        x_values = []
        y_values = []
        velocity_values = []
        angle_values = []
        radius_values = []
        for i in range(self.max_size):
            while True:
                x_value = np.random.randint(0, self.size[0])
                y_value = np.random.randint(0, self.size[1])
                velocity_value = np.random.uniform(0, 5)
                angle_value = np.random.uniform(-np.pi, np.pi)
                radius_value = np.random.randint(1, self.max_radius)
                point = [x_value, y_value, velocity_value, angle_value, radius_value]
                if not self._is_point_too_close(point, points):
                    points.append(point)
                    x_values.append(x_value)
                    y_values.append(y_value)
                    velocity_values.append(velocity_value)
                    angle_values.append(angle_value)
                    radius_values.append(radius_value)
                    break

        # Combine the values into self.points
        return np.column_stack(
            (x_values, y_values, velocity_values, angle_values, radius_values)
        )

    def _is_point_too_close(self, new_point, points):
        # Check if the new point is too close to any existing points
        for point in points:
            x1, y1, r1 = point[0], point[1], point[4]
            x2, y2, r2 = new_point[0], new_point[1], new_point[4]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            max_r = max(r1, r2) + 1
            if r1 == max_r:
                max_x = x1
                max_y = y1
                min_x = x2
                min_y = y2
            else:
                max_x = x2
                max_y = y2
                min_x = x1
                min_y = y1
            if abs(max_x - min_x) <= max_r - 1 and abs(max_y - min_y) <= max_r - 1:
                return True  # New point is too close to an existing point
        return False  # New point is not too close to any existing point

    def _save_image(self, filename):
        # Create an empty RGB image
        image = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        # Set all pixels to black initially
        image[:, :] = [0, 0, 0]

        # Normalize velocity to the range [0, 255]
        normalized_velocity = (self.points[:, 2] / np.max(self.points[:, 2])) * 255

        # Convert angle to the range [0, 255]
        normalized_angle = ((self.points[:, 3] + np.pi) / (2 * np.pi)) * 255

        # Round coordinates to integers for indexing
        x_coords = np.round(self.points[:, 0]).astype(int)
        y_coords = np.round(self.points[:, 1]).astype(int)

        radii = self.points[:, 4].astype(int)
        for x, y, radius, velocity, angle in zip(
            x_coords, y_coords, radii, normalized_velocity, normalized_angle
        ):
            # Draw a filled circle for each point
            cv2.rectangle(
                image, (x, y), (x + radius, y - radius), (255, angle, velocity), -1
            )

        if self.draw_rectangle:
            for i in range(len(self.points)):
                if self.collisions[i]:
                    x, y, radius = int(x_coords[i]), int(y_coords[i]), int(radii[i])
                    x_min, x_max = max(0, x - radius - self.max_radius), min(
                        self.size[0], x + radius + self.max_radius + 1
                    )
                    y_min, y_max = max(0, y - self.max_radius - radius), min(
                        self.size[1], y + self.max_radius + radius + 1
                    )
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

        image_path = os.path.join(self.image_path, filename + "_1" + ".png")
        if self.verbose:
            print(image_path)

        if os.path.exists(image_path):
            os.remove(image_path)
        cv2.imwrite(image_path, image)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.max_size = np.random.randint(1, self.passed_max_size)
        self.points = self._generate_points()
        observation = self._get_obs()
        collision = self._check_collisions()
        self._save_image(f"{self.step_num}")
        return np.array(observation, dtype=np.float32), {"collision": collision}


def generate_dataset(
    size=10000,
    image_shape=(128, 128),
    n_points=14,
    points_max_radius=4,
    steps=1,
    name="test",
    path=None,
    remove_existing_folder=False,
    seed=42,
):
    """
    Generates a dataset of images with points and saves them to a specified path.

    Parameters:
        size (int): The number of images to generate. Default is 10000.
        image_shape (tuple): The shape of each image. Default is (128, 128).
        n_points (int): The number of points to generate in environment. Default is 14.
        points_max_radius (int): The maximum radius of each point. Default is 4.
        steps (int): The number of steps to take in the environment. Default is 1.
        name (str): The name of the dataset. Default is "test".
        path (str): The path to save the dataset. Default is None. If None, the dataset is saved in the current dataset folder.
        remove_existing_folder (bool): Whether to remove the existing folder at the specified path. Default is True. Attention: it will remove all the data in the folder.
        seed (int): The seed value for random number generation. Default is 42.

    Returns:
        None

    Example of usage:
        generate_dataset(size=200000, image_shape=(32, 32), n_points=15, points_max_radius=4, name="32_15_4_train", path=".\\my_datasets", remove_existing_folder=True, seed=42)
        transform = transforms.Compose(
            [
                transforms.RandomRotation(35),
                transforms.ToTensor(),
                transforms.RandomInvert(),
            ]
        )
        train_dataset = CustomDataset(root_dir=".\\my_datasets\\32_15_4_train", transform=transform)
        data_loader = DataLoader(train_dataset, batch_size=batch_size)
    """

    np.random.seed(seed)
    if not path:
        path = ".\\datasets"
    path = os.path.join(path, name)
    if os.path.exists(path):
        if remove_existing_folder:
            shutil.rmtree(path)
    os.mkdir(path)
    env = _MyGymPointsEnv(
        image_shape[0],
        image_shape[1],
        max_size=n_points,
        image_path=path,
        draw_image=False,
        verbose=False,
        max_radius=points_max_radius,
        draw_rectangle=False,
        seed=seed,
    )
    done = False
    zero_counter = 0
    for i in tqdm(range(size)):
        env.step_num = i
        _, ended = env.reset(seed=seed)
        dones = []
        for _ in range(steps):
            _, _, done, _, _ = env.step([0, 0, 0])
            dones.append(done)
        if ended["collision"] == True or any(dones):
            image_path = os.path.join(path, str(i) + "_1" + ".png")
            os.rename(image_path, os.path.join(env.image_path, str(i) + "_0" + ".png"))
            zero_counter += 1

    print(f"Zero classes: {zero_counter}/{size}")


def generate_dataset_multiclass(
    size=10000,
    min_image_shape=(20, 20),
    max_image_shape=(128, 128),
    n_points=14,
    points_max_radius=4,
    steps=10,
    name="test",
    path=None,
    remove_existing_folder=False,
    seed=42,
):
    """
    Generates a dataset of images with points and saves them to a specified path.

    Parameters:
        size (int): The number of images to generate. Default is 10000.
        image_shape (tuple): The shape of each image. Default is (128, 128).
        n_points (int): The number of points to generate in environment. Default is 14.
        points_max_radius (int): The maximum radius of each point. Default is 4.
        steps (int): The number of steps to take in the environment. Default is 1.
        name (str): The name of the dataset. Default is "test".
        path (str): The path to save the dataset. Default is None. If None, the dataset is saved in the current dataset folder.
        remove_existing_folder (bool): Whether to remove the existing folder at the specified path. Default is True. Attention: it will remove all the data in the folder.
        seed (int): The seed value for random number generation. Default is 42.

    Returns:
        None

    Example of usage:
        generate_dataset(size=200000, image_shape=(32, 32), n_points=15, points_max_radius=4, name="32_15_4_train", path=".\\my_datasets", remove_existing_folder=True, seed=42)
        transform = transforms.Compose(
            [
                transforms.RandomRotation(35),
                transforms.ToTensor(),
                transforms.RandomInvert(),
            ]
        )
        train_dataset = CustomDataset(root_dir=".\\my_datasets\\32_15_4_train", transform=transform)
        data_loader = DataLoader(train_dataset, batch_size=batch_size)
    """

    np.random.seed(seed)
    cap = np.ceil(size / (steps + 2))
    results = {str(i): 0 for i in range(steps + 2)}
    if not path:
        path = ".\\datasets"
    path = os.path.join(path, name)
    if os.path.exists(path):
        if remove_existing_folder:
            shutil.rmtree(path)
    os.mkdir(path)

    done = False
    sum = 0
    while sum < size:
        env = _MyGymPointsEnv(
            np.random.randint(min_image_shape[0], max_image_shape[0] - 1),
            np.random.randint(min_image_shape[0], max_image_shape[1] - 1),
            max_size=n_points,
            image_path=path,
            draw_image=False,
            verbose=False,
            max_radius=points_max_radius,
            draw_rectangle=False,
            seed=seed,
        )
        env.step_num = sum
        _, ended = env.reset(seed=seed)
        image_path = os.path.join(path, str(sum) + "_1" + ".png")
        if ended["collision"] == True:
            if results[str(0)] > cap:
                os.remove(image_path)
                continue
            results["0"] += 1

            os.rename(
                image_path, os.path.join(env.image_path, str(sum) + "_0" + ".png")
            )
            sum += 1
            continue
        elif steps == 0:
            if results["1"] < cap:
                results["1"] += 1
                sum += 1
            else:
                image_path = os.path.join(path, str(sum) + "_1" + ".png")
                os.remove(image_path)
                continue
        for j in range(steps):
            _, _, done, _, _ = env.step([0, 0, 0])
            if done:
                if results[str(j + 2)] > cap:
                    os.remove(image_path)
                    break
                os.rename(
                    image_path,
                    os.path.join(
                        env.image_path, str(sum) + "_{}".format(j + 2) + ".png"
                    ),
                )
                results[str(j + 2)] += 1
                sum += 1
                break
            else:
                if j == steps - 1:
                    if results["1"] < cap:
                        results["1"] += 1
                        sum += 1
                    else:
                        image_path = os.path.join(path, str(sum) + "_1" + ".png")
                        os.remove(image_path)
                        continue
        # for k in range(2):
        #     env.step_num = i
        #     _, ended = env.reset(seed=seed)
        #     if ended["collision"] == True:
        #         image_path = os.path.join(path, str(i) + "_1" + ".png")
        #         if k == 1:
        #             os.rename(
        #                 image_path, os.path.join(env.image_path, str(i) + "_0" + ".png")
        #             )
        #             results["0"] += 1
        #             break
        #         else:
        #             os.remove(image_path)
        #             continue
        #     done = False
        #     for j in range(steps):
        #         _, _, done, _, _ = env.step([0, 0, 0])
        #         if done:
        #             image_path = os.path.join(path, str(i) + "_1" + ".png")
        #             os.rename(
        #                 image_path,
        #                 os.path.join(
        #                     env.image_path, str(i) + "_{}".format(j + 2) + ".png"
        #                 ),
        #             )
        #             results[str(j + 2)] += 1
        #             break
        #     if done is True:
        #         break
        #     else:
        #         image_path = os.path.join(path, str(i) + "_1" + ".png")
        #         os.remove(image_path)
        #         continue
    # results["1"] = size - sum(results.values())
    print(f"Results: {results}")
