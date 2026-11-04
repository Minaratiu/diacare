
import random
import numpy as np


class QLearningRouter:
    """
    Q-learning mô phỏng lựa chọn luồng xử lý cho hệ đa Agent.

    State:
        S1: Đánh giá nguy cơ
        S2: Tra cứu kiến thức
        S3: Kết hợp đánh giá và tra cứu

    Action:
        A1: Prediction Agent
        A2: Knowledge Agent
        A3: Cả hai Agent
    """

    STATES = ["S1", "S2", "S3"]
    ACTIONS = ["A1", "A2", "A3"]

    REWARDS = np.array([
        [10, -5, -2],
        [-5, 10, -2],
        [-5, -5, 10]
    ], dtype=float)

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        episodes=2000
    ):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes

        self.q_table = np.zeros((3, 3), dtype=float)
        self.train()

    def train(self):
        """Huấn luyện Q-table trong môi trường mô phỏng."""

        for _ in range(self.episodes):
            state = random.randrange(3)

            if random.random() < self.epsilon:
                action = random.randrange(3)
            else:
                action = int(np.argmax(self.q_table[state]))

            reward = self.REWARDS[state, action]

            # Môi trường một bước: trạng thái kết thúc sau hành động
            target = reward

            self.q_table[state, action] += (
                self.alpha * (target - self.q_table[state, action])
            )

    def choose_action(self, state):
        """Chọn action tốt nhất đã học."""

        if state not in self.STATES:
            raise ValueError(f"State khong hop le: {state}")

        state_index = self.STATES.index(state)
        action_index = int(np.argmax(self.q_table[state_index]))

        return self.ACTIONS[action_index]

    def get_q_table(self):
        return self.q_table.copy()
