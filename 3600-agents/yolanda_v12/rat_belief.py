import numpy as np

NOISE_TABLE = {
    'blocked': {'squeak': 0.5, 'scratch': 0.3,  'squeal': 0.2},
    'space':   {'squeak': 0.7, 'scratch': 0.15, 'squeal': 0.15},
    'primed':  {'squeak': 0.1, 'scratch': 0.8,  'squeal': 0.1},
    'carpet':  {'squeak': 0.1, 'scratch': 0.1,  'squeal': 0.8},
}

DIST_OFFSET = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

class RatBelief:
    def __init__(self, T):
        self.T = np.array(T)
        self.Tt = self.T.T.copy()
        self.belief = np.zeros(64)

    def initialize(self):
        v = np.zeros(64)
        v[0] = 1.0  # rat placed at (0,0)
        for _ in range(1000):
            v = self.Tt @ v
        self.belief = v / v.sum()

    def predict(self):
        self.belief = self.Tt @ self.belief

    def _floor_type(self, i, board):
        bit = 1 << i
        if board._blocked_mask  & bit: return 'blocked'
        if board._carpet_mask & bit: return 'carpet'
        if board._primed_mask   & bit: return 'primed'
        return 'space'

    def update_noise(self, noise, board):
        noise_str = noise.name.lower() if hasattr(noise, 'name') else noise
        if isinstance(noise, int) and noise == 0: noise_str = 'squeak'
        elif isinstance(noise, int) and noise == 1: noise_str = 'scratch'
        elif isinstance(noise, int) and noise == 2: noise_str = 'squeal'
            
        likelihoods = np.array([
            NOISE_TABLE[self._floor_type(i, board)][noise_str]
            for i in range(64)
        ])
        self.belief *= likelihoods
        s = self.belief.sum()
        if s > 1e-12:
            self.belief /= s
        else:
            self.belief = np.ones(64) / 64.0

    def update_distance(self, reported_d, worker_pos):
        wx, wy = worker_pos
        likelihoods = np.zeros(64)
        for i in range(64):
            rx, ry = i % 8, i // 8
            true_d = abs(rx - wx) + abs(ry - wy)

            if reported_d == 0:
                p = 0.0
                for raw_offset, prob in DIST_OFFSET.items():
                    clipped = max(0, true_d + raw_offset)
                    if clipped == 0:
                        p += prob
                likelihoods[i] = p
            else:
                offset = reported_d - true_d
                likelihoods[i] = DIST_OFFSET.get(offset, 0.0)

        self.belief *= likelihoods
        s = self.belief.sum()
        if s > 1e-12:
            self.belief /= s
        else:
            self.belief = np.ones(64) / 64.0

    def update_opponent_search(self, search_result):
        loc, found = search_result
        if loc is None: return
        i = loc[1] * 8 + loc[0]
        if found:
            self.initialize()
        else:
            self.belief[i] = 0.0
            s = self.belief.sum()
            if s > 1e-12:
                self.belief /= s
            else:
                self.belief = np.ones(64) / 64.0

    def search_ev(self):
        p = self.belief.max()
        return 4.0 * p - 2.0 * (1.0 - p)

    def best_search_target(self):
        best_i = int(np.argmax(self.belief))
        return best_i % 8, best_i // 8

    def copy(self):
        rb = RatBelief.__new__(RatBelief)
        rb.T = self.T
        rb.Tt = self.Tt
        rb.belief = self.belief.copy()
        return rb
    
    def infer_movement_bias(self, old_pos, new_pos):
        ox, oy = old_pos
        nx, ny = new_pos
        dx, dy = nx - ox, ny - oy
        
        if dx == 0 and dy == 0: return # Didn't move
        
        gradient = np.ones(64)
        for i in range(64):
            x, y = i % 8, i // 8
            # If they moved right (dx=1), boost columns to the right of their old position
            if dx > 0 and x > ox: gradient[i] = 1.3
            elif dx < 0 and x < ox: gradient[i] = 1.3
            if dy > 0 and y > oy: gradient[i] = 1.3
            elif dy < 0 and y < oy: gradient[i] = 1.3

        self.belief *= gradient
        s = self.belief.sum()
        if s > 1e-12:
            self.belief /= s
        else:
            self.belief = np.ones(64) / 64.0
