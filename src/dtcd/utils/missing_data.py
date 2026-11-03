import numpy as np
import pandas as pd

class MissingDataHandler:
    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def generate_mcar_mask(self, shape, prob):
        """
        Generate Missing Completely At Random (MCAR) mask.
        Returns boolean mask where True indicates missing.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        return np.random.random(shape) < prob

    def generate_block_missing_mask(self, n_points, n_vars, block_prob, min_size=5, max_size=20):
        """
        Generate Block Missingness mask.
        Simulates sensor failures or outages.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        missing_mask = np.zeros((n_points, n_vars), dtype=bool)
        current_block = False
        block_remaining = 0
        
        for t in range(n_points):
            if not current_block and np.random.random() < block_prob:
                # Start new missing block
                current_block = True
                block_remaining = np.random.randint(min_size, max_size + 1)
                
                # Choose which variables to affect (can be 1 to all)
                n_affected = np.random.randint(1, n_vars + 1)
                affected_vars = np.random.choice(n_vars, size=n_affected, replace=False)
                
                # Set missing pattern for this block
                end_idx = min(t + block_remaining, n_points)
                missing_mask[t:end_idx, affected_vars] = True
                
            if current_block:
                block_remaining -= 1
                if block_remaining <= 0:
                    current_block = False
                    
        return missing_mask

    def generate_mar_mask(self, data, observed_vars, prob_fn):
        """
        Generate MAR mask. Missingness depends on observed variables.
        prob_fn: function taking observed_values -> probability of missing
        """
        n_points, n_vars = data.shape
        mask = np.zeros(data.shape, dtype=bool)
        
        # For simplicity, let's assume we are masking a specific target variable
        # based on other fully observed variables.
        # If we want general MAR, it's iterative.
        
        # Simplified MAR:
        # P(X_i missing) = sigmoid(w * X_j + b)
        # We need to define weights.
        
        # Let's implement a logistic MAR for a target column based on a predictor column
        pass

    def apply_missingness(self, df, mechanism='mcar', **params):
        """
        Apply missingness to a DataFrame.
        mechanism: 'mcar', 'block', 'mixed', 'mar', 'mnar'
        params: 
            - prob (for MCAR)
            - block_prob, min_size, max_size (for Block)
            - mcar_prob (for Mixed)
            - p_missing (for MNAR/MAR logistic)
        """
        data = df.values
        n_points, n_vars = data.shape
        mask = np.zeros(data.shape, dtype=bool)
        
        if mechanism == 'mcar':
            prob = params.get('prob', 0.1)
            mask = self.generate_mcar_mask(data.shape, prob)
            
        elif mechanism == 'block':
            block_prob = params.get('block_prob', 0.05)
            min_size = params.get('min_size', 5)
            max_size = params.get('max_size', 20)
            mask = self.generate_block_missing_mask(n_points, n_vars, block_prob, min_size, max_size)
            
        elif mechanism == 'mixed':
            mcar_prob = params.get('mcar_prob', 0.05)
            block_prob = params.get('block_prob', 0.05)
            min_size = params.get('min_size', 5)
            max_size = params.get('max_size', 20)
            
            mcar_mask = self.generate_mcar_mask(data.shape, mcar_prob)
            block_mask = self.generate_block_missing_mask(n_points, n_vars, block_prob, min_size, max_size)
            mask = mcar_mask | block_mask
            
        elif mechanism == 'mnar':
            # Missing Not At Random: P(X missing) depends on X itself.
            # Logistic model: P(M=1 | X) = sigmoid(a*X + b)
            # We'll apply this to each variable independently or specified ones.
            # params: 'alpha' (slope), 'beta' (intercept)
            alpha = params.get('alpha', 1.0)
            beta = params.get('beta', 0.0)
            
            # Sigmoid
            probs = 1 / (1 + np.exp(-(alpha * data + beta)))
            
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            random_vals = np.random.random(data.shape)
            mask = random_vals < probs
            
        elif mechanism == 'mar':
            # Missing At Random: P(X_i missing) depends on X_j (j != i).
            # We'll implement a cyclic dependency for simplicity:
            # X1 missing depends on X2, X2 on X3, ..., Xn on X1.
            alpha = params.get('alpha', 1.0)
            beta = params.get('beta', 0.0)
            
            for i in range(n_vars):
                predictor_idx = (i + 1) % n_vars
                predictor = data[:, predictor_idx]
                probs = 1 / (1 + np.exp(-(alpha * predictor + beta)))
                
                if self.random_state is not None:
                    # We need different seed per column to avoid correlation artifacts?
                    # Or just rely on global state if set?
                    # Let's rely on global state advancement.
                    pass
                    
                random_vals = np.random.random(n_points)
                mask[:, i] = random_vals < probs
            
        else:
            raise ValueError(f"Unknown missing data mechanism: {mechanism}")
            
        df_missing = df.copy()
        df_missing.mask(mask, np.nan, inplace=True)
        
        return df_missing, mask
