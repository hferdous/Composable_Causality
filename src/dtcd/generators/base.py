import numpy as np
from scipy import stats
from ..utils.missing_data import MissingDataHandler
from ..utils.seasonality import SeasonalityGenerator

class BaseTimeSeriesGenerator:
    def __init__(self, noise_type='gaussian', noise_scale=0.1, df=3, random_state=None):
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.df = df
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            if noise_type == 'student_t':
                stats.t.random_state = np.random.RandomState(random_state)
                
        self.missing_handler = MissingDataHandler(random_state)
        self.seasonality_generator = SeasonalityGenerator(random_state)

    def generate_noise(self, size):
        """Generate noise from specified distribution"""
        if self.noise_type == 'gaussian':
            return np.random.normal(0, self.noise_scale, size=size)
        elif self.noise_type == 'student_t':
            return stats.t.rvs(df=self.df, loc=0, scale=self.noise_scale, size=size)
        elif self.noise_type == 'laplace':
            return np.random.laplace(loc=0, scale=self.noise_scale, size=size)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
            
    def apply_missingness(self, df, mechanism='mcar', **params):
        """
        Apply missing data mechanism to the dataset.
        Returns (df_with_missing, mask)
        """
        # Exclude 'time' and 'U' columns from missingness usually
        cols_to_process = [c for c in df.columns if c.startswith('X')]
        
        df_subset = df[cols_to_process]
        df_missing_subset, mask = self.missing_handler.apply_missingness(df_subset, mechanism, **params)
        
        df_result = df.copy()
        df_result[cols_to_process] = df_missing_subset
        
        return df_result, mask
        
    def add_seasonality(self, df, seasonality_config):
        """
        Add seasonality to the dataset.
        seasonality_config: dict {col_name: {'periods': [12, 24], 'strengths': [0.5, 0.2]}}
        """
        return self.seasonality_generator.add_seasonality(df, seasonality_config)
