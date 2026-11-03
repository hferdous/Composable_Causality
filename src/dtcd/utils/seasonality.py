import numpy as np

class SeasonalityGenerator:
    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def generate_seasonality(self, n_points, periods, strengths, phases=None, combine_sin_cos=True, timestamps=None):
        """
        Generate seasonality component.
        periods: list of periods (e.g. [12, 24])
        strengths: list of strengths corresponding to periods
        phases: list of phase shifts in radians. If 'random', generates random phase shifts. If None, uses 0.
        combine_sin_cos: If True, uses both sine and cosine components for a full Fourier series representation.
        timestamps: optional array of timestamps. If None, uses 0..n_points-1
        """
        if timestamps is None:
            t = np.arange(n_points)
        else:
            t = timestamps
            
        seasonality = np.zeros(n_points)
        
        if phases == 'random':
            rng = np.random.RandomState(self.random_state)
            phase_list = rng.uniform(0, 2 * np.pi, size=len(periods))
        elif phases is None:
            phase_list = [0] * len(periods)
        else:
            phase_list = phases
        
        for period, strength, phase in zip(periods, strengths, phase_list):
            if combine_sin_cos:
                term = (strength / np.sqrt(2)) * np.sin(2 * np.pi * t / period + phase) + \
                       (strength / np.sqrt(2)) * np.cos(2 * np.pi * t / period + phase)
            else:
                term = strength * np.sin(2 * np.pi * t / period + phase)
            seasonality += term
            
        return seasonality

    def add_seasonality(self, df, seasonality_config):
        """
        Add seasonality to DataFrame.
        seasonality_config: dict mapping column name to {'periods': [], 'strengths': []}
        """
        df_seasonal = df.copy()
        n_points = len(df)
        timestamps = df['time'].values if 'time' in df.columns else None
        
        for col, config in seasonality_config.items():
            if col not in df.columns:
                continue
                
            periods = config.get('periods', [12])
            strengths = config.get('strengths', [0.5])
            phases = config.get('phases', 'random')
            combine_sin_cos = config.get('combine_sin_cos', True)
            
            if len(periods) != len(strengths):
                # Broadcast strength if only one provided
                if len(strengths) == 1:
                    strengths = strengths * len(periods)
                else:
                    raise ValueError(f"Length mismatch between periods and strengths for {col}")
            
            season_component = self.generate_seasonality(n_points, periods, strengths, phases=phases, combine_sin_cos=combine_sin_cos, timestamps=timestamps)
            df_seasonal[col] += season_component
            
        return df_seasonal
