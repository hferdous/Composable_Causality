class StructuralModel:
    def __init__(self, links, n_vars=None):
        """
        Initialize with links: {(source, lag, target, power): weight}
        n_vars: int, optional. If provided, ensures equations are generated for all variables X1..Xn.
        """
        self.links = {}
        for key, weight in links.items():
            if len(key) == 3:
                source, lag, target = key
                power = 1
            else:
                source, lag, target, power = key
            self.links[(source, lag, target, power)] = weight

        self.n_vars = n_vars
        self.equations = self._generate_equation_strings()
        
        # Organize by target for faster computation
        self.coeffs = {} # target -> [(source, lag, weight, power)]
        for (source, lag, target, power), weight in self.links.items():
            if target not in self.coeffs:
                self.coeffs[target] = []
            self.coeffs[target].append((source, lag, weight, power))
            
    def _generate_equation_strings(self):
        eqs = []
        targets = set(t for s, l, t, p in self.links.keys())
        
        # If n_vars is provided, ensure all X1..Xn are in targets
        if self.n_vars is not None:
            for i in range(self.n_vars):
                targets.add(f"X{i+1}")
        
        for target in sorted(list(targets)):
            terms = []
            for (source, lag, t, power), weight in self.links.items():
                if t == target:
                    lag_str = "t" if lag == 0 else f"t{lag}"
                    power_str = f"^{power}" if power != 1 else ""
                    terms.append(f"{weight:.2f} * {source}[{lag_str}]{power_str}")
            
            if terms:
                rhs = " + ".join(terms) + f" + e_{target}"
            else:
                rhs = f"e_{target}"
                
            eqs.append(f"{target}[t] = {rhs}")
        return eqs


    def compute_value(self, target, t, history, current_values=None):
        """
        Compute value for target at time t.
        history: dict of variable name -> array of past values
        current_values: dict of variable name -> current value (for lag 0 dependencies)
        """
        if target not in self.coeffs:
            return 0.0
            
        value = 0.0
        for source, lag, weight, power in self.coeffs[target]:
            term_val = 0.0
            if lag == 0:
                if current_values is None or source not in current_values:
                    # If source is U, it might be in current_values
                    if source == 'U' and current_values and 'U' in current_values:
                         term_val = current_values['U']
                    else:
                        raise ValueError(f"Lag 0 dependency {source}->{target} but {source} not computed yet.")
                else:
                    term_val = current_values[source]
            else:
                # lag is negative
                past_idx = t + lag
                if past_idx >= 0:
                    term_val = history[source][past_idx]
            
            # Apply power
            if power != 1:
                # Handle negative bases with fractional powers if necessary, 
                # but usually power is integer 1, 2, 3.
                term_val = term_val ** power
                
            value += weight * term_val
                    
        return value

