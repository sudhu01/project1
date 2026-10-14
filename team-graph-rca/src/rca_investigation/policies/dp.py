from time import perf_counter


class SearchLimit(RuntimeError):
    pass


class BoundedDP:
    def __init__(self, model, specs, depth=2, max_expanded_states=50000, defer=0.25, correct=1.0, incorrect=0.0):
        if depth < 1 or max_expanded_states < 1:
            raise ValueError('positive search limits required')
        self.model, self.specs = model, tuple(specs)
        self.depth, self.max_expanded_states = depth, max_expanded_states
        self.defer, self.correct, self.incorrect = defer, correct, incorrect
        self.stats = {}

    def choose(self, observation):
        started = perf_counter()
        self.stats = {'expanded_states': 0, 'cache_hits': 0, 'fallback': False}
        permitted = set(observation['mask'])
        specs = tuple(s for s in self.specs if s.action_id in permitted)
        cache = {}

        def values(history, budget, horizon, depth, limited=True):
            key = (tuple(sorted(history)), budget, horizon, depth)
            if key in cache:
                self.stats['cache_hits'] += 1
                return cache[key]
            self.stats['expanded_states'] += 1
            if limited and self.stats['expanded_states'] > self.max_expanded_states:
                raise SearchLimit()
            probs = self.model.diagnosis(history)
            result = {'STOP': self.incorrect + max(probs.values())*(self.correct-self.incorrect), 'DEFER': self.defer}
            seen = {a for a, _ in history}
            if depth > 0 and horizon > 0:
                for spec in specs:
                    if spec.action_id in seen or spec.credits > budget:
                        continue
                    total = -observation['lambda_cost'] * spec.credits
                    for symbol, probability in self.model.predictions(history, spec.action_id).items():
                        if probability > 0:
                            future = values(history+((spec.action_id, symbol),), budget-spec.credits, horizon-1, depth-1, limited)
                            total += probability * max(future.values())
                    result[spec.action_id] = total
            cache[key] = result
            return result

        try:
            result = values(tuple(observation['history']), observation['remaining_credits'], observation['remaining_probes'], self.depth)
        except SearchLimit:
            cache.clear()
            self.stats['fallback'] = True
            result = values(tuple(observation['history']), observation['remaining_credits'], observation['remaining_probes'], 1, False)
        self.stats['elapsed_seconds'] = perf_counter()-started
        self.stats['values'] = result
        # Deferral wins an exact terminal tie; terminals win a probe tie.
        order = ('DEFER', 'STOP') + tuple(s.action_id for s in specs)
        best = max(result.values())
        return next(a for a in order if a in result and abs(result[a]-best) < 1e-12)
