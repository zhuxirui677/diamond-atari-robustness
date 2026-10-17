"""Task definitions and CPU-only metrics for the paired DIAMOND evaluation."""
import math

TASKS = {
    'Breakout': {'task': 'score_once', 'positive_steps_required': 1,
                 'description': '100步内至少1个正奖励决策步'},
    'Pong': {'task': 'score_one_point', 'positive_steps_required': 1,
             'description': '100步内至少获得1分（不是赢得整局）'},
    'Boxing': {'task': 'score_five_times', 'positive_steps_required': 5,
               'description': '100步内至少5个正奖励决策步（不等于5拳）'},
}
HORIZON = 100
DROP_LEVELS = (.05, .10, .20, .30, .40)
NOISE_LEVELS = (.01, .02, .05, .10, .20)

def conditions(smoke=False):
    if smoke:
        return [('normal', 0.), ('sham', 0.), ('drop', .10), ('noise', .02)]
    return [('normal', 0.), ('sham', 0.)] + [('drop', x) for x in DROP_LEVELS] + [('noise', x) for x in NOISE_LEVELS]

def metrics(rewards, threshold, horizon=HORIZON):
    if len(rewards) > horizon:
        raise ValueError('Rollout exceeds the fixed decision-step budget')
    positive = sum(r > 0 for r in rewards)
    hits = 0
    first = None
    for step, r in enumerate(rewards, 1):
        hits += r > 0
        if hits >= threshold and first is None:
            first = step
    return dict(success=int(positive >= threshold), positive_steps=positive,
                raw_return=sum(rewards), clipped_return=sum((r > 0)-(r < 0) for r in rewards),
                first_success_step=first, executed_steps=len(rewards), budget=horizon)

def wilson(successes, n):
    if n <= 0:
        raise ValueError('No trials; never report missing runs as failures')
    z = 1.959963984540054
    p = successes / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-r, c+r

def summarize(rows, cases):
    import random
    groups = {}
    for row in rows:
        key = row['game'], row['domain'], row['condition'], row['level']
        group = groups.setdefault(key, {})
        if row['case'] in group:
            raise ValueError('Duplicate case')
        group[row['case']] = row
    result = []
    for (game, domain, condition, level), group in groups.items():
        if set(group) != set(range(cases)):
            raise ValueError('Missing paired cases')
        base = groups[(game, domain, 'normal', 0.)]
        deltas = [base[i]['success']-group[i]['success'] for i in range(cases)]
        rng = random.Random(20261011)
        samples = sorted(sum(rng.choice(deltas) for _ in deltas)/cases for _ in range(2000))
        k = sum(r['success'] for r in group.values())
        lo, hi = wilson(k, cases)
        result.append(dict(game=game, domain=domain, condition=condition, level=level,
            cases=cases, successes=k, success_rate=k/cases, ci95_low=lo, ci95_high=hi,
            drop_vs_normal=sum(deltas)/cases, paired_drop_ci95_low=samples[50],
            paired_drop_ci95_high=samples[1949],
            mean_clipped_return=sum(r['clipped_return'] for r in group.values())/cases))
    return result
