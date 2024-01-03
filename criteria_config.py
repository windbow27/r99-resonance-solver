
class Criteria:
    def __init__(self):
        self.get_hp = lambda x: x.stats.hp
        self.get_atk = lambda x: x.stats.atk
        self.get_reality_def = lambda x: x.stats.reality_def
        self.get_mental_def = lambda x: x.stats.mental_def
        self.get_crit_rate = lambda x: x.stats.crit_rate
        self.get_crit_rate_def = lambda x: x.stats.crit_rate_def
        self.get_crit_dmg = lambda x: x.stats.crit_dmg
        self.get_crit_dmg_def = lambda x: x.stats.crit_dmg_def
        self.get_dmg_bonus = lambda x: x.stats.dmg_bonus
        self.get_dmg_taken_reduction = lambda x: x.stats.dmg_taken_reduction
        self.get_total_stats = lambda x: int(round(x.stats.total_stat_gains(), 2))

        self._ordering = None
        self._name = "Criteria"
    
    # List[Tuple(int, lambda)]
    # The integer should be the number of candidates to take 
    # (2, lambda x: x.stats.crit_rate) - this would
    def get_ordering(self):
        return self._ordering
    
    def __str__(self):
        return self._name
    

class CritBuild(Criteria):
    def __init__(self):
        super().__init__()
        self._name = 'Crit'

        self._ordering = [
            (3, lambda x: x.stats.crit_rate),
            (3, lambda x: x.stats.atk + x.stats.crit_dmg),
            (2, lambda x: x.stats.dmg_taken_reduction + x.stats.reality_def + x.stats.mental_def)
        ]

class AttackBuild(Criteria):
    def __init__(self):
        super().__init__()
        self._name = 'Attack'
        self._ordering = [
            (3, lambda x: x.stats.atk + x.stats.dmg_bonus),
            (3, lambda x: x.stats.reality_def + x.stats.mental_def + x.stats.dmg_taken_reduction),
        ]

class DefBuild(Criteria):
    def __init__(self):
        super().__init__()
        self._name = 'Def'
        self._ordering = [
            (3, lambda x: x.stats.dmg_taken_reduction),
            (3, lambda x: x.stats.hp + x.stats.reality_def + x.stats.mental_def),
            (2, lambda x: x.stats.atk + x.stats.dmg_bonus)
        ]