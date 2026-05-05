import pyscipopt as scip

from utils import timing_store, timeit_accumulate


class OrigVarBranchingEventhdlr(scip.Eventhdlr):
    def __init__(self, all_varschedules, all_patterns_jmt, forbidden_arcs, forced_arcs, 
                 dur,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_varschedules = all_varschedules  # dict indexed by job
        self.all_patterns_jmt = all_patterns_jmt
        self.forbidden_arcs = forbidden_arcs
        self.forced_arcs = forced_arcs
        self.dur = dur

    def eventinit(self):
        self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEFOCUSED, self)

    @timeit_accumulate(timing_store["branching_eventhandler"])
    def eventexec(self, event):
        """
        add branching decisions
        #TODO: only record the newly added branches wrt parent (if possible) to avoid resetting local bounds to 0.  
        """

        mod = self.model
        curr_node_obj = mod.getCurrentNode()
        curr_node = curr_node_obj.getNumber()

        if curr_node >= 2:

            # OrigVar branching
            arcs_to_forbid = self.forbidden_arcs[curr_node]
            arcs_to_force = self.forced_arcs[curr_node]

            if len(arcs_to_forbid) > 0:
                for arc in arcs_to_forbid:
                    (j, m, t0) = arc
                    for var in self.all_patterns_jmt[j, m, t0]:
                        mod.chgVarUb(var, 0)

            if len(arcs_to_force) > 0:
                for arc in arcs_to_force:
                    (j, m, t0) = arc
                    for varsch in self.all_varschedules[j]:
                        var = varsch.var 
                        sch = varsch.schedule
                        if sch.start_times[m] > t0 or sch.compl_times[m] < t0:
                            mod.chgVarUb(var, 0)

        return
