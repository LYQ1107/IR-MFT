"""Transactional guard for future ledger experiments, not the frozen Dev run.

The first ledger could release a currently observed owner with an unscored
edge after a dependent handoff was rejected. Unscored does not imply a low
posterior. Roll back that frame's remapping; never alter physical tracking.
"""
from tools.mft_identity_ledger import LedgerTracker


class ExclusiveLedgerTracker(LedgerTracker):
    def remap(self,frame,x,eligible,roots,assignments,obs):
        before=self.ledger.bindings.copy();pending=self.pending.copy();counter=self.ledger.generated
        applied={k:self.statistics[k]for k in ['rebindings','displaced_owners']}
        super().remap(frame,x,eligible,roots,assignments,obs)
        columns={root:j for j,root in enumerate(roots)}
        unknown_displacement=any(self.ledger.bindings[owner]>counter and before[owner]in columns
            and not eligible[index,columns[before[owner]]]for owner,index in assignments.items())
        if unknown_displacement:
            self.ledger.bindings=before;self.ledger.generated=counter;self.pending=pending
            self.statistics.update(applied)
            self.statistics['unscored_owner_frame_rollbacks']=self.statistics.get('unscored_owner_frame_rollbacks',0)+1
