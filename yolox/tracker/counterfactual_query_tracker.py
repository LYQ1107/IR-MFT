"""Accept native evidence only when it changes a frozen RGB query usefully.

The ordinary point matcher owns the default decision. A merely confident
auxiliary query cannot replace it: the native cue must improve a candidate's
posterior relative to the same query with the native increment removed.
"""
import numpy as np

from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.native_query_tracker import NativeQueryTracker


class CounterfactualQueryTracker(NativeQueryTracker):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.statistics.update(counterfactual_rows=0, counterfactual_clear_preserved=0,
                               counterfactual_null_rejections=0)

    def counterfactual_probabilities(self, packet):
        # Undo only the bounded native real-candidate increment. The null logit
        # is unchanged. This ratio cancels the native normalization constant.
        weights = packet['probability'].astype(np.float64)*np.exp(-packet['delta'].astype(np.float64))
        null = packet['null'].astype(np.float64)
        denominator = np.maximum(weights.sum(-1)+null, 1e-30)
        return weights/denominator[:,None], null/denominator

    def recovery_scores(self, frame, context, data):
        a,b = map(int,data['offsets'][frame-1:frame+1])
        maximum_lag = self.config.get('counterfactual_recovery_maximum_lag')
        if self.config.get('disable_native_increment', False) or (maximum_lag is not None and frame-context['frame']>maximum_lag):
            return np.zeros(b-a,np.float64), np.zeros(b-a,np.float32)
        packet = self.query_posterior(frame,context)
        reference,_ = self.counterfactual_probabilities(packet)
        row = context['index']-packet['start']
        probability = packet['probability'][row].astype(np.float64).copy()
        accept = ((probability-reference[row] >= self.config.get('counterfactual_minimum_gain',.05))
                  & (probability > packet['null'][row]+self.config.get('query_margin',.1)))
        probability[~accept] = 0
        return probability,packet['delta'][row]

    def rank_cost(self, tracks, observations, base, frame, stage):
        result = CalibratedForegroundTracker.rank_cost(self,tracks,observations,base,frame,stage)
        if (not base.size or self.config.get('disable_native_increment',False)
                or not self.config.get('query_ranking',True)
                or stage not in self.config.get('point_stages',['primary','secondary'])):
            return result
        columns = np.array([o.evidence['index']-self.current_start for o in observations],int)
        limit = {'primary':.9,'secondary':.5,'pending':.7}[stage]
        gain_required = float(self.config.get('counterfactual_minimum_gain',.05))
        for row,track in enumerate(tracks):
            history = track.object_observations if self.config.get('object_files',True) else [track.evidence]
            if not history:
                continue
            context = history[-1]
            lag = frame-context['frame']
            maximum_lag = self.config.get('counterfactual_matching_maximum_lag')
            if maximum_lag is not None and lag>maximum_lag:
                continue
            if self.config.get('counterfactual_preserve_clear',True) and lag in self.config.get('lags',[1,2,4,8]):
                start,old_p,_,vote,good,total = self.predict_pairs(frame,lag)
                index = context['index']-start
                p = old_p[index]
                best = int(p.argmax())
                second = float(np.partition(p,-2)[-2]) if len(p)>1 else 0.
                if p[best]>=.95 and p[best]-second>=.15 and good[index]>=4 and vote[index,best]>=.3:
                    self.statistics['counterfactual_clear_preserved'] += 1
                    continue
            packet = self.query_posterior(frame,context)
            reference,reference_null = self.counterfactual_probabilities(packet)
            index = context['index']-packet['start']
            all_p = packet['probability'][index]
            best = int(all_p.argmax())
            maximum = float(all_p[best])
            second = float(np.partition(all_p,-2)[-2]) if len(all_p)>1 else 0.
            null = float(packet['null'][index])
            if (self.config.get('counterfactual_allow_null_rejection',True) and self.config.get('query_null',True) and null>=.8 and null-reference_null[index]>=gain_required
                    and (packet['good'][index]<4 or packet['vote'][index].max(initial=0)<.15)):
                result[row] = np.maximum(result[row],limit+1e-4)
                self.statistics['counterfactual_null_rejections'] += 1
                continue
            if (maximum<self.config.get('query_minimum_probability',.8)
                    or maximum-max(second,null)<self.config.get('query_margin',.1)
                    or maximum-reference[index,best]<gain_required
                    or self.data['scores'][self.current_start+best]<.3):
                continue
            p = all_p[columns].astype(np.float64)
            scores = np.array([o.score for o in observations]) if stage!='secondary' else np.ones(len(observations))
            weight = float(self.config.get('query_weight',.8))*maximum
            adjusted = (1-weight)*base[row]+weight*(1-p*scores)
            rejected = base[row]>limit
            support = ((packet['vote'][index,columns]>=.15)
                       | (packet['base'][index,columns,23]>=.2)
                       | ((packet['cue'][index,columns,0]>=.4)
                          & (packet['cue'][index,columns,3:5].min(-1)>=.4)))
            opening = (p>=self.config.get('query_opening_probability',.95))&support
            adjusted[rejected&~opening] = result[row,rejected&~opening]
            result[row] = adjusted
            if best in columns:
                self.trusted[(track.track_id,self.current_start+best)] = maximum
            self.statistics['counterfactual_rows'] += 1
            self.statistics['IR_applied_rows'] += int(self.native_cue=='ir')
            self.statistics['native_opened_edges'] += int((rejected&(adjusted<=limit)).sum())
        return result
