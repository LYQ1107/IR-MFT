"""Read-only full-Train recheck of fitted risk statistics and calibration."""
import hashlib
import numpy as np
from tools.mft_candidate_risk_data import stream_video
from tools.mft_candidate_risk_features import FEATURE_NAMES,IR_FEATURES
from tools.mft_candidate_risk_model import NewtonPass,class_video_weights


class AuditAccumulator:
    def __init__(self,sequence,cut,early_counts,models,videos):
        assert cut>0 and early_counts['positive']+early_counts['negative']==early_counts['count']>0
        self.sequence,self.cut,self.early_counts,self.models=sequence,cut,early_counts,models
        self.videos=videos;self.last_frame=0
        self.count=0;self.positive=0;self.sums=np.zeros(len(FEATURE_NAMES));self.squared=self.sums.copy()
        self.signature=hashlib.sha256()
        self.objectives={k:NewtonPass(m,1e-3)for k,m in models.items()}
        self.calibration_counts=dict(known=0,positive=0,negative=0)
        self.calibration_diagnostics={k:dict(Brier_sum=0.,positive_probability_sum=0.,negative_probability_sum=0.)for k in models}
        self.wrong_scores={k:[]for k in models};self.wrong_score_bytes={k:0 for k in models}

    def observe(self,batch):
        assert batch.sequence==self.sequence and batch.frame>self.last_frame
        self.last_frame=batch.frame
        if batch.frame<=self.cut:
            x=batch.x.astype(np.float64);self.count+=len(x);self.positive+=int(batch.y.sum())
            self.sums+=x.sum(axis=0);self.squared+=(x*x).sum(axis=0)
            self.signature.update(np.asarray([batch.frame,len(x)],dtype='<i8').tobytes())
            self.signature.update(batch.x.tobytes());self.signature.update(batch.y.tobytes())
            weights=class_video_weights(batch.y,self.early_counts['positive'],self.early_counts['negative'],self.videos)
            for objective in self.objectives.values():objective.add(batch.x,batch.y,weights)
        else:
            c=self.calibration_counts;c['known']+=len(batch.y);c['positive']+=int(batch.y.sum());c['negative']+=int((batch.y==0).sum())
            for kind,model in self.models.items():
                p=model.probabilities(batch.x);wrong=p[batch.y==0].copy()
                self.wrong_score_bytes[kind]+=wrong.nbytes
                assert self.wrong_score_bytes[kind]<=67108864,'No wrong calibration edge can be dropped'
                self.wrong_scores[kind].append(wrong)
                d=self.calibration_diagnostics[kind]
                d['Brier_sum']+=float(((p-batch.y)**2).sum())
                d['positive_probability_sum']+=float(p[batch.y==1].sum())
                d['negative_probability_sum']+=float(wrong.sum())

    def assert_early_matches(self,reference):
        assert self.cut==reference['time_cut_frame']
        assert self.count==reference['count']and self.positive==reference['positive']
        assert self.count-self.positive==reference['negative']
        assert self.signature.hexdigest()==reference['feature_target_packet_sha256']
        np.testing.assert_array_equal(self.sums/self.count,reference['mean'])
        np.testing.assert_array_equal(self.squared/self.count,reference['second_moment'])


def audit_full_Train(qualified,statistics,private,models,speed_upper,progress):
    assert len(qualified)==6 and sum(q['frames']for q in qualified.values())==27251
    assert set(qualified)==set(statistics['rows'])==set(statistics['full_teacher_references'])
    assert statistics['feature_names']==list(FEATURE_NAMES)and statistics['Dev_rows']==statistics['Test_rows']==0
    original_coefficients={k:m.coefficients.copy()for k,m in models.items()}
    objective_totals={k:NewtonPass(m,1e-3)for k,m in models.items()}
    all_scores={k:[]for k in models};all_bytes={k:0 for k in models};rows={};means=[];seconds=[]
    for sequence,q in qualified.items():
        cut=int(.75*q['frames']);ref=statistics['rows'][sequence]
        accumulator=AuditAccumulator(sequence,cut,ref,models,len(qualified))
        reference=stream_video(sequence,q,speed_upper,accumulator.observe,
            progress=lambda f:progress(sequence,f))
        assert reference==statistics['full_teacher_references'][sequence]
        accumulator.assert_early_matches(ref)
        expected=private['calibration'][sequence]
        assert accumulator.calibration_counts==expected['counts']
        assert accumulator.calibration_diagnostics==expected['models']
        means.append(accumulator.sums/accumulator.count);seconds.append(accumulator.squared/accumulator.count)
        for kind,total in objective_totals.items():
            partial=accumulator.objectives[kind]
            total.gradient+=partial.gradient;total.hessian+=partial.hessian
            total.loss+=partial.loss;total.weight+=partial.weight;total.rows+=partial.rows
            all_bytes[kind]+=accumulator.wrong_score_bytes[kind];assert all_bytes[kind]<=67108864
            all_scores[kind].extend(accumulator.wrong_scores[kind])
        rows[sequence]=dict(frames=q['frames'],proposals=reference['proposal_rows'],
            early_cut_frame=cut,early_known_edges=accumulator.count,
            early_positive_edges=accumulator.positive,early_negative_edges=accumulator.count-accumulator.positive,
            late_counts=accumulator.calibration_counts,late_diagnostics=accumulator.calibration_diagnostics,
            feature_target_packet_sha256=accumulator.signature.hexdigest(),
            teacher_MOT_sha256=reference['original_teacher_MOT_sha256'],
            bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'],
            teacher_and_all_early_moments_exact=True,all_Train_frame_and_proposal_coverage_exact=True)
    mean=np.mean(means,axis=0);second=np.mean(seconds,axis=0)
    scale=np.maximum(np.sqrt(np.maximum(second-mean*mean,0)),1e-4)
    np.testing.assert_array_equal(mean,statistics['equal_video_mean'])
    np.testing.assert_array_equal(scale,statistics['equal_video_scale'])
    for kind,model in models.items():
        expected_mean=mean.copy();expected_scale=scale.copy()
        if kind=='RGB':expected_mean[list(IR_FEATURES)]=0.;expected_scale[list(IR_FEATURES)]=1.
        np.testing.assert_array_equal(model.mean,expected_mean)
        np.testing.assert_array_equal(model.scale,expected_scale)
    calibrated={};convergence={}
    for kind,total in objective_totals.items():
        values=np.concatenate(all_scores[kind]);assert len(values)>0 and np.isfinite(values).all()
        calibrated[kind]=float(np.quantile(values,.99));assert calibrated[kind]==private['thresholds'][kind]
        assert abs(total.weight-1.)<1e-7 and total.rows==statistics['known_early_edges']
        loss,gradient,hessian=total.objective()
        direction=np.linalg.solve(hessian+1e-8*np.eye(len(gradient)),gradient)
        assert np.isfinite(loss)and np.isfinite(gradient).all()and np.isfinite(hessian).all()
        gradient_norm=float(np.linalg.norm(gradient));direction_norm=float(np.linalg.norm(direction))
        last=private['fit_history'][-1]['models'][kind]
        convergence[kind]=dict(exact_final_weighted_Train_objective=loss,
            exact_final_gradient_L2_norm=gradient_norm,estimated_undamped_next_Newton_step_L2_norm=direction_norm,
            final_objective_no_larger_than_before_last_step=loss<=last['weighted_objective_before_step']+1e-10,
            stationarity_diagnostic_gradient_below_1e_minus3=gradient_norm<1e-3,
            no_optimizer_update_performed_by_auditor=True,known_early_edges=total.rows,
            objective_weight=total.weight,late_wrong_edges=len(values),late_wrong_score_bytes=values.nbytes)
        np.testing.assert_array_equal(models[kind].coefficients,original_coefficients[kind])
    assert np.max(np.abs(models['RGB'].coefficients[list(IR_FEATURES)]))<1e-12
    return dict(status='INDEPENDENTLY_REBUILT_ALL6_TRAIN_STATISTICS_CALIBRATION_AND_FINAL_OBJECTIVE',
        frames=27251,sequences=6,rows=rows,thresholds=calibrated,convergence=convergence,
        unique_proposals=sum(q['proposals']for q in rows.values()),
        all_early_feature_target_packets_exact=True,all_late_calibration_diagnostics_exact=True,
        RGB_mask_IR_coefficients_zero=True,coefficient_values_public=False,
        new_model_fits=0,new_optimizer_steps=0,new_NN_work=False,
        Dev_or_Test_inputs_read=False,teacher_replay_with_existing_compact_banks=True,
        encoder_Train_identity_exposure_disclosed=True)
