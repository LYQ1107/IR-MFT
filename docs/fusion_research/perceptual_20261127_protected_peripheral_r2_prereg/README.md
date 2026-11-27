# R2 execution correction before any new HOTA

The R1 real-Train engineering worker and controller exited1 because the experiment worker module omitted the mot_line import. No source producer, diagnostic or HOTA evaluation started in that attempt. Its source snapshots, registration, unit receipt, actual execution status, state and raw logs are preserved here. The preregistered algorithm, 18 cases, thresholds, selection rule and all neural inputs are unchanged.

R2 adds the missing import, preserves the failure as a registered dependency, and uses a separate execution directory. The six passed core/serialization regressions are reused with their unchanged source hashes. Real-Train engineering is being checked again before any data source or HOTA phase. This correction establishes no new accuracy result. Fixed original RGB baseline29.5289706874, minimum34.5289706874 and stretch39.5289706874 remain unchanged; goal active and unmet.
