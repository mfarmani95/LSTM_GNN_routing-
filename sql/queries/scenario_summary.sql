SELECT
    scenario_id,
    label,
    loss_type,
    lag_days,
    architecture,
    n_gauges,
    mean_gnn_kgess,
    median_gnn_kgess,
    mean_rapid_kgess,
    mean_nwm_kgess,
    mean_improvement_vs_rapid,
    mean_improvement_vs_nwm
FROM scenario_summary
ORDER BY mean_gnn_kgess DESC;
