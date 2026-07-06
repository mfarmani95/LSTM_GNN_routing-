SELECT
    gauge_id,
    lat,
    lon,
    scenario_id,
    label,
    loss_type,
    lag_days,
    architecture,
    gnn_kgess,
    rapid_kgess,
    nwm_kgess,
    kgess_improvement_rapid,
    kgess_improvement_nwm
FROM best_scenario_by_gauge
ORDER BY gauge_id;
