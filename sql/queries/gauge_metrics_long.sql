SELECT
    m.gauge_id,
    g.lat,
    g.lon,
    m.scenario_id,
    s.label,
    s.loss_type,
    s.lag_days,
    s.architecture,
    m.gnn_kgess,
    m.rapid_kgess,
    m.nwm_kgess,
    m.kgess_improvement_rapid,
    m.kgess_improvement_nwm
FROM gauge_metrics m
JOIN scenarios s
    ON m.scenario_id = s.scenario_id
LEFT JOIN gauges g
    ON m.gauge_id = g.gauge_id
ORDER BY
    s.loss_type,
    s.lag_days,
    m.gauge_id;
