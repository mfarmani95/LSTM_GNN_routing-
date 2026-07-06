SELECT

    b.gauge_id,

    gl.lat,

    gl.lon,

    b.scenario_id,

    b.label,

    b.loss_type,

    b.lag_days,

    b.architecture,

    b.gnn_kgess,

    b.rapid_kgess,

    b.nwm_kgess,

    b.kgess_improvement_rapid,

    b.kgess_improvement_nwm

FROM best_scenario_by_gauge b

LEFT JOIN gauge_locations_combined gl

    ON b.gauge_id = gl.gauge_id

ORDER BY b.gauge_id;