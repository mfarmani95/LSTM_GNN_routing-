SELECT

    (SELECT COUNT(*) FROM routing_nodes) AS n_nodes,

    (SELECT COUNT(*) FROM routing_edges) AS n_edges,

    (SELECT COUNT(*) FROM routing_gauges) AS n_graph_gauges,

    (SELECT COUNT(*) FROM runoff_mapping) AS n_runoff_mappings;