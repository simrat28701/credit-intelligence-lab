SELECT
    "Credit band",
    "September status",
    COUNT(*) AS customers,
    SUM(DEFAULT_NEXT_MONTH) AS observed_defaults,
    AVG(DEFAULT_NEXT_MONTH) AS default_rate,
    MEDIAN(LIMIT_BAL) AS median_limit_ntd
FROM portfolio
GROUP BY "Credit band", "September status"
ORDER BY "Credit band", "September status"
