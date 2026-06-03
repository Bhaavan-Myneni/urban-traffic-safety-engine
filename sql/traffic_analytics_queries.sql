-- =============================================================================
-- Urban Traffic Safety Engine — Advanced Analytics Queries
-- =============================================================================
-- Database : PostgreSQL
-- Schema   : traffic_detections, congestion_metrics, object_summary
--
-- Usage:
--   psql -U traffic_user -d traffic_safety -f sql/traffic_analytics_queries.sql
--
-- Each query is self-contained and can be run independently.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Query 1: Total detections by object type
-- Purpose  : Baseline volume profile — how many YOLO detections exist per class.
-- Techniques: CTE, GROUP BY, CASE WHEN (confidence tier), window function (share)
-- -----------------------------------------------------------------------------
WITH detection_counts AS (
    SELECT
        object_type,
        COUNT(*) AS total_detections,
        ROUND(AVG(confidence)::numeric, 4) AS avg_confidence
    FROM traffic_detections
    GROUP BY object_type
),
ranked AS (
    SELECT
        object_type,
        total_detections,
        avg_confidence,
        CASE
            WHEN avg_confidence >= 0.80 THEN 'High confidence'
            WHEN avg_confidence >= 0.60 THEN 'Moderate confidence'
            ELSE 'Low confidence'
        END AS confidence_tier,
        SUM(total_detections) OVER () AS grand_total
    FROM detection_counts
)
SELECT
    object_type,
    total_detections,
    ROUND(100.0 * total_detections / grand_total, 2) AS pct_of_all_detections,
    avg_confidence,
    confidence_tier
FROM ranked
ORDER BY total_detections DESC;


-- -----------------------------------------------------------------------------
-- Query 2: Top 10 busiest cameras by average traffic density
-- Purpose  : Identify cameras with the highest sustained congestion pressure.
-- Techniques: CTE, GROUP BY, CASE WHEN (density band), RANK()
-- -----------------------------------------------------------------------------
WITH camera_density AS (
    SELECT
        camera_id,
        COUNT(DISTINCT video_id) AS videos_observed,
        COUNT(*) AS minute_samples,
        ROUND(AVG(traffic_density)::numeric, 4) AS avg_traffic_density,
        ROUND(MAX(traffic_density)::numeric, 4) AS peak_traffic_density,
        SUM(vehicle_count) AS total_vehicles,
        SUM(pedestrian_count) AS total_pedestrians
    FROM congestion_metrics
    GROUP BY camera_id
),
ranked_cameras AS (
    SELECT
        camera_id,
        videos_observed,
        minute_samples,
        avg_traffic_density,
        peak_traffic_density,
        total_vehicles,
        total_pedestrians,
        CASE
            WHEN avg_traffic_density < 5  THEN 'Low'
            WHEN avg_traffic_density < 10 THEN 'Medium'
            WHEN avg_traffic_density < 20 THEN 'High'
            ELSE 'Severe'
        END AS avg_density_band,
        RANK() OVER (ORDER BY avg_traffic_density DESC) AS density_rank
    FROM camera_density
)
SELECT
    density_rank,
    camera_id,
    avg_traffic_density,
    peak_traffic_density,
    avg_density_band,
    minute_samples,
    total_vehicles,
    total_pedestrians
FROM ranked_cameras
WHERE density_rank <= 10
ORDER BY density_rank;


-- -----------------------------------------------------------------------------
-- Query 3: Vehicle vs pedestrian ratio by camera
-- Purpose  : Compare modal mix — useful for safety planning and signal timing.
-- Techniques: CTE, GROUP BY, CASE WHEN (vehicle classification), window function
-- -----------------------------------------------------------------------------
WITH modal_counts AS (
    SELECT
        camera_id,
        SUM(
            CASE
                WHEN object_type IN ('car', 'truck', 'bus', 'motorcycle', 'bicycle')
                    THEN count
                ELSE 0
            END
        ) AS vehicle_detections,
        SUM(
            CASE
                WHEN object_type = 'person' THEN count
                ELSE 0
            END
        ) AS pedestrian_detections,
        SUM(count) AS total_detections
    FROM object_summary
    GROUP BY camera_id
),
ratios AS (
    SELECT
        camera_id,
        vehicle_detections,
        pedestrian_detections,
        total_detections,
        CASE
            WHEN pedestrian_detections = 0 THEN NULL
            ELSE ROUND(vehicle_detections::numeric / pedestrian_detections, 2)
        END AS vehicle_to_pedestrian_ratio,
        CASE
            WHEN total_detections = 0 THEN 'Unknown'
            WHEN vehicle_detections >= pedestrian_detections * 5 THEN 'Vehicle-dominated'
            WHEN pedestrian_detections >= vehicle_detections * 2 THEN 'Pedestrian-heavy'
            ELSE 'Mixed traffic'
        END AS traffic_profile,
        SUM(total_detections) OVER () AS network_total
    FROM modal_counts
)
SELECT
    camera_id,
    vehicle_detections,
    pedestrian_detections,
    vehicle_to_pedestrian_ratio,
    traffic_profile,
    ROUND(100.0 * total_detections / network_total, 2) AS pct_of_network_detections
FROM ratios
ORDER BY vehicle_to_pedestrian_ratio DESC NULLS LAST;


-- -----------------------------------------------------------------------------
-- Query 4: Congestion level distribution
-- Purpose  : Network-wide breakdown of how often each congestion tier appears.
-- Techniques: CTE, GROUP BY, CASE WHEN (severity ordering), window function
-- -----------------------------------------------------------------------------
WITH level_counts AS (
    SELECT
        congestion_level,
        COUNT(*) AS observation_count,
        COUNT(DISTINCT camera_id) AS cameras_affected,
        ROUND(AVG(traffic_density)::numeric, 4) AS avg_density_in_level,
        ROUND(MIN(traffic_density)::numeric, 4) AS min_density,
        ROUND(MAX(traffic_density)::numeric, 4) AS max_density
    FROM congestion_metrics
    GROUP BY congestion_level
),
ordered_levels AS (
    SELECT
        congestion_level,
        observation_count,
        cameras_affected,
        avg_density_in_level,
        min_density,
        max_density,
        CASE congestion_level
            WHEN 'Low'    THEN 1
            WHEN 'Medium' THEN 2
            WHEN 'High'   THEN 3
            WHEN 'Severe' THEN 4
            ELSE 5
        END AS severity_order,
        SUM(observation_count) OVER () AS total_observations
    FROM level_counts
)
SELECT
    congestion_level,
    observation_count,
    ROUND(100.0 * observation_count / total_observations, 2) AS pct_of_observations,
    cameras_affected,
    avg_density_in_level,
    min_density,
    max_density
FROM ordered_levels
ORDER BY severity_order;


-- -----------------------------------------------------------------------------
-- Query 5: Peak congestion minute per camera
-- Purpose  : Find the single worst minute for each camera (highest density).
-- Techniques: CTE, window function (ROW_NUMBER), CASE WHEN
-- -----------------------------------------------------------------------------
WITH minute_ranked AS (
    SELECT
        camera_id,
        video_id,
        minute,
        traffic_density,
        congestion_level,
        vehicle_count,
        pedestrian_count,
        ROW_NUMBER() OVER (
            PARTITION BY camera_id
            ORDER BY traffic_density DESC, minute ASC
        ) AS peak_rank
    FROM congestion_metrics
),
peak_minutes AS (
    SELECT
        camera_id,
        video_id,
        minute AS peak_minute,
        traffic_density AS peak_density,
        congestion_level,
        vehicle_count,
        pedestrian_count,
        CASE congestion_level
            WHEN 'Severe' THEN 'Critical — immediate review'
            WHEN 'High'   THEN 'Elevated — monitor closely'
            WHEN 'Medium' THEN 'Moderate — routine monitoring'
            ELSE 'Normal operations'
        END AS operational_alert
    FROM minute_ranked
    WHERE peak_rank = 1
)
SELECT
    camera_id,
    video_id,
    peak_minute,
    peak_density,
    congestion_level,
    vehicle_count,
    pedestrian_count,
    operational_alert
FROM peak_minutes
ORDER BY peak_density DESC;


-- -----------------------------------------------------------------------------
-- Query 6: Average confidence by object type
-- Purpose  : Model quality check — which classes does YOLO detect most reliably?
-- Techniques: CTE, GROUP BY, CASE WHEN (quality bucket), window function
-- -----------------------------------------------------------------------------
WITH confidence_stats AS (
    SELECT
        object_type,
        COUNT(*) AS detection_count,
        ROUND(AVG(confidence)::numeric, 4) AS avg_confidence,
        ROUND(MIN(confidence)::numeric, 4) AS min_confidence,
        ROUND(MAX(confidence)::numeric, 4) AS max_confidence,
        ROUND(STDDEV(confidence)::numeric, 4) AS stddev_confidence
    FROM traffic_detections
    GROUP BY object_type
),
quality_tiers AS (
    SELECT
        object_type,
        detection_count,
        avg_confidence,
        min_confidence,
        max_confidence,
        stddev_confidence,
        CASE
            WHEN avg_confidence >= 0.75 THEN 'Reliable'
            WHEN avg_confidence >= 0.55 THEN 'Acceptable'
            ELSE 'Review threshold'
        END AS model_quality,
        RANK() OVER (ORDER BY avg_confidence DESC) AS confidence_rank
    FROM confidence_stats
)
SELECT
    confidence_rank,
    object_type,
    detection_count,
    avg_confidence,
    min_confidence,
    max_confidence,
    stddev_confidence,
    model_quality
FROM quality_tiers
ORDER BY confidence_rank;


-- -----------------------------------------------------------------------------
-- Query 7: Camera ranking using RANK()
-- Purpose  : Ordered leaderboard of cameras by average traffic density.
-- Techniques: CTE, GROUP BY, RANK(), DENSE_RANK(), CASE WHEN
-- -----------------------------------------------------------------------------
WITH camera_averages AS (
    SELECT
        camera_id,
        COUNT(*) AS samples,
        ROUND(AVG(traffic_density)::numeric, 4) AS avg_traffic_density,
        ROUND(AVG(avg_vehicles_per_frame)::numeric, 4) AS avg_vehicles_per_frame,
        ROUND(AVG(avg_pedestrians_per_frame)::numeric, 4) AS avg_pedestrians_per_frame,
        MODE() WITHIN GROUP (ORDER BY congestion_level) AS modal_congestion_level
    FROM congestion_metrics
    GROUP BY camera_id
),
ranked AS (
    SELECT
        camera_id,
        samples,
        avg_traffic_density,
        avg_vehicles_per_frame,
        avg_pedestrians_per_frame,
        modal_congestion_level,
        RANK() OVER (ORDER BY avg_traffic_density DESC) AS rank_by_density,
        DENSE_RANK() OVER (ORDER BY avg_traffic_density DESC) AS dense_rank_by_density,
        CASE
            WHEN RANK() OVER (ORDER BY avg_traffic_density DESC) <= 3 THEN 'Top priority'
            WHEN RANK() OVER (ORDER BY avg_traffic_density DESC) <= 10 THEN 'High priority'
            ELSE 'Standard monitoring'
        END AS monitoring_priority
    FROM camera_averages
)
SELECT
    rank_by_density,
    dense_rank_by_density,
    camera_id,
    avg_traffic_density,
    modal_congestion_level,
    avg_vehicles_per_frame,
    avg_pedestrians_per_frame,
    samples,
    monitoring_priority
FROM ranked
ORDER BY rank_by_density;


-- -----------------------------------------------------------------------------
-- Query 8: Rolling average traffic density by camera
-- Purpose  : Smooth minute-level noise to reveal congestion trends over time.
-- Techniques: CTE, window function (rolling AVG), CASE WHEN (trend direction)
-- -----------------------------------------------------------------------------
WITH ordered_metrics AS (
    SELECT
        camera_id,
        video_id,
        minute,
        traffic_density,
        congestion_level,
        ROW_NUMBER() OVER (
            PARTITION BY camera_id, video_id
            ORDER BY minute
        ) AS seq_minute
    FROM congestion_metrics
),
rolling AS (
    SELECT
        camera_id,
        video_id,
        minute,
        traffic_density,
        congestion_level,
        ROUND(
            AVG(traffic_density) OVER (
                PARTITION BY camera_id, video_id
                ORDER BY minute
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            )::numeric,
            4
        ) AS rolling_3min_avg_density,
        LAG(traffic_density, 1) OVER (
            PARTITION BY camera_id, video_id
            ORDER BY minute
        ) AS prior_minute_density
    FROM ordered_metrics
)
SELECT
    camera_id,
    video_id,
    minute,
    traffic_density AS raw_density,
    rolling_3min_avg_density,
    congestion_level,
    CASE
        WHEN prior_minute_density IS NULL THEN 'Baseline'
        WHEN traffic_density > prior_minute_density * 1.10 THEN 'Worsening'
        WHEN traffic_density < prior_minute_density * 0.90 THEN 'Improving'
        ELSE 'Stable'
    END AS minute_trend
FROM rolling
ORDER BY camera_id, video_id, minute;


-- -----------------------------------------------------------------------------
-- Query 9: Percentile ranking of cameras by traffic density
-- Purpose  : Relative standing — where does each camera sit in the network?
-- Techniques: CTE, GROUP BY, PERCENT_RANK(), NTILE(), CASE WHEN
-- -----------------------------------------------------------------------------
WITH camera_density AS (
    SELECT
        camera_id,
        ROUND(AVG(traffic_density)::numeric, 4) AS avg_traffic_density,
        ROUND(MAX(traffic_density)::numeric, 4) AS max_traffic_density,
        COUNT(*) AS minute_observations
    FROM congestion_metrics
    GROUP BY camera_id
),
percentiles AS (
    SELECT
        camera_id,
        avg_traffic_density,
        max_traffic_density,
        minute_observations,
        PERCENT_RANK() OVER (ORDER BY avg_traffic_density) AS density_percentile,
        NTILE(4) OVER (ORDER BY avg_traffic_density) AS density_quartile,
        CASE
            WHEN PERCENT_RANK() OVER (ORDER BY avg_traffic_density) >= 0.90 THEN 'Top 10%'
            WHEN PERCENT_RANK() OVER (ORDER BY avg_traffic_density) >= 0.75 THEN 'Upper quartile'
            WHEN PERCENT_RANK() OVER (ORDER BY avg_traffic_density) >= 0.50 THEN 'Above median'
            WHEN PERCENT_RANK() OVER (ORDER BY avg_traffic_density) >= 0.25 THEN 'Below median'
            ELSE 'Bottom quartile'
        END AS relative_standing
    FROM camera_density
)
SELECT
    camera_id,
    avg_traffic_density,
    max_traffic_density,
    minute_observations,
    ROUND(density_percentile::numeric, 4) AS density_percentile,
    density_quartile,
    relative_standing
FROM percentiles
ORDER BY density_percentile DESC;


-- -----------------------------------------------------------------------------
-- Query 10: High-risk cameras where congestion_level = 'Severe'
-- Purpose  : Safety escalation list — cameras with severe congestion events.
-- Techniques: CTE, GROUP BY, CASE WHEN, window function (event ranking)
-- -----------------------------------------------------------------------------
WITH severe_events AS (
    SELECT
        camera_id,
        video_id,
        minute,
        traffic_density,
        vehicle_count,
        pedestrian_count,
        sampled_frame_count,
        congestion_level
    FROM congestion_metrics
    WHERE congestion_level = 'Severe'
),
camera_risk AS (
    SELECT
        camera_id,
        COUNT(*) AS severe_minute_count,
        ROUND(AVG(traffic_density)::numeric, 4) AS avg_severe_density,
        ROUND(MAX(traffic_density)::numeric, 4) AS worst_density,
        SUM(vehicle_count) AS vehicles_during_severe,
        SUM(pedestrian_count) AS pedestrians_during_severe,
        CASE
            WHEN SUM(pedestrian_count) > 0 AND SUM(vehicle_count) > 0
                THEN 'Mixed-mode severe risk'
            WHEN SUM(pedestrian_count) > SUM(vehicle_count)
                THEN 'Pedestrian severe risk'
            ELSE 'Vehicle severe risk'
        END AS risk_category
    FROM severe_events
    GROUP BY camera_id
),
ranked_risk AS (
    SELECT
        camera_id,
        severe_minute_count,
        avg_severe_density,
        worst_density,
        vehicles_during_severe,
        pedestrians_during_severe,
        risk_category,
        RANK() OVER (ORDER BY worst_density DESC, severe_minute_count DESC) AS risk_rank,
        SUM(severe_minute_count) OVER () AS network_severe_minutes
    FROM camera_risk
)
SELECT
    risk_rank,
    camera_id,
    severe_minute_count,
    ROUND(100.0 * severe_minute_count / network_severe_minutes, 2) AS pct_of_severe_events,
    avg_severe_density,
    worst_density,
    vehicles_during_severe,
    pedestrians_during_severe,
    risk_category
FROM ranked_risk
ORDER BY risk_rank;
