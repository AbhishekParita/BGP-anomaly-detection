-- Add missing ensemble_is_anomaly column to ml_results table
ALTER TABLE ml_results 
ADD COLUMN IF NOT EXISTS ensemble_is_anomaly BOOLEAN DEFAULT FALSE;

-- Create index for faster anomaly queries
CREATE INDEX IF NOT EXISTS idx_ml_results_ensemble_anomaly 
ON ml_results(ensemble_is_anomaly, timestamp DESC);

-- Update existing records where ensemble_score > 0.4 to mark as anomaly
UPDATE ml_results 
SET ensemble_is_anomaly = TRUE 
WHERE ensemble_score > 0.4 
AND ensemble_is_anomaly = FALSE;
