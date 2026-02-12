-- Fix historical games: set status='final' where scores exist but status='scheduled'
UPDATE games 
SET status = 'final' 
WHERE status = 'scheduled' 
  AND home_score IS NOT NULL 
  AND away_score IS NOT NULL;