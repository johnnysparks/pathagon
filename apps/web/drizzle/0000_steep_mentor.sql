CREATE TABLE `human_games` (
	`id` text PRIMARY KEY NOT NULL,
	`schema_version` integer NOT NULL,
	`recorded_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL,
	`opponent_id` text NOT NULL,
	`winner` text NOT NULL,
	`plies` integer NOT NULL,
	`actions` text NOT NULL,
	`compact` text NOT NULL,
	`validation` text DEFAULT 'replay-valid' NOT NULL,
	`source` text DEFAULT 'web-human-v1' NOT NULL
);
--> statement-breakpoint
CREATE INDEX `human_games_recorded_at_idx` ON `human_games` (`recorded_at`);--> statement-breakpoint
CREATE INDEX `human_games_opponent_idx` ON `human_games` (`opponent_id`);