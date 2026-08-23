CREATE TABLE `selfplay_games` (
	`id` text PRIMARY KEY NOT NULL,
	`schema_version` integer NOT NULL,
	`recorded_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL,
	`engine` text NOT NULL,
	`mode` text NOT NULL,
	`run_id` text,
	`seed` integer NOT NULL,
	`light_agent` text NOT NULL,
	`dark_agent` text NOT NULL,
	`winner` text,
	`result` text NOT NULL,
	`reason` text NOT NULL,
	`plies` integer NOT NULL,
	`record` text NOT NULL,
	`source` text DEFAULT 'selfplay-v1' NOT NULL
);
--> statement-breakpoint
CREATE INDEX `selfplay_games_recorded_at_idx` ON `selfplay_games` (`recorded_at`);
--> statement-breakpoint
CREATE INDEX `selfplay_games_engine_mode_idx` ON `selfplay_games` (`engine`,`mode`);
--> statement-breakpoint
CREATE INDEX `selfplay_games_agents_idx` ON `selfplay_games` (`light_agent`,`dark_agent`);
--> statement-breakpoint
CREATE INDEX `selfplay_games_result_idx` ON `selfplay_games` (`result`,`winner`);
