import { compactHumanGame, createGameId, validateGameId, validateHumanGame } from "../../game-record";
import { storeHumanGame } from "../../../db/human-games";

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    if (!payload || typeof payload !== "object") throw new Error("Game record must be an object");
    const record = payload as Record<string, unknown>;
    let id: string;
    if (record.id === undefined) {
      id = createGameId();
    } else {
      validateGameId(record.id);
      id = record.id;
    }
    const game = validateHumanGame(payload);
    const compact = compactHumanGame(game);
    await storeHumanGame(id, game, compact);
    return Response.json({ accepted: true, id });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid game record";
    return Response.json({ accepted: false, error: message }, { status: 400 });
  }
}
