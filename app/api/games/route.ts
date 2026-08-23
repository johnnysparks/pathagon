import { compactHumanGame, validateHumanGame } from "../../game-record";
import { storeHumanGame } from "../../../db/human-games";

export async function POST(request: Request) {
  try {
    const game = validateHumanGame(await request.json());
    const compact = compactHumanGame(game);
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(compact));
    const id = [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("").slice(0, 24);
    await storeHumanGame(id, game, compact);
    return Response.json({ accepted: true, id });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid game record";
    return Response.json({ accepted: false, error: message }, { status: 400 });
  }
}
