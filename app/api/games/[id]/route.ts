import { getHumanGame } from "../../../../db/human-games";
import { validateGameId } from "../../../game-record";

type RouteContext = { params: Promise<{ id: string }> };

export async function GET(_request: Request, context: RouteContext) {
  try {
    const { id } = await context.params;
    validateGameId(id);
    const game = await getHumanGame(id);
    if (!game) return Response.json({ found: false, error: "Game not found" }, { status: 404 });
    return Response.json({ found: true, game }, { headers: { "cache-control": "private, no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to read game record";
    const status = message === "Invalid game ID" ? 400 : 500;
    return Response.json({ found: false, error: message }, { status });
  }
}
