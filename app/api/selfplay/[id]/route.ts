import { getSelfPlayGame } from "../../../../db/selfplay-games";

const ID_PATTERN = /^[a-zA-Z0-9._:-]{1,160}$/;
type RouteContext = { params: Promise<{ id: string }> };

export async function GET(_request: Request, context: RouteContext) {
  try {
    const { id } = await context.params;
    if (!ID_PATTERN.test(id)) return Response.json({ found: false, error: "Invalid self-play record ID" }, { status: 400 });
    const game = await getSelfPlayGame(id);
    if (!game) return Response.json({ found: false, error: "Self-play record not found" }, { status: 404 });
    return Response.json({ found: true, game }, { headers: { "cache-control": "private, no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to read self-play record";
    return Response.json({ found: false, error: message }, { status: 500 });
  }
}
