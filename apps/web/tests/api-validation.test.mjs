import assert from "node:assert/strict";
import test from "node:test";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
const { default: worker } = await import(workerUrl.href);

function appFetch(path, init) {
  return worker.fetch(
    new Request(`http://localhost${path}`, init),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

async function responseBody(response) {
  return await response.json();
}

test("self-play upload rejects malformed payloads before database access", async () => {
  const response = await appFetch("/api/selfplay", {
    method: "POST",
    body: JSON.stringify({}),
    headers: { "content-type": "application/json" },
  });
  assert.equal(response.status, 400);
  assert.deepEqual(await responseBody(response), {
    accepted: false,
    error: "Self-play upload must contain a games array",
  });
});

test("self-play query validates pagination and record IDs at the HTTP boundary", async () => {
  const queryResponse = await appFetch("/api/selfplay?limit=not-a-number");
  assert.equal(queryResponse.status, 400);
  assert.equal((await responseBody(queryResponse)).error, "Invalid pagination value");

  const recordResponse = await appFetch("/api/selfplay/bad%20id");
  assert.equal(recordResponse.status, 400);
  assert.equal((await responseBody(recordResponse)).error, "Invalid self-play record ID");
});

test("human-game upload rejects invalid records before database access", async () => {
  const response = await appFetch("/api/games", {
    method: "POST",
    body: JSON.stringify({ opponentId: "surveyor-v0", winner: "light", actions: [] }),
    headers: { "content-type": "application/json" },
  });
  assert.equal(response.status, 400);
  assert.equal((await responseBody(response)).error, "Invalid action count");
});

test("cross-play requires an explicit run or aggregate selector", async () => {
  const response = await appFetch("/api/cross-play");
  assert.equal(response.status, 400);
  assert.equal((await responseBody(response)).error, "A cross-play run ID is required");
});
