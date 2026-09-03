import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { OPPONENTS } from "../app/opponents.ts";

const promotedArtifacts = [
  {
    opponent: "Tile Driver",
    dataPath: "../../../data/models/tile-driver-gnn-policy-value-v1/pathagon-gnn-policy-value.onnx",
    dataManifestPath: "../../../data/models/tile-driver-gnn-policy-value-v1/manifest.json",
    publicPath: "../public/models/pathagon-gnn-policy-value.onnx",
    publicManifestPath: "../public/models/pathagon-gnn-policy-value.manifest.json",
  },
  {
    opponent: "Double Dragon",
    dataPath: "../../../data/models/double-dragon-gnn-qadv-v1/pathagon-gnn-qadv.onnx",
    dataManifestPath: "../../../data/models/double-dragon-gnn-qadv-v1/manifest.json",
    publicPath: "../public/models/pathagon-gnn-qadv.onnx",
    publicManifestPath: "../public/models/pathagon-gnn-qadv.manifest.json",
  },
  {
    opponent: "Yann Tileson",
    dataPath: "../../../data/models/yann-tileson-jepa-afterstate-v1/pathagon-jepa-afterstate.onnx",
    dataManifestPath: "../../../data/models/yann-tileson-jepa-afterstate-v1/manifest.json",
    publicPath: "../public/models/pathagon-jepa-afterstate.onnx",
    publicManifestPath: "../public/models/pathagon-jepa-afterstate.manifest.json",
  },
];

async function bytes(relativePath: string) {
  return readFile(new URL(relativePath, import.meta.url));
}

test("all six opponent cards are playable and learned artifacts are real manifest-backed files", async () => {
  assert.deepEqual(
    OPPONENTS.map((opponent) => opponent.name),
    ["Pathman", "Tile Driver", "Seer", "Double Dragon", "Yann Tileson", "Rando Raccon"],
  );
  assert.ok(OPPONENTS.every((opponent) => opponent.playable));
  for (const artifact of promotedArtifacts) {
    const [data, dataManifest, publicModel, publicManifest] = await Promise.all([
      bytes(artifact.dataPath),
      bytes(artifact.dataManifestPath),
      bytes(artifact.publicPath),
      bytes(artifact.publicManifestPath),
    ]);
    const dataHash = `sha256:${createHash("sha256").update(data).digest("hex")}`;
    const publicHash = `sha256:${createHash("sha256").update(publicModel).digest("hex")}`;
    const manifest = JSON.parse(dataManifest.toString()) as { artifactSha256: string };
    const publicManifestJson = JSON.parse(publicManifest.toString()) as { artifactHash: string };
    assert.ok(data.length > 10_000, `${artifact.opponent} data artifact is unexpectedly small`);
    assert.equal(dataHash, manifest.artifactSha256, `${artifact.opponent} data hash mismatch`);
    assert.equal(publicHash, publicManifestJson.artifactHash, `${artifact.opponent} public hash mismatch`);
    assert.equal(dataHash, publicHash, `${artifact.opponent} public artifact differs from promoted data`);
  }
});
