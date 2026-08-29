import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const databaseId = process.argv[2];

if (!databaseId || !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(databaseId)) {
  throw new Error("Expected a D1 database UUID as the first argument");
}

const configPath = fileURLToPath(new URL("../wrangler.jsonc", import.meta.url));
const config = JSON.parse(await readFile(configPath, "utf8"));
const bindings = config.d1_databases?.filter(
  (binding) => binding.binding === "DB" && binding.database_name === "pathagon-web",
) ?? [];

if (bindings.length !== 1) {
  throw new Error(`Expected exactly one pathagon-web DB binding, found ${bindings.length}`);
}

bindings[0].database_id = databaseId;
await writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`);
