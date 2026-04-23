
import test from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

test("prints help text", async () => {
  const { stdout } = await execFileAsync("node", ["src/index.mjs", "--help"]);
  assert.match(stdout, /Usage:/);
  assert.match(stdout, /--list/);
});
