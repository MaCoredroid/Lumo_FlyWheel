import { cpSync, mkdirSync, rmSync } from "node:fs";

rmSync(new URL("../dist", import.meta.url), { recursive: true, force: true });
mkdirSync(new URL("../dist/src", import.meta.url), { recursive: true });
mkdirSync(new URL("../dist/plugins", import.meta.url), { recursive: true });
cpSync(new URL("../src", import.meta.url), new URL("../dist/src", import.meta.url), { recursive: true });
cpSync(new URL("../plugins", import.meta.url), new URL("../dist/plugins", import.meta.url), { recursive: true });
