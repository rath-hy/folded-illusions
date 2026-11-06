// Run Flat-Folder's solver without its web page, and write the same cp/state FOLD files as its
// Export button.
//
//   git clone https://github.com/origamimagiro/flat-folder.git
//   node ff_solve.mjs path/to/flat-folder model.fold out_prefix [state]
//
// The model can be any format Flat-Folder imports by extension (.fold, .cp: one crease per line,
// "type x1 y1 x2 y2" with type 1 border, 2 mountain, 3 valley, 5 unassigned).
//
// Writes out_prefix_cp.fold and out_prefix_state.fold. `state` picks one of the valid folded
// states by its index in each independent component (comma-separated, default all 0s, which is
// the state Flat-Folder shows first), or "all".
import * as fs from "fs";
import * as path from "path";
import { pathToFileURL } from "url";

const [ffDir, input, prefix, stateArg] = process.argv.slice(2);
if (!prefix) {
    console.error("usage: node ff_solve.mjs FLAT_FOLDER_DIR MODEL.fold OUT_PREFIX [STATE]");
    process.exit(1);
}

// compute.js is a web worker: it reads requests from onmessage and answers with postMessage.
// Stand in for the page by calling its handlers directly and keeping the last answer.
let reply;
globalThis.postMessage = (m) => { if (m.type === "end") reply = m.arg; };
globalThis.onmessage = null;
const src = (f) => pathToFileURL(path.resolve(ffDir, "src", f)).href;
const { NOTE } = await import(src("note.js"));
const { X } = await import(src("conversion.js"));
await import(src("compute.js"));
NOTE.show = false;
const call = (type, ...args) => { reply = undefined; globalThis.onmessage({ data: { type, args } }); return reply; };

call("setup");
const FOLD = call("doc_type_side_2_fold", fs.readFileSync(input, "utf8"),
                  path.extname(input).slice(1).toLowerCase(), true);
call("get_cell");
call("build_variables");
call("build_constraints");
const [ok, err] = call("presolve");
if (ok !== "ok") throw new Error(`not flat-foldable: ${JSON.stringify(err)}`);
const [status, Gn] = call("solve", Infinity);
if (status !== "success") throw new Error(`no folded state: ${JSON.stringify(Gn)}`);
// "all" writes every state (out_prefix_state_0.fold, _1, ...), for comparing them.
const total = Gn.reduce((s, n) => s * n, 1);
const picks = stateArg === "all"
    ? Array.from({ length: total }, (_, k) => Gn.map((n, i) => Math.floor(k / Gn.slice(0, i).reduce((s, m) => s * m, 1)) % n))
    : [stateArg ? stateArg.split(",").map(Number) : Gn.map(() => 0)];
console.log(`${path.basename(input)}: ${total} folded states (components |${Gn.join("|")}|)`);

const { V, Vf, EV, EA, FV } = FOLD;
const name = path.basename(prefix);
const cp = { file_spec: 1.1, file_creator: "flat-folder", file_title: `${name}_cp`,
             file_classes: ["singleModel"], vertices_coords: V, edges_vertices: EV,
             edges_assignment: EA, faces_vertices: FV };
fs.writeFileSync(`${prefix}_cp.fold`, JSON.stringify(cp, undefined, 2));
picks.forEach((Gi, k) => {
    const [, FO] = call("Gi_2_CD_FO", Gi);
    const out = stateArg === "all" ? `${prefix}_state_${k}.fold` : `${prefix}_state.fold`;
    fs.writeFileSync(out, JSON.stringify(
        { ...cp, file_title: `${name}_state`, vertices_coords: Vf, faceOrders: FO }, undefined, 2));
});
