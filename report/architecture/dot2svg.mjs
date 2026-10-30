import { Graphviz } from "@hpcc-js/wasm-graphviz";
import fs from "fs";
const gv = await Graphviz.load();
const dot = fs.readFileSync(process.argv[2], "utf8");
fs.writeFileSync(process.argv[3], gv.dot(dot, "svg"));
console.log("svg written", process.argv[3]);
