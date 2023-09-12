#!/usr/bin/env Node
 // console.log("hello node");
const { program } = require('commander');
const myhelp = require("../lib/myhelp");
myhelp(program);
const mycommander = require("../lib/mycommander");
mycommander(program);

program.parse(process.argv);