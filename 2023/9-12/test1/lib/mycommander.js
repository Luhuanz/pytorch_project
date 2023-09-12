const myaction = require('./myaction');
const mycommander = function(program) {
    program.
    command("Create <project> [other]")
        .description("创建项目")
        .alias("crt")
        .action(myaction)
}
module.exports = mycommander;