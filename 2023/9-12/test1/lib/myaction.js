const inqurier = require('inquirer');
var config = require('../config');
const downloadFun = require('./download')
const myaction = async function(project, args) {
    // console.log(project);
    // console.log(args);
    const asnwers = await inqurier.prompt([{
        type: "list",
        name: "framework",
        message: "请选择你所需要的框架:\n",
        choices: config.framework
    }])
    console.log(asnwers);
    // 下载模板
    downloadFun(config.frameworkUrl[asnwers.framework], project)
}

module.exports = myaction;