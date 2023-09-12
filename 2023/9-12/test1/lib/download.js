const donwload = require('download-git-repo')
const chalk = require('chalk')
const ora = require('ora')

const downloadFun = (url, project) => {
    const spinner = ora().start()
    spinner.text = "coding downloading...";
    donwload('direct:' + url, project, { clone: true }, function(err) {
        if (!err) {
            spinner.succeed(chalk.blue(" downloading  successfully!"));
        } else {
            spinner.fail("err~");
        }
    })
}
module.exports = downloadFun