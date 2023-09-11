var fs = require("fs");
// console.log(fs);
fs.writeFile('./a.txt', '666', function(error) {
    console.log(error);
})