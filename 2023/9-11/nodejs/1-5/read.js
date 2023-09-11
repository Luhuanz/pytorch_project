var fs = require('fs');
// 读取
fs.readFile('./a.txt', 'utf8', function(err, data) {
    console.log(err);
    console.log(data);
})