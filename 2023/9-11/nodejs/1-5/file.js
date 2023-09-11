// var fs = require('fs');
// fs.readFile('a.txt', 'utf8', function(err, data) {
//             if (!err) {
//                 var newdata = data + '8888';
//                 fs.writeFile('a.txt', newdata, function(err) {
//                     if (!err) {
//                         console.log("追加成功！")；
//                     }
//                 });
//             }
//         }

var fs = require('fs');
fs.readFile('a.txt', 'utf8', function(err, data) {
    if (!err) {
        var newdata = data + "8888";
        fs.writeFile('a.txt', newdata, function(err) {
            if (!err) {
                console.log("追加成功！");
            }
        })
    }

})