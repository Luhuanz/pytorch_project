"""
 * Project name(项目名称)：Python_os_path模块
 * Package(包名): 
 * File(文件名): test1
 * Author(作者）: mao
 * Author QQ：1296193245
 * GitHub：https://github.com/maomao124/
 * Date(创建日期)： 2022/3/30 
 * Time(创建时间)： 20:45
 * Version(版本): 1.0
 * Description(描述)：
 方法	                说明
 os.path.abspath(path)	返回 path 的绝对路径。
 os.path.basename(path)	获取 path 路径的基本名称，即 path 末尾到最后一个斜杠的位置之间的字符串。
 os.path.commonprefix(list)	返回 list（多个路径）中，所有 path 共有的最长的路径。
 os.path.dirname(path)	返回 path 路径中的目录部分。
 os.path.exists(path)	判断 path 对应的文件是否存在，如果存在，返回 True；反之，
                        返回 False。和 lexists() 的区别在于，exists()会自动判断失效的文件链接（类似 Windows 系统中文件的快捷方式），
                        而 lexists() 却不会。
 os.path.lexists(path)	判断路径是否存在，如果存在，则返回 True；反之，返回 False。
 os.path.expanduser(path)	把 path 中包含的 "~" 和 "~user" 转换成用户目录。
 os.path.expandvars(path)	根据环境变量的值替换 path 中包含的 "$name" 和 "${name}"。
 os.path.getatime(path)	返回 path 所指文件的最近访问时间（浮点型秒数）。
 os.path.getmtime(path)	返回文件的最近修改时间（单位为秒）。
 os.path.getctime(path)	返回文件的创建时间（单位为秒，自 1970 年 1 月 1 日起（又称 Unix 时间））。
 os.path.getsize(path)	返回文件大小，如果文件不存在就返回错误。
 os.path.isabs(path)	判断是否为绝对路径。
 os.path.isfile(path)	判断路径是否为文件。
 os.path.isdir(path)	判断路径是否为目录。
 os.path.islink(path)	判断路径是否为链接文件（类似 Windows 系统中的快捷方式）。
 os.path.ismount(path)	判断路径是否为挂载点。
 os.path.join(path1[, path2[, ...]])	把目录和文件名合成一个路径。
 os.path.normcase(path)	转换 path 的大小写和斜杠。
 os.path.normpath(path)	规范 path 字符串形式。
 os.path.realpath(path)	返回 path 的真实路径。
 os.path.relpath(path[, start])	从 start 开始计算相对路径。
 os.path.samefile(path1, path2)	判断目录或文件是否相同。
 os.path.sameopenfile(fp1, fp2)	判断 fp1 和 fp2 是否指向同一文件。
 os.path.samestat(stat1, stat2)	判断 stat1 和 stat2 是否指向同一个文件。
 os.path.split(path)	把路径分割成 dirname 和 basename，返回一个元组。
 os.path.splitdrive(path)	一般用在 windows 下，返回驱动器名和路径组成的元组。
 os.path.splitext(path)	分割路径，返回路径名和文件扩展名的元组。
 os.path.splitunc(path)	把路径分割为加载点与文件。
 os.path.walk(path, visit, arg)	遍历path，进入每个目录都调用 visit 函数，
        visit 函数必须有 3 个参数(arg, dirname, names)，dirname 表示当前目录的目录名，
        names 代表当前目录下的所有文件名，args 则为 walk 的第三个参数。
 os.path.supports_unicode_filenames	设置是否可以将任意 Unicode 字符串用作文件名。
  """
 import os

if __name__ == '__main__':
    path = "D:\\files\\file.txt"
    print(path)
    print(os.path.abspath("."))
    print(os.path.basename(path))
    print(os.path.dirname(path))
    print(os.path.exists(path))
    print(os.path.expanduser("~"))
    print(os.path.expandvars("${JAVA_HOME}"))
    path = "."
    print(os.path.getatime(path))
    print(os.path.getmtime(path))
    print(os.path.getctime(path))
    path = ".\\test1.py"
    print(os.path.getsize(path))
    print(os.path.isabs(path))
    print(os.path.isfile(path))
    print(os.path.isdir(path))
    print(os.path.islink(path))
    print(os.path.ismount(path))
    print(os.path.normcase(path))
    print(os.path.realpath(path))
    print(os.path.split(os.path.realpath(path)))
    print(os.path.splitdrive(os.path.realpath(path)))
    print(os.path.splitext(os.path.realpath(path)))