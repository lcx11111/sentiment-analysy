from DrissionPage import ChromiumOptions
#配置DrissionPage ，google
path = r'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
ChromiumOptions().set_browser_path(path).save()