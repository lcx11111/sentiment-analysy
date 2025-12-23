#使用DrissionPage爬取京东评论
from DrissionPage import ChromiumPage
import csv
f=open('data/data.csv', mode='w', encoding='utf-8', newline='')
csv_writer=csv.DictWriter(f, fieldnames=[
        '买家名称',
        '评论',
        '日期',
        '分数',
        '产品',
   ])
csv_writer.writeheader()
dp = ChromiumPage()
dp.get('https://item.jd.com/100055310629.html')

dp.listen.start('client.action')

dp.ele('css:.all-btn').click()


for page in range(1,50):
    print("正在采集{}".format(page))
    resp=dp.listen.wait(2)
    json_data=resp[1].response.body
    print(json_data)
    #提取评论
    datas=json_data['result']['floors'][2]['data']
    for index in datas:
        keys=index.keys()
        if 'commentInfo' in keys:
            dit={
                '买家名称':index['commentInfo']['userNickName'],
                '评论':index['commentInfo']['commentData'],
                '日期':index['commentInfo']['commentDate'],
                '分数':index['commentInfo']['commentScore'],
                '产品':index['commentInfo']['productSpecifications'],
            }
            csv_writer.writerow(dit)
            print(dit)

    else:
        pass



    k = dp.ele('css:._rateListContainer_1ygkr_45')
    k.scroll.to_bottom()

