import json

import re
import requests
import json
import time
from model import ProductItemBO
import datetime
    
from redis_process import update_redis
from logger import recorder

data_created_date = datetime.datetime.now()
site_id =83
crawler = "hasaki.vn"
max_page = 100
from bs4 import BeautifulSoup


def get_data(crawlurl):
    all_results = []
    for i in range(1,100):
        for_result = []
        page_url = f"{crawlurl}?p={i}"

        payload={}
        headers = {
        'Cookie': 'HASAKI_SESSID=897840dcf3f6615d2fc44f95ebb38670; PHPSESSID=48pokk0bbdt25t5fcscl9fn304; form_key=897840dcf3f6615d2fc44f95ebb38670; sessionChecked=1645516716'
        }

        response = requests.request("GET", url, headers=headers, data=payload)

        content = response.text

        bs = BeautifulSoup(content, "html.parser")


        data_content = bs.findAll('div', attrs={'class': 'ProductGridItem__itemOuter'})
       
		for data in data_content:
			try:
				product_name = data.find('h2').find('div' , attrs={'class':'vn_names'}).text.strip()#<Lấy tên sản phẩm>
				parse_product_name = product_name.split('\n')
				s = []
				for val in parse_product_name:
					if val == '':
						continue
					val = val.strip('\r ')
					s.append(val)
					
				product_name = ' '.join(s)
			except:
				product_name = 'NoName'

			try:
				product_link = "https://hasaki.vn/san-pham/sp-"+data.find('a')[-1].data('product')#<Lấy id>
			except:
				product_link = ''

			""" Lấy trạng thái sản phẩm """
			try:
				product_status = data.find('figure', class_='product_image').find('div', class_='text-sold-out').text.strip()
				if 'Hết' in product_status:
					product_status = f"hết hàng"
				else:
					product_status = f"còn hàng"
			except Exception as ex:
				product_status = f"còn hàng"
			

			""" - Lấy giá gốc và giá khuyến mại -"""
			try:
                price = data.find('span', class_='item_giacu').text#<Lấy giá gốc của dhbt>
                price = utils.getNumber(price)
                promotional_price = data.find('strong', class_='item_giamoi').text
            except Exception as ex:
                price = '0'
                promotional_price = '0'
			
			product_item = ProductItemBO(product_name=str(product_name), product_price=str(promotional_price), product_link=str(product_link), product_status=product_status, product_realprice=str(price))

			for_result.append(product_item)
        if len(for_result)==0:
            break

        all_results.extend(for_result)
	return all_results



url = "http://10.1.12.58:3041/get-category"

payload = json.dumps({
  "site_id": site_id
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)

list_url = json.loads(response.text)

import datetime




for itemurl in list_url:
    url = itemurl["category_link"]
    result = get_data(itemurl["category_link"])
    len_lazada += len(result)
    recorder(logger_name=crawler).InfoLog(f"Crawler category{url} total item: {str(len(result))}")
    update_redis(result,site_id,data_created_date,"https://hasaki.vn")

    url = "http://10.1.12.58:3041/log-category"

    payload = json.dumps({
    "category_id": itemurl["category_id"],
    "created_date": datetime.datetime.now().isoformat(),
    "crawler_total": len(result)
    }, default=str)
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    time.sleep(2)