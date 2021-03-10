from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime

def injectNotificationDataSet(device,image_url,time,area,stream_url,nomask,allp):
	client = bigquery.Client(project='chatbot-108aea001-296006',credentials=service_account.Credentials.from_service_account_file('2020chatbot-108AEA001-7234299f4f96.json'))
	tableId = 'chatbot-108aea001-296006.warning_alert.hama114514'
	model = [{
		'device': device,
		'area':area,
		'all':allp,
		'nomask':nomask,
		'image_url': image_url,
		'stream_url':stream_url,
		'time':time
	}] 
	client.insert_rows_json(tableId, model)