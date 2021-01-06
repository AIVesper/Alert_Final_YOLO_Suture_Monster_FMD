import json
import requests

from flask import Flask, request
from linebot import WebhookHandler, LineBotApi
from linebot.exceptions import InvalidSignatureError
from linebot.models import FollowEvent, TextSendMessage, MessageEvent, TextMessage


channel_secret = "3d283e43e4c970ceffd66d64f48db84e"
channel_access_token = "a2IZVQTYiYY3j/3impVPSehSjtmtt6g+m9KqxcnSLvIUvhVoElemRXLt7FRoaoBq2rINjjoOGMzYXktM/21/xYnZskxNFAj2oFUV3mLKa+B6i5Brd+lI6/49dJEmz0kBfuL+jRTft81WLx6gx0EBKAdB04t89/1O/w1cDnyilFU="

handler = WebhookHandler(channel_secret)
linebotApi = LineBotApi(channel_access_token)


# web server for debug
app = Flask(__name__)

@app.route("/")
def index():
    return "this is index"


def replyMessageToUser(replyToken, texts):
    replyMessages = []
    for text in texts:
        replyMessages.append(TextSendMessage(text=text))
    linebotApi.reply_message(replyToken, replyMessages)


def pushMessageToUser(lineId, texts):
    pushMessages = []
    for text in texts:
        pushMessages.append(TextSendMessage(text=text))
    linebotApi.push_message(lineId, pushMessages)


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message_text = event.message.text
    print("[MESG]%s"%(message_text))

    #test reply
    reply_text = "[回复]收到信息：%s"%(message_text)
    texts = []
    texts.append(reply_text)
    replyMessageToUser(event.reply_token, texts)

    #test push
    lineId=event.source.user_id
    push_text = "[推送]收到信息：%s"%(message_text)
    texts = []
    texts.append(push_text)
    pushMessageToUser(lineId,texts)


# for flask
@app.route("/webhook",methods=['GET', 'POST'])
def lineWebhook():
# for Line Message API
# def lineWebhook(request):

    # get X-Line-Signature header value
    signature = request.headers.get('X-Line-Signature')

    # get request body as text
    body = request.get_data(as_text=True)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return '200 OK'


if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0")