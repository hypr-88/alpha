import requests
from tardis_dev import datasets
from datetime import datetime, timedelta
from dateutil import parser
from multiprocessing import Pool
import pytz
import logging

utc = pytz.UTC

# comment out to disable debug log

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


exchanges = ["bitmex", "binance-futures", "ftx", "kraken", "coinbase", "okex"]
API_KEY = "TD.19wIOfpOFIJf8FBf.TCeI95B3pEnYl2h.boRm52ApcBX1TWR.kv4WeDT67QPopoP.769wRDdUdIy3OE7.f1MH"
start_date = utc.localize(parser.parse("2021-07-22"))
today = utc.localize(datetime.today())


def crawl(exchange: str):
    print("Starting crawl exchange: " + exchange)
    r = requests.get("https://api.tardis.dev/v1/exchanges/{}".format(exchange))
    response = r.json()

    for symbol in response["datasets"]["symbols"]:
        if "availableTo" in symbol:
            to_date = parser.parse(symbol["availableTo"])
        else:
            to_date = today

        from_date = parser.parse(symbol["availableSince"])

        if from_date < start_date:
            from_date = start_date

        if from_date > to_date:
            continue

        to_date_str = to_date.strftime('%Y-%m-%d')
        from_date_str = from_date.strftime('%Y-%m-%d')
        try:
            datasets.download(
                exchange=exchange,
                data_types=symbol["dataTypes"],
                from_date=from_date_str,
                to_date=to_date_str,
                symbols=[symbol["id"]],
                api_key=API_KEY,
                get_filename=file_name_nested,
                download_dir="./datasets",
                concurrency=20,
            )
        except Exception as e:
            logger.error(e)

if __name__ == '__main__':
    pool = Pool(processes=len(exchanges))
    pool.map(crawl, exchanges)
    pool.close()
    pool.join()
    # for i in exchanges:
    #     crawl(i)
    # print(parser.parse("2021-07-07"))
