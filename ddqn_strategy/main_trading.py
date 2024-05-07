import os
import schedule

from stockstats import wrap, unwrap 

import time
import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta

from tinkoff.invest import AsyncClient, CandleInterval, HistoricCandle
from tinkoff.invest.utils import now, decimal_to_quotation, quotation_to_decimal
from tinkoff.invest import Client
from tinkoff.invest import MoneyValue
from tinkoff.invest.sandbox.client import SandboxClient
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX, INVEST_GRPC_API
from tinkoff.invest.grpc.orders_pb2 import (
    ORDER_DIRECTION_SELL,
    ORDER_DIRECTION_BUY,
    ORDER_TYPE_MARKET,
)

from uuid import uuid4

from decimal import Decimal

import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import json


class TinkoffInvestInteraction:
    def __init__(self, token, app_name, is_sandbox = True):
        self.token = token
        self.app_name = app_name
        self.is_sandbox = is_sandbox
        self.target = INVEST_GRPC_API_SANDBOX if is_sandbox else INVEST_GRPC_API

    def get_candles(self, figi:str, days_ago:int = 100):
        """ 
        Функция для получения свечей по инструменту
        """
        all_candles = {'Date': [], 'volume': [],'open': [],'close': [],'high': [],'low': []}

        with Client(self.token, target=self.target, app_name = self.app_name) as client:

            candles = client.get_all_candles(
                figi= figi,
                from_=now()-timedelta(days = days_ago),
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
                )
            
            instruments_service = client.instruments
            
            # Лотность инструмента
            lot = [i for i in instruments_service.shares().instruments if i.figi == figi][0].lot

            count = 0
            try:
                for candle in candles:
                    all_candles['Date'].append(candle.time)
                    all_candles['volume'].append(candle.volume)
                    all_candles['open'].append(self.cast_money(candle.open, lot))
                    all_candles['close'].append(self.cast_money(candle.close, lot))
                    all_candles['high'].append(self.cast_money(candle.high, lot))
                    all_candles['low'].append(self.cast_money(candle.low, lot))
                    #time.sleep(0.001)
                    count +=1

                    if count%1000 == 0:
                        time.sleep(1)

                        
            except Exception as e:
                # TODO: Сделать логи ошибок
                print('Что то пошло не так')
                print(e)
                return pd.DataFrame(all_candles)

            # print('Всё прошло успешно')
            return pd.DataFrame(all_candles)
    

    def cast_money(self, v, lot:int) -> float: 
        """
        Перевод в рубли

        :params 
            v: цена
            lot: лотность инструмента

        :return:
        """
        return float(quotation_to_decimal(v))*lot
    

    def get_info_accounts(self):
        """ 
        Получение всех аккаунтов песочницы
        """
        with Client(self.token, target=self.target, app_name = self.app_name) as client:
            sandbox_accounts = client.users.get_accounts()
            return sandbox_accounts


    def close_all_accounts(self):
        """ 
        Закрыть все аккаунты
        """
        with Client(self.token, target=self.target, app_name = self.app_name) as client:
            all_accounts = self.get_info_accounts()
            try:
                for sandbox_account in all_accounts.accounts:
                    client.sandbox.close_sandbox_account(account_id=sandbox_account.id)
            except Exception as e:
                print(f"Произошла ошибка: {e}")


    
    def create_new_account(self):
        with Client(self.token, target=self.target, app_name = self.app_name) as client:
            sandbox_account = client.sandbox.open_sandbox_account(name = "contest2024:YarickVodila/TinkoffRobotRL:1")
            # print(f"Account id: {sandbox_account.account_id}")
            return sandbox_account.account_id


    def add_money_sandbox(self, account_id, money, currency="rub"):
        """ 
        Пополнение баланса

        """
        with Client(self.token, target=self.target, app_name = self.app_name) as client:
            money = decimal_to_quotation(Decimal(money))
            return client.sandbox.sandbox_pay_in(
                account_id=account_id,
                amount=MoneyValue(units=money.units, nano=money.nano, currency=currency),
            )


    def get_ballans(self, account_id):
        with Client(self.token, target=self.target, app_name = self.app_name) as client:
            # print(client.operations.get_positions(account_id=account_id).money)
            balans = float(quotation_to_decimal(client.operations.get_positions(account_id=account_id).money[0]))
            return balans
        


    def get_portfolio_eval(self, account_id):
        with Client(self.token, target=self.target, app_name = self.app_name) as client:
            return float(quotation_to_decimal(client.operations.get_portfolio(account_id=account_id).total_amount_portfolio))
        


    def post_order(self, figi:str, type_action:int, quantity, account_id):
        """ 
        Выставление ордеров
        
        """
        with Client(self.token, target=self.target, app_name = self.app_name) as client:

            if type_action == 1:
                posted_order = client.orders.post_order(
                                order_id=str(uuid4()),
                                figi=figi,
                                direction=ORDER_DIRECTION_BUY,
                                quantity=int(quantity),
                                order_type=ORDER_TYPE_MARKET,
                                account_id=account_id,
                )
            else:
                posted_order = client.orders.post_order(
                                order_id=str(uuid4()),
                                figi=figi,
                                direction=ORDER_DIRECTION_SELL,
                                quantity=int(quantity),
                                order_type=ORDER_TYPE_MARKET,
                                account_id=account_id,
                )

            return posted_order

    # def get_trading_status(self, figi):
    #     with Client(TOKEN, target=self.target, app_name = self.app_name) as client:
    #         status = client.get



class TradingSystem:
    def __init__(self, model_path, config_path, logs_path, trading_statistic_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = torch.load(model_path, map_location=torch.device(self.device))
        
        self.config_path = config_path
        self.logs_path = logs_path
        self.trading_statistic_path = trading_statistic_path

        with open(self.config_path, "r") as read_file:
            self.config = json.load(read_file)


        try:
            self.trading_statistic = pd.read_csv(self.trading_statistic_path)
        except:
            self.trading_statistic = pd.DataFrame({"date":[], "portfolio_valuation":[], "last_price":[], "action": [], "quantity_stock": []})
            self.trading_statistic.to_csv(self.trading_statistic_path, index=False)


        self.figi = self.config['figi']
        self.trans_commission = self.config['trans_commission']

        self.interaction = TinkoffInvestInteraction(token=self.config["TOKEN"], app_name=self.config["app_name"], is_sandbox = True)
        
        if self.config["is_create_new_account"]==True:
            # Закрываем все аккаунты
            self.interaction.close_all_accounts()

            self.account_id = self.interaction.create_new_account()
            
            self.interaction.add_money_sandbox(self.account_id, self.config["start_balance"])

            self.config["account_id"] = self.account_id 
            self.config["is_create_new_account"] = False
            self.config["quantity_stock"] = 0
            self.config["is_open"] = False

            # with open(self.config_path, "w") as write_file:
            #     json.dump(self.config, write_file)
            self.save_config()

        else:
            self.account_id = self.config["account_id"]



    def get_features(self):

        # получаем свечи за 30 дней
        data = self.interaction.get_candles(self.figi, 30)
        data['day_of_week'] = data['Date'].dt.day_name()
        # Определение времени суток
        data['time_of_day'] = pd.cut(data['Date'].dt.hour, bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
        data = pd.get_dummies(data, columns=['day_of_week', "time_of_day"], drop_first= True, dtype = int)

        # Добавляем индикаторы
        data = self.add_indicators(data)

        # Берём последние 10 свечей
        data = data.iloc[-11:-1]

        print(data[["close", 'macd', 'macds', 'macdh', 'rsi_12']])
        
        # Цена потенциальной покупки
        last_price = data.iloc[-1]['close']

        # print(data.shape)
        # display(data)

        # Стандартизируем данные в вектор
        standartizer_data = self.standartizer(data.to_numpy())

        # print(standartizer_data.shape)

        return standartizer_data, last_price


    def add_indicators(self, data):
        df = data.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        # Получаем индикаторы
        df_indicators = wrap(df.copy())
        df_indicators = df_indicators[['macd', 'macds', 'macdh', 'rsi_12']] 
        df_indicators = unwrap(df_indicators)

        df = df.set_index('Date')
        data = df.merge(df_indicators, left_index=True, right_index=True).copy()

        data.dropna(inplace= True)
        # Сортируем индексы и переводить в tz_localize
        data.sort_index(inplace=True)
        data.index = data.index.tz_localize(None)
        data.index = pd.to_datetime(data.index)
        return data
    


    def standartizer(self, array):
        """
        Метод для стандартизации данных
        
        Столбцы 0: Объём торгов 
        Столбцы 1 - 5: Свечи
        Столбцы 5 - 14: Время дня и прочие данные по времени
        Столбцы 14 - 18: Индикаторы

        """
        valume = array[:, 0].copy()
        array_candle = array[:, 1:5].copy()
        time_date = array[:, 5:14].copy()
        indicators = array[:, 14:18].copy()
        
        # Стандартизируем объёмы
        array_candle = (array_candle - array_candle.min()) / (array_candle.max() - array_candle.min())
        valume = (valume - valume.min()) / (valume.max() - valume.min())

        array = np.c_[indicators, array_candle]
        array = np.c_[array, valume]
        array = np.c_[array, time_date]

        return array.reshape(-1)


    def predict_action(self):
        
        # Получаем фичи
        data, last_price = self.get_features()

        # Делаем предсказание
        action = self.model(torch.tensor(data).to(self.device).float()).argmax().item()

        # print(pred)
        return action, last_price
    
    def save_config(self):
        """ 
        Метод для сохранения конфига
        """
        with open(self.config_path, "w") as write_file:
            json.dump(self.config, write_file)


    def select_action(self):
        """ 
        Метод для исполнения действий (покупки, продажи или удержания)
        """

        current_time = datetime.datetime.now()
        # Получение текущего времени
        date = current_time.strftime("%Y/%m/%d %H:%M:%S")

        # Оценка портфеля
        portfolio_valuation = self.interaction.get_portfolio_eval(self.account_id)

        action, last_price = self.predict_action()
        # action = "buy" if action == 1 else "sell"

        is_open = self.config['is_open']
        trans_commission = self.config['trans_commission']

        try:
            # Если в позицию ранее не входили и пропускаем
            if action == 0 and is_open == False:
                print(f"{date} - в позицию ранее не входили и пропускаем")
                quantity_stock = 0
                df = pd.DataFrame({"date":[date], "portfolio_valuation":[portfolio_valuation], "last_price":[last_price], "action": [action], "quantity_stock": [quantity_stock]})

                self.trading_statistic = pd.concat([self.trading_statistic, df], ignore_index=True)
                self.trading_statistic.to_csv(self.trading_statistic_path, index=False)

            
            # Если в позицию ранее входили и продаём
            elif action == 0 and is_open == True:
                print(f"{date} - в позицию ранее входили и продаём")

                quantity_stock = self.config['quantity_stock']

                self.interaction.post_order(
                    figi = self.figi,
                    type_action = action,
                    quantity = quantity_stock,
                    account_id =  self.account_id
                )

                df = pd.DataFrame({"date":[date], "portfolio_valuation":[portfolio_valuation], "last_price":[last_price], "action": [action], "quantity_stock": [quantity_stock]})

                self.trading_statistic = pd.concat([self.trading_statistic, df], ignore_index=True)
                self.trading_statistic.to_csv(self.trading_statistic_path, index=False)

                self.config['is_open'] = False
                self.config['quantity_stock'] = 0
                self.save_config()
            
            # Если в позицию ранее не входили и покупаем
            elif action == 1 and is_open == False:
                print(f"{date} - в позицию ранее не входили и покупаем")

                balance = self.interaction.get_ballans(self.account_id)
                quantity_stock = int(balance // (last_price * (1 + trans_commission)))

                self.interaction.post_order(
                    figi = self.figi,
                    type_action = action,
                    quantity = quantity_stock,
                    account_id =  self.account_id
                )

                df = pd.DataFrame({"date":[date], "portfolio_valuation":[portfolio_valuation], "last_price":[last_price], "action": [action], "quantity_stock": [quantity_stock]})

                self.trading_statistic = pd.concat([self.trading_statistic, df], ignore_index=True)
                self.trading_statistic.to_csv(self.trading_statistic_path, index=False)

                self.config['is_open'] = True
                self.config['quantity_stock'] = quantity_stock
                self.save_config()


            # Если в позицию ранее входили и держим
            elif action == 1 and is_open == True:
                print(f"{date} - в позицию ранее входили и держим")

                quantity_stock = self.config['quantity_stock']

                df = pd.DataFrame({"date":[date], "portfolio_valuation":[portfolio_valuation], "last_price":[last_price], "action": [action], "quantity_stock": [quantity_stock]})

                self.trading_statistic = pd.concat([self.trading_statistic, df], ignore_index=True)
                self.trading_statistic.to_csv(self.trading_statistic_path, index=False)

        except Exception as e:
            print("Произошла ошибка!\n", e)


def main():
    model_path = "models\\best_model.pth"
    config_path = "config.json"
    logs_path = "ddqn_stratregy"
    trading_statistic_path = "trading_statistic.csv"

    system = TradingSystem(model_path, config_path, logs_path, trading_statistic_path)

    # Запускать каждый час на 4 минуте
    schedule.every().hour.at(":04").do(system.select_action)
    # schedule.every().minute.at(":50").do(system.select_action)
    while True:
        schedule.run_pending()
        time.sleep(60)
        

main()