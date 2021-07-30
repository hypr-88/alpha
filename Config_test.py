# test_with_unittest.py

from unittest import TestCase

from Config import deriveFeature, basicFeature
from PreFetch import getData

class ConfigTesting(TestCase):
    data = getData()

    def testDeriveConfig(self):
        derive = [
            {
                "type": "EMA",
                "window": 5
            },
            {
                "type": "MACD",
                "fastPeriod": 5,
                "slowPeriod": 10,
                "signalPeriod": 6
            },
            {
                "type": "BBANDS",
                "window": 5,
                "nbdevup": 10,
                "nbdevdn": 5
            }
        ]

        expected = [
            "bbands_upper_5_10_5",
            "macdhist_5_10_6",
            "EMA_5",
            "macdsignal_5_10_6",
            "bbands_middle_5_10_5",
            "macd_5_10_6",
            "bbands_lower_5_10_5",
        ]

        assert sorted(deriveFeature(derive, self.data)) == sorted(expected)
        for df in self.data.values():
            assert set(expected).issubset(set(df.columns))

    def testBasicConfg(self):
        basic = ["open", "close", "return", "abc"]
        expected = ["open", "close", "return"]
        basicFeature(basic, self.data)
        for df in self.data.values():
            assert set(expected).issubset(set(df.columns))