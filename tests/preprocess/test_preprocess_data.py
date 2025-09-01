import unittest
import pandas as pd

from datetime import time

from pvtools.preprocess.preprocess_data import delete_night_period, average_measurements

class MyTestCase(unittest.TestCase):

    def test_delete_night_period(self):
        data = {
            'time': pd.to_datetime([
                "2023-09-01 01:00:00+00:00",  # 03:00 UTC+2
                "2023-09-01 08:00:00+00:00",  # 06:00 UTC+2
                "2023-09-01 19:30:00+00:00",  # 21:30 UTC+2
            ]),
            'value': [10, 20, 30]
        }

        df = pd.DataFrame(data)

        result = delete_night_period(
            df=df,
            save_dir=None,
            start=time(3, 0),
            end=time(20, 0)
        )

        assert len(result) == 2
        assert all(result['value'].values == [20, 30])

    def test_average_measurments(self):
        result = average_measurements(
            df=self.df,
            save_dir=None,
        )

if __name__ == '__main__':
    unittest.main()