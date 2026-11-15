import tempfile
from pathlib import Path
import unittest
from fetch_data import weather_frame, stock_frame, save

class FetchTests(unittest.TestCase):
    def test_weather_aggregate_is_end_of_day(self):
        obj={'daily':{'time':['2024-01-01'],'temperature_2m_mean':[20.],
            'relative_humidity_2m_mean':[60.],'precipitation_sum':[0.],'wind_speed_10m_max':[10.]}}
        row=weather_frame(obj).iloc[0]
        self.assertEqual(row.timestamp.hour,23)
        self.assertEqual(row.timestamp.second,59)
        self.assertEqual(row.temperature,20.)

    def test_quota_response_is_not_a_dataset(self):
        with self.assertRaises(ValueError):
            stock_frame({'Information':'API limit'},'2020-01-01','2025-12-31')

    def test_compact_history_cannot_claim_full_coverage(self):
        obj={'Time Series (Daily)':{'2025-12-31':{'1. open':'1','2. high':'2','3. low':'1','4. close':'2','5. volume':'10'}}}
        with self.assertRaises(ValueError):
            stock_frame(obj,'2020-01-01','2025-12-31')

    def test_snapshot_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            save(Path(folder),{'data.csv':b'a\n1\n'},{'source':'test'},False)
            with self.assertRaises(FileExistsError):
                save(Path(folder),{'data.csv':b'a\n2\n'},{'source':'test'},False)
            self.assertEqual((Path(folder)/'data.csv').read_bytes(),b'a\n1\n')

if __name__=='__main__': unittest.main()
